from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("fallguard.camera_health")

_W, _H = 160, 90


@dataclass(frozen=True)
class CameraIssueChange:
    kind: str  # "frozen" | "moved"
    active: bool
    detail: str = ""


def _gray_small(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    return cv2.resize(gray, (_W, _H), interpolation=cv2.INTER_AREA)


def _structure(gray_small: np.ndarray, static_mask: np.ndarray | None = None) -> np.ndarray:
    """Карта контуров комнаты (модуль градиента), нормированная — не зависит от яркости и ИК/дня."""
    g = cv2.GaussianBlur(gray_small.astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if static_mask is not None:
        return _apply_static_mask(mag, static_mask)
    mag -= float(mag.mean())
    std = float(mag.std())
    return mag / std if std > 1e-6 else mag


def _apply_static_mask(struct_map: np.ndarray, static_mask: np.ndarray | None) -> np.ndarray:
    """Нормирует структурную карту строго по неподвижному фону комнаты и обнуляет динамические зоны."""
    if static_mask is None or struct_map.shape != static_mask.shape:
        mu = float(struct_map.mean())
        std = float(struct_map.std())
        out_full = struct_map.astype(np.float32) - mu
        return out_full / std if std > 1e-6 else out_full
    vals = struct_map[static_mask]
    if vals.size == 0:
        return np.zeros_like(struct_map, dtype=np.float32)
    mu = float(vals.mean())
    std = float(vals.std())
    out = np.zeros_like(struct_map, dtype=np.float32)
    if std > 1e-6:
        out[static_mask] = (vals - mu) / std
    return out


def _normalize_poly_key(
    poly: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, float], ...]:
    if not poly or len(poly) < 3:
        return ()
    pts: list[tuple[float, float]] = []
    for p in poly:
        if len(p) >= 2:
            pts.append((round(float(p[0]), 5), round(float(p[1]), 5)))
    return tuple(pts) if len(pts) >= 3 else ()


def _build_static_background_mask(
    bed_key: tuple[tuple[float, float], ...],
    door_key: tuple[tuple[float, float], ...],
    ceiling_top_ratio: float = 0.08,
    min_static_ratio: float = 0.25,
) -> np.ndarray | None:
    """
    Строит булеву маску неподвижных стен/мебели комнаты (160x90):
    исключает верхнюю полосу засвета потолочной люстры (y < 0.08), внутренность bed_polygon и door_polygon.
    Если статическая площадь < 25% кадра — безопасно откатывается на полный кадр (None).
    """
    mask = np.ones((_H, _W), dtype=bool)
    top_rows = max(0, min(_H, int(round(ceiling_top_ratio * _H))))
    if top_rows > 0:
        mask[:top_rows, :] = False

    for poly_key in (bed_key, door_key):
        if len(poly_key) < 3:
            continue
        canvas = np.zeros((_H, _W), dtype=np.uint8)
        pts = np.array(
            [[int(round(px * _W)), int(round(py * _H))] for px, py in poly_key],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pts], 1)
        mask[canvas > 0] = False

    if float(mask.mean()) < min_static_ratio:
        return None
    return mask


class CameraHealthMonitor:
    """
    Замечает то, что не видно по «кадры приходят»: застывшее изображение (поток жив, картинка стоит)
    и сдвиг камеры (её задели — разметка кровати больше не совпадает с комнатой).
    Сравнение ведётся строго по статическому фону комнаты (вне кровати, двери и потолочного плафона),
    эталоны хранятся отдельно для дня и ночного ИК и переживают перезапуск.
    """

    FROZEN_AFTER_SEC = 90.0
    CHECK_EVERY_SEC = 20.0
    CONFIRM_CHECKS = 3
    REF_WARMUP_SEC = 30.0
    SHIFT_LIMIT_PX = 6.0  # ~4% ширины кадра
    STEADY_SHIFT_PX = 2.5
    MIN_CORRELATION = 0.16
    MIN_CORRELATION_STEADY = 0.16
    MIN_CORRELATION_MEDIUM = 0.25
    CEILING_GLARE_TOP_RATIO = 0.08
    MIN_STATIC_RATIO = 0.25
    REF_ADAPT_ALPHA = 0.05

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._dir = Path(data_dir) if data_dir is not None else None
        self._lock = threading.Lock()
        self._prev_small: np.ndarray | None = None
        self._still_since: float | None = None
        self._frozen = False
        self._refs: dict[str, np.ndarray] = {}
        self._mode_since: dict[str, float] = {}
        self._last_check = -1e9
        self._bad_checks = 0
        self._good_checks = 0
        self._moved = False
        self._last_shift: tuple[float, float, float] = (0.0, 0.0, 1.0)
        self._poly_keys: tuple[
            tuple[tuple[float, float], ...],
            tuple[tuple[float, float], ...],
        ] = ((), ())
        self._static_mask: np.ndarray | None = _build_static_background_mask(
            (),
            (),
            ceiling_top_ratio=self.CEILING_GLARE_TOP_RATIO,
            min_static_ratio=self.MIN_STATIC_RATIO,
        )
        self._hanning_win = cv2.createHanningWindow((_W, _H), cv2.CV_32F)
        self._load_refs()

    # ------------------------------------------------------------------ разметка фона
    def _set_dynamic_polygons_unlocked(
        self,
        bed_polygon: Sequence[Sequence[float]] | None = None,
        door_polygon: Sequence[Sequence[float]] | None = None,
    ) -> None:
        new_keys = (_normalize_poly_key(bed_polygon), _normalize_poly_key(door_polygon))
        if new_keys == self._poly_keys:
            return
        self._poly_keys = new_keys
        self._static_mask = _build_static_background_mask(
            new_keys[0],
            new_keys[1],
            ceiling_top_ratio=self.CEILING_GLARE_TOP_RATIO,
            min_static_ratio=self.MIN_STATIC_RATIO,
        )

    def set_dynamic_polygons(
        self,
        bed_polygon: Sequence[Sequence[float]] | None = None,
        door_polygon: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Исключает внутренность кровати и двери (плюс верхнюю полосу y < 0.08) из проверки сдвига камеры."""
        with self._lock:
            self._set_dynamic_polygons_unlocked(bed_polygon=bed_polygon, door_polygon=door_polygon)

    def set_zones(self, zones: Any) -> None:
        """Обновляет динамические полигоны из объекта ZonesConfig или словаря разметки."""
        if zones is None:
            return
        if isinstance(zones, dict):
            bed_p = zones.get("bed_polygon")
            door_p = zones.get("door_polygon")
        else:
            bed_p = getattr(zones, "bed_polygon", None)
            door_p = getattr(zones, "door_polygon", None)
        self.set_dynamic_polygons(bed_polygon=bed_p, door_polygon=door_p)

    # ------------------------------------------------------------------ состояние
    @property
    def issue(self) -> str:
        with self._lock:
            return "frozen" if self._frozen else ("moved" if self._moved else "")

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            dx, dy, corr = self._last_shift
            static_ratio = (
                round(float(self._static_mask.mean()), 3)
                if self._static_mask is not None
                else 1.0
            )
            return {
                "issue": "frozen" if self._frozen else ("moved" if self._moved else ""),
                "shift_px": round(float(np.hypot(dx, dy)), 1),
                "correlation": round(corr, 3),
                "has_reference": sorted(self._refs.keys()),
                "static_ratio": static_ratio,
            }

    def reset_reference(self) -> None:
        """Разметку перерисовали или поменяли поворот кадра: текущий вид — новый эталон."""
        with self._lock:
            self._refs.clear()
            self._mode_since.clear()
            self._bad_checks = self._good_checks = 0
            self._moved = False
            self._last_shift = (0.0, 0.0, 1.0)
            if self._dir is not None:
                for mode in ("day", "night"):
                    try:
                        (self._dir / f"camera_ref_{mode}.npy").unlink(missing_ok=True)
                    except OSError:
                        pass

    # ------------------------------------------------------------------ хранение
    def _load_refs(self) -> None:
        if self._dir is None:
            return
        for mode in ("day", "night"):
            path = self._dir / f"camera_ref_{mode}.npy"
            try:
                if path.exists():
                    arr = np.load(path)
                    if arr.shape == (_H, _W):
                        self._refs[mode] = arr.astype(np.float32)
            except Exception as exc:
                logger.warning("Не удалось прочитать эталон камеры %s: %s", path, exc)

    def _save_ref(self, mode: str) -> None:
        if self._dir is None or mode not in self._refs:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            np.save(self._dir / f"camera_ref_{mode}.npy", self._refs[mode])
        except OSError as exc:
            logger.warning("Не удалось сохранить эталон камеры: %s", exc)

    # ------------------------------------------------------------------ анализ
    def update(
        self,
        frame_bgr: np.ndarray | None,
        now: float,
        night: bool,
        zones: Any = None,
    ) -> list[CameraIssueChange]:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return []
        small = _gray_small(frame_bgr)
        changes: list[CameraIssueChange] = []
        with self._lock:
            if zones is not None:
                if isinstance(zones, dict):
                    self._set_dynamic_polygons_unlocked(
                        zones.get("bed_polygon"),
                        zones.get("door_polygon"),
                    )
                else:
                    self._set_dynamic_polygons_unlocked(
                        getattr(zones, "bed_polygon", None),
                        getattr(zones, "door_polygon", None),
                    )

            # 1. Застывшее изображение: ни один пиксель уменьшенного кадра не меняется
            if self._prev_small is not None and small.shape == self._prev_small.shape:
                identical = int(cv2.absdiff(small, self._prev_small).max()) == 0
                if identical:
                    if self._still_since is None:
                        self._still_since = now
                    if not self._frozen and (now - self._still_since) >= self.FROZEN_AFTER_SEC:
                        self._frozen = True
                        changes.append(
                            CameraIssueChange(
                                "frozen",
                                True,
                                "Изображение с камеры не меняется больше 1,5 мин",
                            )
                        )
                else:
                    self._still_since = None
                    if self._frozen:
                        self._frozen = False
                        changes.append(CameraIssueChange("frozen", False))
            self._prev_small = small
            if self._frozen:
                return changes

            # 2. Сдвиг камеры — раз в 20 с сравниваем контуры статического фона комнаты с эталоном
            mode = "night" if night else "day"
            self._mode_since.setdefault(mode, now)
            if (now - self._last_check) < self.CHECK_EVERY_SEC:
                return changes
            self._last_check = now
            cur = _structure(small)
            ref = self._refs.get(mode)
            if ref is None:
                if (now - self._mode_since[mode]) >= self.REF_WARMUP_SEC:
                    self._refs[mode] = cur
                    self._save_ref(mode)
                return changes

            mask = self._static_mask
            ref_m = _apply_static_mask(ref, mask)
            cur_m = _apply_static_mask(cur, mask)
            (dx, dy), _resp = cv2.phaseCorrelate(ref_m, cur_m, self._hanning_win)
            a_vec = ref_m[mask] if mask is not None else ref_m.ravel()
            b_vec = cur_m[mask] if mask is not None else cur_m.ravel()
            if float(a_vec.std()) <= 1e-6 or float(b_vec.std()) <= 1e-6:
                corr = 0.0
            else:
                raw_corr = float(np.corrcoef(a_vec, b_vec)[0, 1])
                corr = 0.0 if np.isnan(raw_corr) else raw_corr

            shift_px = float(np.hypot(dx, dy))
            self._last_shift = (float(dx), float(dy), corr)
            shifted = shift_px >= self.SHIFT_LIMIT_PX
            if shifted:
                bad = True
            elif shift_px < self.STEADY_SHIFT_PX:
                bad = corr < self.MIN_CORRELATION_STEADY
            else:
                bad = corr < self.MIN_CORRELATION_MEDIUM

            if bad:
                self._bad_checks += 1
                self._good_checks = 0
                if not self._moved and self._bad_checks >= self.CONFIRM_CHECKS:
                    self._moved = True
                    changes.append(
                        CameraIssueChange(
                            "moved",
                            True,
                            "Вид комнаты изменился — камеру, похоже, сдвинули. Разметка кровати может не совпадать.",
                        )
                    )
            else:
                self._good_checks += 1
                self._bad_checks = 0
                # Плавно подстраиваем эталон под переход освещения (окно -> люстра)
                alpha = float(self.REF_ADAPT_ALPHA)
                self._refs[mode] = ((1.0 - alpha) * ref + alpha * cur).astype(np.float32)
                if self._good_checks % 15 == 0:
                    self._save_ref(mode)
                if self._moved and self._good_checks >= self.CONFIRM_CHECKS:
                    self._moved = False
                    changes.append(CameraIssueChange("moved", False))
        return changes
