from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

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


def _structure(gray_small: np.ndarray) -> np.ndarray:
    """Карта контуров комнаты (модуль градиента), нормированная — не зависит от яркости и ИК/дня."""
    g = cv2.GaussianBlur(gray_small.astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag -= float(mag.mean())
    std = float(mag.std())
    return mag / std if std > 1e-6 else mag


class CameraHealthMonitor:
    """
    Замечает то, что не видно по «кадры приходят»: застывшее изображение (поток жив, картинка стоит)
    и сдвиг камеры (её задели — разметка кровати больше не совпадает с комнатой).
    Эталоны вида комнаты хранятся отдельно для дня и ночного ИК и переживают перезапуск.
    """

    FROZEN_AFTER_SEC = 90.0
    CHECK_EVERY_SEC = 20.0
    CONFIRM_CHECKS = 3
    REF_WARMUP_SEC = 30.0
    SHIFT_LIMIT_PX = 6.0  # ~4% ширины кадра
    MIN_CORRELATION = 0.35

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
        self._load_refs()

    # ------------------------------------------------------------------ состояние
    @property
    def issue(self) -> str:
        with self._lock:
            return "frozen" if self._frozen else ("moved" if self._moved else "")

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            dx, dy, corr = self._last_shift
            return {
                "issue": "frozen" if self._frozen else ("moved" if self._moved else ""),
                "shift_px": round(float(np.hypot(dx, dy)), 1),
                "correlation": round(corr, 3),
                "has_reference": sorted(self._refs.keys()),
            }

    def reset_reference(self) -> None:
        """Разметку перерисовали или поменяли поворот кадра: текущий вид — новый эталон."""
        with self._lock:
            self._refs.clear()
            self._mode_since.clear()
            self._bad_checks = self._good_checks = 0
            self._moved = False
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
    def update(self, frame_bgr: np.ndarray | None, now: float, night: bool) -> list[CameraIssueChange]:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return []
        small = _gray_small(frame_bgr)
        changes: list[CameraIssueChange] = []
        with self._lock:
            # 1. Застывшее изображение: ни один пиксель уменьшенного кадра не меняется
            if self._prev_small is not None and small.shape == self._prev_small.shape:
                identical = int(cv2.absdiff(small, self._prev_small).max()) == 0
                if identical:
                    if self._still_since is None:
                        self._still_since = now
                    if not self._frozen and (now - self._still_since) >= self.FROZEN_AFTER_SEC:
                        self._frozen = True
                        changes.append(CameraIssueChange("frozen", True, "Изображение с камеры не меняется больше 1,5 мин"))
                else:
                    self._still_since = None
                    if self._frozen:
                        self._frozen = False
                        changes.append(CameraIssueChange("frozen", False))
            self._prev_small = small
            if self._frozen:
                return changes

            # 2. Сдвиг камеры — раз в 20 с сравниваем контуры комнаты с эталоном
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

            win = cv2.createHanningWindow((_W, _H), cv2.CV_32F)
            (dx, dy), _resp = cv2.phaseCorrelate(ref, cur, win)
            corr = float(np.corrcoef(ref.ravel(), cur.ravel())[0, 1])
            self._last_shift = (float(dx), float(dy), corr)
            shifted = float(np.hypot(dx, dy)) >= self.SHIFT_LIMIT_PX
            bad = shifted or corr < self.MIN_CORRELATION

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
                # Медленно подстраиваем эталон под освещение и мелкие перестановки
                self._refs[mode] = (0.97 * ref + 0.03 * cur).astype(np.float32)
                if self._good_checks % 15 == 0:
                    self._save_ref(mode)
                if self._moved and self._good_checks >= self.CONFIRM_CHECKS:
                    self._moved = False
                    changes.append(CameraIssueChange("moved", False))
        return changes
