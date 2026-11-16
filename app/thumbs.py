from __future__ import annotations

import bisect
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("fallguard.thumbs")

SAFE_THUMB_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9]{3}_[a-z0-9_]+_[0-9a-f]{4}$")


def _sanitize_state_token(raw_state: Any) -> str:
    raw = str(getattr(raw_state, "value", raw_state) or "idle").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return cleaned[:24] or "idle"


class TimelineThumbnailStore:
    """
    Потокобезопасное хранилище поминутных мини-превью комнаты (320×180 JPEG, quality=74)
    для интерактивного скраббинга хроники в веб-интерфейсе.
    """

    def __init__(
        self,
        thumbs_dir: Path | str,
        periodic_interval_sec: float = 60.0,
        min_gap_sec: float = 5.0,
        retention_sec: float = 86400.0,
        max_thumbs: int = 1500,
        thumb_size: tuple[int, int] = (320, 180),
        jpeg_quality: int = 74,
    ) -> None:
        self._dir = Path(thumbs_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"

        self._periodic_interval_sec = max(1.0, float(periodic_interval_sec))
        self._min_gap_sec = max(0.0, float(min_gap_sec))
        self._retention_sec = max(60.0, float(retention_sec))
        self._max_thumbs = max(5, int(max_thumbs))
        self._thumb_w = max(64, int(thumb_size[0]))
        self._thumb_h = max(36, int(thumb_size[1]))
        self._jpeg_quality = max(30, min(95, int(jpeg_quality)))

        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._ts_keys: list[float] = []
        self._by_id: dict[str, dict[str, Any]] = {}

        # Сбрасываются при старте процесса, чтобы первый же валидный кадр записался мгновенно
        self._last_periodic_ts: float = 0.0
        self._last_any_ts: float = 0.0
        self._last_state: str = ""

        with self._lock:
            self._load_locked()

    @property
    def thumbs_dir(self) -> Path:
        return self._dir

    def _rebuild_lookup_locked(self) -> None:
        self._entries.sort(key=lambda item: (float(item["ts"]), str(item["id"])))
        self._ts_keys = [float(item["ts"]) for item in self._entries]
        self._by_id = {str(item["id"]): item for item in self._entries}

    def _load_locked(self) -> None:
        if not self._index_path.exists():
            return
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Не удалось прочитать индекс мини-превью %s: %s", self._index_path, exc)
            return

        items = raw.get("thumbs") if isinstance(raw, dict) else (raw if isinstance(raw, list) else None)
        if not isinstance(items, list):
            return

        root_resolved = self._dir.resolve()
        loaded: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            entry_id = str(item.get("id") or "").strip()
            if not entry_id or entry_id in seen_ids or not SAFE_THUMB_ID_RE.fullmatch(entry_id):
                continue
            try:
                ts_val = round(float(item["ts"]), 3)
            except (KeyError, TypeError, ValueError):
                continue
            jpg_path = (self._dir / f"{entry_id}.jpg").resolve()
            if jpg_path.parent != root_resolved or jpg_path.is_symlink() or not jpg_path.is_file():
                continue
            state_str = _sanitize_state_token(item.get("state") or "idle")
            seen_ids.add(entry_id)
            loaded.append({"id": entry_id, "ts": ts_val, "state": state_str})

        self._entries = loaded
        self._rebuild_lookup_locked()

    def _save_locked(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_ts": round(time.time(), 3),
                "thumbs": self._entries,
            }
            tmp_path = self._index_path.with_name(
                f"{self._index_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
            )
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._index_path)
        except OSError as exc:
            logger.warning("Ошибка сохранения индекса мини-превью %s: %s", self._index_path, exc)

    def _unlink_thumb_file(self, entry_id: str) -> None:
        if not SAFE_THUMB_ID_RE.fullmatch(entry_id):
            return
        target = self._dir / f"{entry_id}.jpg"
        try:
            if target.exists() and not target.is_symlink():
                target.unlink()
        except OSError:
            pass

    def _enforce_retention_locked(self, ref_ts: float) -> bool:
        cutoff = ref_ts - self._retention_sec
        changed = False

        if self._entries and float(self._entries[0]["ts"]) < cutoff:
            kept: list[dict[str, Any]] = []
            for item in self._entries:
                if float(item["ts"]) < cutoff:
                    self._unlink_thumb_file(str(item["id"]))
                    changed = True
                else:
                    kept.append(item)
            self._entries = kept

        if len(self._entries) > self._max_thumbs:
            overflow = len(self._entries) - self._max_thumbs
            to_drop = self._entries[:overflow]
            self._entries = self._entries[overflow:]
            for item in to_drop:
                self._unlink_thumb_file(str(item["id"]))
            changed = True

        if changed:
            self._rebuild_lookup_locked()
        return changed

    def _encode_thumb_jpeg(self, frame_bgr: np.ndarray) -> bytes | None:
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self._thumb_w, self._thumb_h):
            interp = cv2.INTER_AREA if (w >= self._thumb_w and h >= self._thumb_h) else cv2.INTER_LINEAR
            resized = cv2.resize(frame_bgr, (self._thumb_w, self._thumb_h), interpolation=interp)
        else:
            resized = frame_bgr
        ok, enc = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            return None
        return enc.tobytes()

    def maybe_record(
        self,
        frame_bgr: np.ndarray | None,
        state: str,
        now_ts: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Сохраняет мини-снимок комнаты сразу при старте, далее раз в 60 с,
        а также внеочередно при смене state (не чаще чем раз в min_gap_sec = 5 с).
        """
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.size == 0
        ):
            return None
        h, w = frame_bgr.shape[:2]
        if h < 16 or w < 16:
            return None

        ts = round(float(time.time() if now_ts is None else now_ts), 3)
        clean_state = _sanitize_state_token(state)

        with self._lock:
            is_first = self._last_periodic_ts == 0.0
            is_periodic = is_first or ((ts - self._last_periodic_ts) >= self._periodic_interval_sec)
            is_state_change = (
                (not is_first)
                and (clean_state != self._last_state)
                and ((ts - self._last_any_ts) >= self._min_gap_sec)
            )
            if not (is_periodic or is_state_change):
                return None

        jpeg_bytes = self._encode_thumb_jpeg(frame_bgr)
        if not jpeg_bytes:
            return None

        dt = datetime.fromtimestamp(max(0.0, ts))
        ms_part = int(round((ts % 1.0) * 1000.0)) % 1000
        entry_id = f"{dt.strftime('%Y%m%d_%H%M%S')}_{ms_part:03d}_{clean_state}_{secrets.token_hex(2)}"
        jpg_path = self._dir / f"{entry_id}.jpg"

        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp_jpg = self._dir / f".{entry_id}.jpg.tmp"
            tmp_jpg.write_bytes(jpeg_bytes)
            os.replace(tmp_jpg, jpg_path)
        except OSError as exc:
            logger.warning("Не удалось записать мини-превью %s: %s", jpg_path, exc)
            return None

        entry: dict[str, Any] = {
            "id": entry_id,
            "ts": ts,
            "state": clean_state,
        }

        with self._lock:
            if not self._ts_keys or ts >= self._ts_keys[-1]:
                self._entries.append(entry)
                self._ts_keys.append(ts)
            else:
                ins = bisect.bisect_right(self._ts_keys, ts)
                self._entries.insert(ins, entry)
                self._ts_keys.insert(ins, ts)
            self._by_id[entry_id] = entry

            self._last_any_ts = ts
            self._last_state = clean_state
            if is_periodic:
                self._last_periodic_ts = ts

            ref_ts = max(ts, self._ts_keys[-1]) if self._ts_keys else ts
            self._enforce_retention_locked(ref_ts)
            self._save_locked()

        return dict(entry)

    def list_index(self, since_ts: float = 0.0) -> list[dict[str, Any]]:
        """Возвращает отсортированный по возрастанию ts список метаданных мини-снимков."""
        cutoff = float(since_ts)
        with self._lock:
            if cutoff <= 0.0:
                return [dict(item) for item in self._entries]
            start_idx = bisect.bisect_left(self._ts_keys, cutoff)
            return [dict(item) for item in self._entries[start_idx:]]

    def get_by_id(self, entry_id: str) -> Path | None:
        """Безопасно возвращает путь к JPEG-файлу по entry_id с защитой от path traversal."""
        clean_id = (entry_id or "").strip()
        if (
            not clean_id
            or "/" in clean_id
            or "\\" in clean_id
            or ".." in clean_id
            or not SAFE_THUMB_ID_RE.fullmatch(clean_id)
        ):
            return None

        with self._lock:
            if clean_id not in self._by_id:
                return None

        try:
            root_resolved = self._dir.resolve()
            candidate = (self._dir / f"{clean_id}.jpg").resolve()
        except OSError:
            return None
        if candidate.parent != root_resolved or candidate.is_symlink() or not candidate.is_file():
            return None
        return candidate

    def get_nearest(
        self,
        target_ts: float,
        max_delta_sec: float = 900.0,
    ) -> tuple[Path, dict[str, Any]] | None:
        """Быстрый бинарный поиск ближайшего по времени мини-кадра в пределах max_delta_sec."""
        target = float(target_ts)
        limit = max(0.0, float(max_delta_sec))

        with self._lock:
            if not self._entries:
                return None
            pos = bisect.bisect_left(self._ts_keys, target)
            candidate_indices: list[int] = []
            if pos < len(self._entries):
                candidate_indices.append(pos)
            if pos > 0:
                candidate_indices.append(pos - 1)
            candidates = [
                (abs(float(self._entries[idx]["ts"]) - target), dict(self._entries[idx]))
                for idx in candidate_indices
            ]

        candidates.sort(key=lambda pair: pair[0])
        root_resolved = self._dir.resolve()
        for delta, meta in candidates:
            if delta > limit:
                continue
            entry_id = str(meta["id"])
            if not SAFE_THUMB_ID_RE.fullmatch(entry_id):
                continue
            path = (self._dir / f"{entry_id}.jpg").resolve()
            if path.parent == root_resolved and not path.is_symlink() and path.is_file():
                return path, meta
        return None
