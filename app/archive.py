from __future__ import annotations

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

from .state_machine import AlertEvent

logger = logging.getLogger("fallguard.archive")

_SAFE_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
_SAFE_STATE_SLUG_RE = re.compile(r"[^a-z0-9_]+")
_ALLOWED_EXTS = ("jpg", "mp4")


def _is_safe_clip_id(clip_id: object) -> bool:
    return isinstance(clip_id, str) and bool(_SAFE_CLIP_ID_RE.fullmatch(clip_id))


class ClipArchiveStore:
    """
    Потокобезопасное локальное хранилище тревожных снимков (.jpg) и видеоклипов (.mp4)
    с атомарным индексом data/clips/index.json и авторотацией по числу записей и объёму МБ.
    """

    def __init__(
        self,
        clips_dir: Path | str,
        max_clips: int = 60,
        max_total_mb: float = 250.0,
    ) -> None:
        self._clips_dir = Path(clips_dir)
        self._index_path = self._clips_dir / "index.json"
        self._max_clips = max(1, int(max_clips))
        self._max_total_mb = max(0.01, float(max_total_mb))
        self._max_total_bytes = int(self._max_total_mb * 1024.0 * 1024.0)

        self._lock = threading.RLock()
        # Храним записи от самых новых к самым старым
        self._entries: list[dict[str, Any]] = []
        self._load_locked()

    @property
    def clips_dir(self) -> Path:
        return self._clips_dir

    def _load_locked(self) -> None:
        with self._lock:
            if not self._index_path.exists():
                self._entries = []
                return
            try:
                raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Не удалось прочитать индекс архива клипов %s: %s", self._index_path, exc)
                self._entries = []
                return

            items = raw.get("clips") if isinstance(raw, dict) else (raw if isinstance(raw, list) else None)
            if not isinstance(items, list):
                self._entries = []
                return

            loaded: list[dict[str, Any]] = []
            resolved_dir = self._clips_dir.resolve()
            for item in items:
                if not isinstance(item, dict):
                    continue
                cid = item.get("id")
                if not _is_safe_clip_id(cid):
                    continue
                assert isinstance(cid, str)
                jpg_file = (resolved_dir / f"{cid}.jpg")
                mp4_file = (resolved_dir / f"{cid}.mp4")
                has_jpg = jpg_file.is_file()
                has_mp4 = mp4_file.is_file()
                if not has_jpg and not has_mp4:
                    continue
                size_bytes = (jpg_file.stat().st_size if has_jpg else 0) + (
                    mp4_file.stat().st_size if has_mp4 else 0
                )
                try:
                    ts = float(item.get("ts") or 0.0)
                except (TypeError, ValueError):
                    ts = 0.0
                dt = datetime.fromtimestamp(ts) if ts > 0 else datetime.now()
                loaded.append(
                    {
                        "id": cid,
                        "ts": round(ts, 3),
                        "time": str(item.get("time") or dt.strftime("%H:%M:%S")),
                        "date": str(item.get("date") or dt.strftime("%Y-%m-%d")),
                        "state": str(item.get("state") or "event"),
                        "severity": str(item.get("severity") or "warning"),
                        "title": str(item.get("title") or ""),
                        "description": str(item.get("description") or ""),
                        "has_jpg": has_jpg,
                        "has_mp4": has_mp4,
                        "jpg_url": f"/api/clips/{cid}.jpg" if has_jpg else None,
                        "mp4_url": f"/api/clips/{cid}.mp4" if has_mp4 else None,
                        "size_bytes": int(size_bytes),
                        "false_alarm": bool(item.get("false_alarm", False)),
                        "false_alarm_by": (
                            str(item["false_alarm_by"])
                            if item.get("false_alarm_by") is not None
                            else None
                        ),
                    }
                )
            loaded.sort(key=lambda e: float(e["ts"]), reverse=True)
            self._entries = loaded

    def _save_index_locked(self) -> None:
        try:
            self._clips_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_ts": round(time.time(), 3),
                "clips": self._entries,
            }
            tmp_path = self._index_path.with_name(
                f"{self._index_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
            )
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._index_path)
        except OSError as exc:
            logger.warning("Не удалось сохранить индекс архива %s: %s", self._index_path, exc)

    def _write_media_atomic(self, target_path: Path, content: bytes) -> None:
        self._clips_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(
            f"{target_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        tmp_path.write_bytes(content)
        os.replace(tmp_path, target_path)

    def _remove_files_for_id(self, clip_id: str) -> None:
        if not _is_safe_clip_id(clip_id):
            return
        for ext in _ALLOWED_EXTS:
            p = self._clips_dir / f"{clip_id}.{ext}"
            try:
                if p.exists() or p.is_symlink():
                    p.unlink()
            except OSError:
                pass

    def _compute_entry_size(self, clip_id: str) -> tuple[bool, bool, int]:
        jpg_p = self._clips_dir / f"{clip_id}.jpg"
        mp4_p = self._clips_dir / f"{clip_id}.mp4"
        has_jpg = jpg_p.is_file()
        has_mp4 = mp4_p.is_file()
        total = (jpg_p.stat().st_size if has_jpg else 0) + (mp4_p.stat().st_size if has_mp4 else 0)
        return has_jpg, has_mp4, int(total)

    def _total_bytes_locked(self) -> int:
        return sum(int(e.get("size_bytes", 0)) for e in self._entries)

    def _enforce_quota_locked(self) -> None:
        self._entries.sort(key=lambda e: float(e["ts"]), reverse=True)
        while len(self._entries) > self._max_clips or (
            len(self._entries) > 1 and self._total_bytes_locked() > self._max_total_bytes
        ):
            oldest = self._entries.pop()
            self._remove_files_for_id(str(oldest["id"]))

    def save_event_snapshot(
        self,
        event: AlertEvent,
        jpeg_bytes: bytes,
        unix_ts: float | None = None,
    ) -> dict[str, Any]:
        """Сохраняет JPEG-снимок тревоги, регистрирует запись в индексе и выполняет авторотацию."""
        ts = float(time.time() if unix_ts is None else unix_ts)
        dt = datetime.fromtimestamp(ts)
        raw_state = str(getattr(event.state, "value", event.state) or "event").strip().lower()
        state_slug = _SAFE_STATE_SLUG_RE.sub("", raw_state).strip("_") or "event"
        severity_str = str(getattr(event.severity, "value", event.severity) or "warning").strip().lower()

        with self._lock:
            clip_id = f"{dt.strftime('%Y%m%d_%H%M%S')}_{state_slug}_{secrets.token_hex(2)}"
            existing_ids = {str(e["id"]) for e in self._entries}
            while clip_id in existing_ids:
                clip_id = f"{dt.strftime('%Y%m%d_%H%M%S')}_{state_slug}_{secrets.token_hex(2)}"

            jpg_path = self._clips_dir / f"{clip_id}.jpg"
            self._write_media_atomic(jpg_path, jpeg_bytes)
            has_jpg, has_mp4, size_bytes = self._compute_entry_size(clip_id)

            entry: dict[str, Any] = {
                "id": clip_id,
                "ts": round(ts, 3),
                "time": dt.strftime("%H:%M:%S"),
                "date": dt.strftime("%Y-%m-%d"),
                "state": raw_state,
                "severity": severity_str,
                "title": str(event.title),
                "description": str(event.description),
                "has_jpg": has_jpg,
                "has_mp4": has_mp4,
                "jpg_url": f"/api/clips/{clip_id}.jpg" if has_jpg else None,
                "mp4_url": f"/api/clips/{clip_id}.mp4" if has_mp4 else None,
                "size_bytes": size_bytes,
                "false_alarm": False,
                "false_alarm_by": None,
            }
            self._entries.insert(0, entry)
            self._enforce_quota_locked()
            self._save_index_locked()
            return dict(entry)

    def mark_false_alarm(self, clip_id: str, reporter_name: str = "") -> dict[str, Any] | None:
        """Помечает архивную запись как ложную тревогу и сохраняет index.json."""
        if not _is_safe_clip_id(clip_id):
            return None
        clean_reporter = str(reporter_name or "").strip() or "Пользователь"
        with self._lock:
            for entry in self._entries:
                if entry.get("id") == clip_id:
                    entry["false_alarm"] = True
                    entry["false_alarm_by"] = clean_reporter
                    self._save_index_locked()
                    return dict(entry)
            return None

    def attach_mp4(self, clip_id: str, mp4_bytes: bytes) -> dict[str, Any] | None:
        """Прикрепляет склеенный MP4-клип к ранее сохранённому событию и обновляет квоту."""
        if not _is_safe_clip_id(clip_id) or not mp4_bytes:
            return None

        with self._lock:
            target_entry: dict[str, Any] | None = None
            for entry in self._entries:
                if entry.get("id") == clip_id:
                    target_entry = entry
                    break
            if target_entry is None:
                return None

            mp4_path = self._clips_dir / f"{clip_id}.mp4"
            self._write_media_atomic(mp4_path, mp4_bytes)
            has_jpg, has_mp4, size_bytes = self._compute_entry_size(clip_id)

            target_entry["has_jpg"] = has_jpg
            target_entry["has_mp4"] = has_mp4
            target_entry["jpg_url"] = f"/api/clips/{clip_id}.jpg" if has_jpg else None
            target_entry["mp4_url"] = f"/api/clips/{clip_id}.mp4" if has_mp4 else None
            target_entry["size_bytes"] = size_bytes

            self._enforce_quota_locked()
            self._save_index_locked()
            return dict(target_entry)

    def list_clips(self, limit: int = 50, state_filter: str | None = None) -> list[dict[str, Any]]:
        max_items = max(1, int(limit))
        norm_state = state_filter.strip().lower() if state_filter else None
        with self._lock:
            items = [
                dict(e)
                for e in self._entries
                if norm_state is None or str(e.get("state", "")).lower() == norm_state
            ]
            return items[:max_items]

    def get_clip(self, clip_id: str) -> dict[str, Any] | None:
        if not _is_safe_clip_id(clip_id):
            return None
        with self._lock:
            for e in self._entries:
                if e.get("id") == clip_id:
                    return dict(e)
            return None

    def resolve_media_path(self, clip_id: str, ext: str) -> Path | None:
        """Безопасно разрешает путь к .jpg/.mp4 файлу со строгой защитой от path traversal."""
        if not _is_safe_clip_id(clip_id) or not isinstance(ext, str):
            return None
        clean_ext = ext.strip().lower().lstrip(".")
        if clean_ext not in _ALLOWED_EXTS:
            return None
        try:
            base_dir = self._clips_dir.resolve()
            candidate = (base_dir / f"{clip_id}.{clean_ext}").resolve()
        except OSError:
            return None
        if candidate.parent != base_dir or candidate.is_symlink() or not candidate.is_file():
            return None
        return candidate

    def get_latest_mp4(self) -> tuple[bytes, str] | None:
        """Возвращает (mp4_bytes, человекочитаемая_подпись) для самого свежего видеоклипа."""
        with self._lock:
            for e in self._entries:
                if not e.get("has_mp4"):
                    continue
                cid = str(e["id"])
                mp4_path = self.resolve_media_path(cid, "mp4")
                if mp4_path is None:
                    continue
                try:
                    data = mp4_path.read_bytes()
                except OSError:
                    continue
                if data:
                    title = str(e.get("title") or e.get("state") or cid).strip()
                    stamp = f"{e.get('date', '')} {e.get('time', '')}".strip()
                    label = f"{title} ({stamp})" if stamp else title
                    return data, label
        return None

    def delete_clip(self, clip_id: str) -> bool:
        if not _is_safe_clip_id(clip_id):
            return False
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.get("id") != clip_id]
            removed_from_index = len(self._entries) < before
            self._remove_files_for_id(clip_id)
            if removed_from_index:
                self._save_index_locked()
            return removed_from_index

    def clear_all(self) -> int:
        with self._lock:
            removed_count = len(self._entries)
            for e in self._entries:
                self._remove_files_for_id(str(e["id"]))
            if self._clips_dir.exists():
                for p in self._clips_dir.iterdir():
                    if p.is_file() and not p.is_symlink() and p.suffix.lower() in (".jpg", ".mp4"):
                        if _is_safe_clip_id(p.stem):
                            try:
                                p.unlink()
                            except OSError:
                                pass
            self._entries = []
            self._save_index_locked()
            return removed_count

    def storage_stats(self) -> dict[str, Any]:
        with self._lock:
            used_bytes = self._total_bytes_locked()
            mp4_count = sum(1 for e in self._entries if e.get("has_mp4"))
            return {
                "count": len(self._entries),
                "mp4_count": mp4_count,
                "max_clips": self._max_clips,
                "used_bytes": used_bytes,
                "used_mb": round(used_bytes / (1024.0 * 1024.0), 2),
                "max_total_mb": round(self._max_total_mb, 1),
            }
