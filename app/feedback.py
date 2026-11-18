from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from .state_machine import AlertEvent

logger = logging.getLogger("fallguard.feedback")

_SAFE_ALERT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")


def is_safe_alert_id(alert_id: object) -> bool:
    return isinstance(alert_id, str) and bool(_SAFE_ALERT_ID_RE.fullmatch(alert_id))


class FalseAlertFeedbackStore:
    """
    Постоянное потокобезопасное хранилище ложных срабатываний для последующего улучшения алгоритмов:
    сохраняет пакеты data/feedback/<alert_id>/ (context.json, raw.jpg, annotated.jpg, clip.mp4)
    и атомарный индекс data/feedback/false_alerts.json независимо от авторотации ClipArchiveStore.
    """

    def __init__(
        self,
        feedback_dir: Path | str,
        max_recent: int = 64,
        max_saved: int = 200,
    ) -> None:
        self._feedback_dir = Path(feedback_dir)
        self._index_path = self._feedback_dir / "false_alerts.json"
        self._max_recent = max(1, int(max_recent))
        self._max_saved = max(1, int(max_saved))

        self._lock = threading.RLock()
        self._recent: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._entries: list[dict[str, Any]] = []
        self._load_locked()

    @property
    def feedback_dir(self) -> Path:
        return self._feedback_dir

    def _resolve_pkg_dir_locked(self, alert_id: str) -> Path | None:
        if not is_safe_alert_id(alert_id):
            return None
        try:
            self._feedback_dir.mkdir(parents=True, exist_ok=True)
            base_dir = self._feedback_dir.resolve()
            pkg_dir = (base_dir / alert_id).resolve()
        except OSError:
            return None
        if pkg_dir.parent != base_dir or pkg_dir.is_symlink():
            return None
        return pkg_dir

    def _write_bytes_atomic(self, target_path: Path, content: bytes) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(
            f"{target_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        tmp_path.write_bytes(content)
        os.replace(tmp_path, target_path)

    def _write_json_atomic(self, target_path: Path, payload: dict[str, Any]) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(
            f"{target_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, target_path)

    def _load_locked(self) -> None:
        with self._lock:
            if not self._index_path.exists():
                self._entries = []
                return
            try:
                raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Не удалось прочитать индекс ошибок %s: %s", self._index_path, exc)
                self._entries = []
                return

            items = (
                raw.get("items")
                if isinstance(raw, dict)
                else (raw if isinstance(raw, list) else None)
            )
            if not isinstance(items, list):
                self._entries = []
                return

            loaded: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            base_dir = self._feedback_dir.resolve()
            for item in items:
                if not isinstance(item, dict):
                    continue
                aid = item.get("alert_id")
                if not is_safe_alert_id(aid):
                    continue
                assert isinstance(aid, str)
                if aid in seen_ids:
                    continue
                pkg_dir = base_dir / aid
                if not pkg_dir.is_dir() or pkg_dir.is_symlink():
                    continue
                seen_ids.add(aid)

                has_raw = (pkg_dir / "raw.jpg").is_file()
                has_ann = (pkg_dir / "annotated.jpg").is_file()
                has_mp4 = (pkg_dir / "clip.mp4").is_file()
                try:
                    rep_ts = float(item.get("reported_ts") or 0.0)
                except (TypeError, ValueError):
                    rep_ts = 0.0
                dt = datetime.fromtimestamp(rep_ts) if rep_ts > 0 else datetime.now()
                raw_tid = item.get("patient_track_id")
                tid_val = int(raw_tid) if isinstance(raw_tid, int) else None

                loaded.append(
                    {
                        "alert_id": aid,
                        "event_ts": item.get("event_ts"),
                        "reported_ts": round(rep_ts, 3),
                        "reported_time": str(item.get("reported_time") or dt.strftime("%H:%M:%S")),
                        "reported_date": str(item.get("reported_date") or dt.strftime("%Y-%m-%d")),
                        "reporter_chat_id": str(item.get("reporter_chat_id") or ""),
                        "reporter_name": str(item.get("reporter_name") or "Пользователь"),
                        "state": str(item.get("state") or "event"),
                        "severity": str(item.get("severity") or "warning"),
                        "title": str(item.get("title") or ""),
                        "description": str(item.get("description") or ""),
                        "patient_track_id": tid_val,
                        "has_raw_jpg": has_raw,
                        "has_annotated_jpg": has_ann,
                        "has_mp4": has_mp4,
                    }
                )
            loaded.sort(key=lambda e: float(e.get("reported_ts") or 0.0), reverse=True)
            self._entries = loaded[: self._max_saved]

    def _save_index_locked(self) -> None:
        try:
            payload = {
                "updated_ts": round(time.time(), 3),
                "count": len(self._entries),
                "items": self._entries,
            }
            self._write_json_atomic(self._index_path, payload)
        except OSError as exc:
            logger.warning("Не удалось сохранить индекс ложных срабатываний %s: %s", self._index_path, exc)

    def _enforce_quota_locked(self) -> None:
        self._entries.sort(key=lambda e: float(e.get("reported_ts") or 0.0), reverse=True)
        while len(self._entries) > self._max_saved:
            oldest = self._entries.pop()
            old_id = str(oldest.get("alert_id") or "")
            pkg_dir = self._resolve_pkg_dir_locked(old_id)
            if pkg_dir is not None and pkg_dir.exists():
                shutil.rmtree(pkg_dir, ignore_errors=True)

    def register_recent_alert(
        self,
        alert_id: str,
        event: AlertEvent,
        annotated_jpeg: bytes,
        raw_jpeg: bytes = b"",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Регистрирует тревогу в кольцевом буфере памяти (последние max_recent событий)."""
        if not is_safe_alert_id(alert_id):
            return
        state_str = str(getattr(event.state, "value", event.state) or "event").strip().lower()
        severity_str = str(getattr(event.severity, "value", event.severity) or "warning").strip().lower()
        now_wall = round(time.time(), 3)

        with self._lock:
            prev = self._recent.pop(alert_id, None)
            prev_mp4 = (prev.get("mp4_bytes") if isinstance(prev, dict) else b"") or b""
            self._recent[alert_id] = {
                "alert_id": alert_id,
                "created_wall_ts": now_wall,
                "event_monotonic_ts": float(event.timestamp),
                "state": state_str,
                "severity": severity_str,
                "title": str(event.title),
                "description": str(event.description),
                "patient_track_id": event.patient_track_id,
                "annotated_jpeg": bytes(annotated_jpeg or b""),
                "raw_jpeg": bytes(raw_jpeg or b""),
                "mp4_bytes": bytes(prev_mp4),
                "context": dict(context or {}),
            }
            while len(self._recent) > self._max_recent:
                self._recent.popitem(last=False)

    def get_recent_alert(self, alert_id: str) -> dict[str, Any] | None:
        """Возвращает копию метаданных недавно сработавшей тревоги из кольца памяти."""
        if not is_safe_alert_id(alert_id):
            return None
        with self._lock:
            rec = self._recent.get(alert_id)
            if rec is None:
                return None
            return {
                "alert_id": rec["alert_id"],
                "created_wall_ts": rec["created_wall_ts"],
                "event_monotonic_ts": rec["event_monotonic_ts"],
                "state": rec["state"],
                "severity": rec["severity"],
                "title": rec["title"],
                "description": rec["description"],
                "patient_track_id": rec["patient_track_id"],
                "has_raw_jpg": bool(rec.get("raw_jpeg")),
                "has_annotated_jpg": bool(rec.get("annotated_jpeg")),
                "has_mp4": bool(rec.get("mp4_bytes")),
                "context": dict(rec.get("context") or {}),
            }

    def get_alert_raw_jpeg_and_context(self, alert_id: str) -> tuple[bytes, dict[str, Any]]:
        """
        Возвращает сырой кадр тревоги raw_jpeg и словарь контекста context (с telemetry_snapshot['tracks'])
        из кольцевого буфера памяти self._recent либо из сохранённого пакета data/feedback/<alert_id>/.
        """
        if not is_safe_alert_id(alert_id):
            return b"", {}
        with self._lock:
            rec = self._recent.get(alert_id)
            raw_jpeg = bytes((rec.get("raw_jpeg") if rec else b"") or b"")
            ctx: dict[str, Any] = dict((rec.get("context") if rec else None) or {})
            if rec is not None:
                if "patient_track_id" not in ctx and rec.get("patient_track_id") is not None:
                    ctx["patient_track_id"] = rec.get("patient_track_id")
                if "state" not in ctx and rec.get("state"):
                    ctx["state"] = rec.get("state")

            pkg_dir = self._resolve_pkg_dir_locked(alert_id)
            if pkg_dir is not None and pkg_dir.is_dir():
                if not raw_jpeg:
                    raw_path = pkg_dir / "raw.jpg"
                    if raw_path.is_file():
                        try:
                            raw_jpeg = raw_path.read_bytes()
                        except OSError:
                            raw_jpeg = b""
                ctx_path = pkg_dir / "context.json"
                if ctx_path.is_file():
                    try:
                        disk_ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                        if isinstance(disk_ctx, dict):
                            merged = dict(disk_ctx)
                            for k, v in ctx.items():
                                if v is not None and v != {} and v != "":
                                    merged[k] = v
                            ctx = merged
                    except Exception:
                        pass
            return raw_jpeg, ctx

    def mark_false_alert(
        self,
        alert_id: str,
        reporter_chat_id: str = "",
        reporter_name: str = "Пользователь",
        fallback_clip: dict[str, Any] | None = None,
        mp4_bytes: bytes | None = None,
        fallback_jpg: bytes | None = None,
        live_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Сохраняет полный диагностический пакет ложной тревоги в data/feedback/<alert_id>/
        и идемпотентно обновляет индекс data/feedback/false_alerts.json.
        """
        with self._lock:
            pkg_dir = self._resolve_pkg_dir_locked(alert_id)
            if pkg_dir is None:
                raise ValueError("Недопустимый идентификатор тревоги")
            pkg_dir.mkdir(parents=True, exist_ok=True)

            recent = self._recent.get(alert_id)
            existing_idx = next(
                (i for i, e in enumerate(self._entries) if e.get("alert_id") == alert_id),
                None,
            )
            existing = self._entries[existing_idx] if existing_idx is not None else None

            existing_ctx: dict[str, Any] = {}
            ctx_path = pkg_dir / "context.json"
            if ctx_path.is_file():
                try:
                    loaded_ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_ctx, dict):
                        existing_ctx = loaded_ctx
                except Exception:
                    existing_ctx = {}

            raw_bytes = bytes((recent.get("raw_jpeg") if recent else b"") or b"")
            if raw_bytes:
                self._write_bytes_atomic(pkg_dir / "raw.jpg", raw_bytes)

            ann_bytes = bytes(
                (recent.get("annotated_jpeg") if recent else b"")
                or (fallback_jpg or b"")
            )
            if ann_bytes and (recent is not None or not (pkg_dir / "annotated.jpg").is_file()):
                self._write_bytes_atomic(pkg_dir / "annotated.jpg", ann_bytes)

            vid_bytes = bytes(
                mp4_bytes
                or (recent.get("mp4_bytes") if recent else b"")
                or b""
            )
            if vid_bytes:
                self._write_bytes_atomic(pkg_dir / "clip.mp4", vid_bytes)

            has_raw_jpg = (pkg_dir / "raw.jpg").is_file()
            has_annotated_jpg = (pkg_dir / "annotated.jpg").is_file()
            has_mp4 = (pkg_dir / "clip.mp4").is_file()

            now_ts = round(time.time(), 3)
            now_dt = datetime.fromtimestamp(now_ts)
            clean_name = str(reporter_name or "").strip() or "Пользователь"
            clean_chat = str(reporter_chat_id or "").strip()

            if existing is not None:
                reported_ts = float(existing.get("reported_ts") or now_ts)
                reported_time = str(existing.get("reported_time") or now_dt.strftime("%H:%M:%S"))
                reported_date = str(existing.get("reported_date") or now_dt.strftime("%Y-%m-%d"))
                eff_chat = clean_chat or str(existing.get("reporter_chat_id") or "")
                eff_name = str(existing.get("reporter_name") or clean_name)
            else:
                reported_ts = now_ts
                reported_time = now_dt.strftime("%H:%M:%S")
                reported_date = now_dt.strftime("%Y-%m-%d")
                eff_chat = clean_chat
                eff_name = clean_name

            fb_clip = fallback_clip or {}
            rec_ctx = dict((recent.get("context") if recent else None) or {})
            live_ctx = dict(live_context or {})

            state_str = str(
                (recent.get("state") if recent else None)
                or fb_clip.get("state")
                or existing_ctx.get("state")
                or live_ctx.get("state")
                or "event"
            )
            severity_str = str(
                (recent.get("severity") if recent else None)
                or fb_clip.get("severity")
                or existing_ctx.get("severity")
                or "warning"
            )
            title_str = str(
                (recent.get("title") if recent else None)
                or fb_clip.get("title")
                or existing_ctx.get("title")
                or ""
            )
            desc_str = str(
                (recent.get("description") if recent else None)
                or fb_clip.get("description")
                or existing_ctx.get("description")
                or ""
            )
            patient_tid = (
                recent.get("patient_track_id")
                if (recent is not None and recent.get("patient_track_id") is not None)
                else existing_ctx.get("patient_track_id")
            )
            event_ts = (
                rec_ctx.get("event_wall_ts")
                or (recent.get("created_wall_ts") if recent else None)
                or fb_clip.get("ts")
                or existing_ctx.get("event_ts")
                or reported_ts
            )
            telemetry_snap = (
                rec_ctx.get("telemetry_snapshot")
                or existing_ctx.get("telemetry_snapshot")
                or live_ctx.get("telemetry_snapshot")
                or {}
            )
            baseline_bed = (
                rec_ctx.get("baseline_in_bed")
                or existing_ctx.get("baseline_in_bed")
                or live_ctx.get("baseline_in_bed")
                or {}
            )
            zones_dict = (
                rec_ctx.get("zones")
                or existing_ctx.get("zones")
                or live_ctx.get("zones")
                or {}
            )
            was_bed_latched = rec_ctx.get(
                "was_bed_latched",
                existing_ctx.get("was_bed_latched", live_ctx.get("was_bed_latched")),
            )

            context_payload: dict[str, Any] = {
                "alert_id": alert_id,
                "event_ts": round(float(event_ts), 3) if isinstance(event_ts, (int, float)) else now_ts,
                "reported_ts": round(reported_ts, 3),
                "reported_time": reported_time,
                "reported_date": reported_date,
                "reporter_chat_id": eff_chat,
                "reporter_name": eff_name,
                "state": state_str,
                "severity": severity_str,
                "title": title_str,
                "description": desc_str,
                "patient_track_id": int(patient_tid) if isinstance(patient_tid, int) else None,
                "telemetry_snapshot": telemetry_snap,
                "baseline_in_bed": baseline_bed,
                "zones": zones_dict,
                "was_bed_latched": bool(was_bed_latched) if was_bed_latched is not None else None,
                "has_raw_jpg": has_raw_jpg,
                "has_annotated_jpg": has_annotated_jpg,
                "has_mp4": has_mp4,
            }
            self._write_json_atomic(ctx_path, context_payload)

            entry: dict[str, Any] = {
                "alert_id": alert_id,
                "event_ts": context_payload["event_ts"],
                "reported_ts": context_payload["reported_ts"],
                "reported_time": reported_time,
                "reported_date": reported_date,
                "reporter_chat_id": eff_chat,
                "reporter_name": eff_name,
                "state": state_str,
                "severity": severity_str,
                "title": title_str,
                "description": desc_str,
                "patient_track_id": context_payload["patient_track_id"],
                "has_raw_jpg": has_raw_jpg,
                "has_annotated_jpg": has_annotated_jpg,
                "has_mp4": has_mp4,
            }
            if existing_idx is not None:
                self._entries[existing_idx] = entry
            else:
                self._entries.insert(0, entry)
                self._enforce_quota_locked()
            self._save_index_locked()
            return dict(entry)

    def attach_mp4_if_flagged(self, alert_id: str, mp4_bytes: bytes) -> bool:
        """
        Кэширует готовый MP4 в кольце recent, а если тревога уже была отмечена ложной
        на 1–3-й секунде до готовности видео — сразу дописывает clip.mp4 на диск и обновляет флаги.
        """
        if not is_safe_alert_id(alert_id) or not mp4_bytes:
            return False

        with self._lock:
            rec = self._recent.get(alert_id)
            if rec is not None:
                rec["mp4_bytes"] = bytes(mp4_bytes)

            target_entry: dict[str, Any] | None = None
            for entry in self._entries:
                if entry.get("alert_id") == alert_id:
                    target_entry = entry
                    break
            if target_entry is None:
                return False

            pkg_dir = self._resolve_pkg_dir_locked(alert_id)
            if pkg_dir is None:
                return False
            pkg_dir.mkdir(parents=True, exist_ok=True)
            self._write_bytes_atomic(pkg_dir / "clip.mp4", bytes(mp4_bytes))
            target_entry["has_mp4"] = True

            ctx_path = pkg_dir / "context.json"
            if ctx_path.is_file():
                try:
                    ctx_data = json.loads(ctx_path.read_text(encoding="utf-8"))
                    if isinstance(ctx_data, dict):
                        ctx_data["has_mp4"] = True
                        self._write_json_atomic(ctx_path, ctx_data)
                except Exception as exc:
                    logger.warning("Не удалось обновить context.json для %s: %s", alert_id, exc)

            self._save_index_locked()
            return True

    def list_feedback(self, limit: int = 50) -> list[dict[str, Any]]:
        max_items = max(1, int(limit))
        with self._lock:
            return [dict(e) for e in self._entries[:max_items]]

    def false_alert_ids(self) -> set[str]:
        """Идентификаторы тревог, отмеченных семьёй как ложные (для отчёта врачу)."""
        with self._lock:
            return {str(e.get("alert_id")) for e in self._entries if e.get("alert_id")}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            with_mp4 = sum(1 for e in self._entries if e.get("has_mp4"))
            return {
                "count": len(self._entries),
                "with_mp4": with_mp4,
            }
