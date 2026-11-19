from __future__ import annotations

import json
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Callable

from .capture import CREATE_NO_WINDOW, find_ffmpeg_bin
from .stream_hub import H264StreamHub, StreamFragment, rebase_moof_tfdt

logger = logging.getLogger("fallguard.dvr")

_SAFE_SEG_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,80}$")


@dataclass
class _OpenSegment:
    seg_id: str
    tier: str
    date_str: str
    rel_path: str
    abs_path: Path
    start_ts: float
    end_ts: float
    start_mono: float
    last_frag_mono: float
    width: int
    height: int
    rotate: int
    flip_h: bool
    fh: IO[bytes]
    base_tfdt: int | None
    last_seq: int
    size_bytes: int
    last_flush_mono: float


class DvrRecorderStore:
    """
    7-дневный адаптивный видеоархив (DVR):
    - Слой «low» (360p): непрерывная запись 24/7 сегментами по ~300 с с нулевой нагрузкой (-c:v copy);
    - Слой «hi» (1080p): адаптивная запись при движении/людях/тревогах с предсобытийным GOP-буфером;
    - Автоматическое удержание по сроку (retention_days), объёму (max_gb) и остатку на диске (min_free_gb).
    """

    FLUSH_INDEX_INTERVAL_SEC: float = 20.0
    RETENTION_CHECK_INTERVAL_SEC: float = 60.0

    def __init__(
        self,
        dvr_dir: Path | str,
        segment_seconds: int = 300,
        retention_days: float = 7.0,
        max_gb: float = 120.0,
        min_free_gb: float = 5.0,
        motion_hold_seconds: float = 15.0,
        motion_log: Any | None = None,
        get_camera_transform: Callable[[], dict[str, Any]] | None = None,
        ffmpeg_bin: str | None = None,
        enable_remux: bool = True,
        disk_free_fn: Callable[[Path], int] | None = None,
    ) -> None:
        self.dvr_dir = Path(dvr_dir)
        self.segments_dir = self.dvr_dir / "segments"
        self.index_path = self.dvr_dir / "dvr_index.json"

        self.segment_seconds = max(10, int(segment_seconds))
        self.retention_days = max(0.5, float(retention_days))
        self.max_bytes = int(max(0.01, float(max_gb)) * (1024 ** 3))
        self.min_free_bytes = int(max(0.0, float(min_free_gb)) * (1024 ** 3))
        self.motion_hold_seconds = max(3.0, float(motion_hold_seconds))

        self.motion_log = motion_log
        self._get_camera_transform = get_camera_transform
        self._explicit_ffmpeg = ffmpeg_bin
        self._enable_remux = bool(enable_remux)
        self._disk_free_fn = disk_free_fn

        self._lock = threading.RLock()
        self._segments: list[dict[str, Any]] = []
        self._open_segments: dict[str, _OpenSegment] = {}
        self._total_bytes: int = 0

        self._motion_until_mono: float = 0.0
        self._motion_until_wall: float = 0.0
        self._last_index_save_mono: float = 0.0
        self._last_retention_mono: float = 0.0

        self._low_hub: H264StreamHub | None = None
        self._hi_hub: H264StreamHub | None = None
        self._low_sink: Any = None
        self._hi_sink: Any = None

        self._stop_event = threading.Event()
        self._remux_queue: queue.Queue[str | None] = queue.Queue(maxsize=48)
        self._remux_thread: threading.Thread | None = None
        self._housekeep_thread: threading.Thread | None = None
        self._active_remux_proc: subprocess.Popen[bytes] | None = None
        self._started = False

        self._load_index_locked()

    def _current_cam_meta(self) -> tuple[int, bool]:
        if self._get_camera_transform is None:
            return 0, False
        try:
            raw = self._get_camera_transform() or {}
            rot = int(raw.get("rotate", 0) or 0) % 360
            flip = bool(raw.get("flip_h", False))
            return rot, flip
        except Exception:
            return 0, False

    def _load_index_locked(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Не удалось прочитать индекс DVR %s: %s", self.index_path, exc)
            return
        items = raw.get("segments") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return
        loaded: list[dict[str, Any]] = []
        total_b = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or "")
            rel = str(item.get("rel_path") or "")
            if not _SAFE_SEG_ID_RE.match(sid) or not rel:
                continue
            fpath = (self.segments_dir / rel).resolve()
            try:
                if not fpath.is_file():
                    continue
                fsize = int(fpath.stat().st_size)
            except OSError:
                continue
            if fsize <= 0:
                continue
            st = float(item.get("start_ts") or 0.0)
            en = max(st, float(item.get("end_ts") or st))
            rec = {
                "id": sid,
                "rel_path": rel,
                "tier": "hi" if str(item.get("tier")) == "hi" else "low",
                "date": str(item.get("date") or datetime.fromtimestamp(st).strftime("%Y-%m-%d")),
                "start_ts": round(st, 3),
                "end_ts": round(en, 3),
                "width": int(item.get("width") or 0),
                "height": int(item.get("height") or 0),
                "rotate": int(item.get("rotate") or 0),
                "flip_h": bool(item.get("flip_h", False)),
                "size_bytes": fsize,
                "faststart": bool(item.get("faststart", False)),
            }
            loaded.append(rec)
            total_b += fsize

        loaded.sort(key=lambda s: (float(s["start_ts"]), 0 if s["tier"] == "hi" else 1))
        self._segments = loaded
        self._total_bytes = total_b

    def _save_index_locked(self) -> None:
        try:
            self.dvr_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": 1,
                "updated_ts": round(time.time(), 3),
                "segments": list(self._segments),
            }
            tmp_path = self.index_path.with_name(
                f"{self.index_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
            )
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, self.index_path)
            self._last_index_save_mono = time.monotonic()
        except OSError as exc:
            logger.warning("Не удалось сохранить индекс DVR %s: %s", self.index_path, exc)

    def attach_hubs(
        self,
        low_hub: H264StreamHub | None,
        hi_hub: H264StreamHub | None,
    ) -> None:
        with self._lock:
            self.detach_hubs()
            self._low_hub = low_hub
            self._hi_hub = hi_hub
            if low_hub is not None:
                self._low_sink = lambda init_b, frag, gop, h=low_hub: self.on_fragment(
                    "low", init_b, frag, gop, width=h.width, height=h.height
                )
                low_hub.add_fragment_sink(self._low_sink)
            if hi_hub is not None:
                self._hi_sink = lambda init_b, frag, gop, h=hi_hub: self.on_fragment(
                    "hi", init_b, frag, gop, width=h.width, height=h.height
                )
                hi_hub.add_fragment_sink(self._hi_sink)

    def detach_hubs(self) -> None:
        with self._lock:
            if self._low_hub is not None and self._low_sink is not None:
                self._low_hub.remove_fragment_sink(self._low_sink)
            if self._hi_hub is not None and self._hi_sink is not None:
                self._hi_hub.remove_fragment_sink(self._hi_sink)
            self._low_sink = None
            self._hi_sink = None

    def start(
        self,
        low_hub: H264StreamHub | None = None,
        hi_hub: H264StreamHub | None = None,
    ) -> None:
        self.dvr_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        if low_hub is not None or hi_hub is not None:
            self.attach_hubs(low_hub, hi_hub)

        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
            if self._enable_remux:
                self._remux_thread = threading.Thread(
                    target=self._remux_loop,
                    name="dvr-remux",
                    daemon=True,
                )
                self._remux_thread.start()
            self._housekeep_thread = threading.Thread(
                target=self._housekeep_loop,
                name="dvr-housekeep",
                daemon=True,
            )
            self._housekeep_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.detach_hubs()
        with self._lock:
            for tier in list(self._open_segments.keys()):
                self._close_open_segment_locked(tier, enqueue_remux=False)
            self._save_index_locked()

        if self._enable_remux:
            try:
                self._remux_queue.put_nowait(None)
            except queue.Full:
                pass
        proc = self._active_remux_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        for th in (self._remux_thread, self._housekeep_thread):
            if th and th.is_alive():
                th.join(timeout=2.5)
        self._started = False

    def notify_activity(self, wall_ts: float | None = None, hold_sec: float | None = None) -> None:
        hold = float(self.motion_hold_seconds if hold_sec is None else hold_sec)
        now_mono = time.monotonic()
        now_wall = float(time.time() if wall_ts is None else wall_ts)
        with self._lock:
            self._motion_until_mono = max(self._motion_until_mono, now_mono + hold)
            self._motion_until_wall = max(self._motion_until_wall, now_wall + hold)

    def _is_hi_active_locked(self, now_mono: float, now_wall: float) -> bool:
        return now_mono <= self._motion_until_mono or now_wall <= self._motion_until_wall

    def _open_new_segment_locked(
        self,
        tier: str,
        init_bytes: bytes,
        first_frags: list[StreamFragment],
        wall_now: float,
        mono_now: float,
        width: int,
        height: int,
    ) -> _OpenSegment | None:
        if not init_bytes or not first_frags:
            return None
        dt = datetime.fromtimestamp(wall_now)
        date_str = dt.strftime("%Y-%m-%d")
        seg_id = f"{tier}_{dt.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        rel_path = f"{date_str}/{seg_id}.mp4"
        abs_path = self.segments_dir / date_str / f"{seg_id}.mp4"

        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(abs_path, "wb", buffering=262144)
        except OSError as exc:
            logger.warning("Не удалось создать сегмент DVR %s: %s", abs_path, exc)
            return None

        rot, flip_h = self._current_cam_meta()
        base_tfdt: int | None = None
        total_written = 0
        last_seq = -1

        try:
            fh.write(init_bytes)
            total_written += len(init_bytes)
            for fr in first_frags:
                if fr.seq <= last_seq:
                    continue
                patched, eff_base = rebase_moof_tfdt(fr.data, base_tfdt)
                if base_tfdt is None:
                    base_tfdt = eff_base
                fh.write(patched)
                total_written += len(patched)
                last_seq = fr.seq
            fh.flush()
        except OSError as exc:
            logger.warning("Ошибка записи заголовка сегмента DVR %s: %s", abs_path, exc)
            try:
                fh.close()
            except Exception:
                pass
            return None

        seg = _OpenSegment(
            seg_id=seg_id,
            tier=tier,
            date_str=date_str,
            rel_path=rel_path,
            abs_path=abs_path,
            start_ts=wall_now,
            end_ts=wall_now,
            start_mono=mono_now,
            last_frag_mono=mono_now,
            width=int(width),
            height=int(height),
            rotate=rot,
            flip_h=flip_h,
            fh=fh,
            base_tfdt=base_tfdt,
            last_seq=last_seq,
            size_bytes=total_written,
            last_flush_mono=mono_now,
        )
        self._open_segments[tier] = seg
        return seg

    def _close_open_segment_locked(self, tier: str, enqueue_remux: bool = True) -> dict[str, Any] | None:
        seg = self._open_segments.pop(tier, None)
        if seg is None:
            return None
        try:
            seg.fh.flush()
            seg.fh.close()
        except Exception:
            pass

        try:
            final_size = int(seg.abs_path.stat().st_size) if seg.abs_path.is_file() else seg.size_bytes
        except OSError:
            final_size = seg.size_bytes

        if final_size <= 0:
            try:
                if seg.abs_path.is_file():
                    seg.abs_path.unlink()
            except OSError:
                pass
            return None

        entry = {
            "id": seg.seg_id,
            "rel_path": seg.rel_path,
            "tier": seg.tier,
            "date": seg.date_str,
            "start_ts": round(seg.start_ts, 3),
            "end_ts": round(max(seg.start_ts + 0.2, seg.end_ts), 3),
            "width": seg.width,
            "height": seg.height,
            "rotate": seg.rotate,
            "flip_h": seg.flip_h,
            "size_bytes": final_size,
            "faststart": False,
        }
        self._segments.append(entry)
        self._segments.sort(key=lambda s: (float(s["start_ts"]), 0 if s["tier"] == "hi" else 1))
        self._total_bytes += final_size
        self._save_index_locked()

        if enqueue_remux and self._enable_remux and not self._stop_event.is_set():
            try:
                self._remux_queue.put_nowait(seg.seg_id)
            except queue.Full:
                pass
        return entry

    def on_fragment(
        self,
        tier: str,
        init_bytes: bytes | None,
        frag: StreamFragment,
        gop_snapshot: list[StreamFragment],
        width: int = 0,
        height: int = 0,
        wall_ts: float | None = None,
        mono_ts: float | None = None,
    ) -> None:
        """Принимает очередной fMP4-фрагмент от H264StreamHub и пишет его на диск."""
        if not init_bytes or tier not in ("low", "hi"):
            return
        wall_now = float(time.time() if wall_ts is None else wall_ts)
        mono_now = float(time.monotonic() if mono_ts is None else mono_ts)

        with self._lock:
            if tier == "hi" and not self._is_hi_active_locked(mono_now, wall_now):
                if "hi" in self._open_segments:
                    self._close_open_segment_locked("hi", enqueue_remux=True)
                return

            open_seg = self._open_segments.get(tier)
            if open_seg is None:
                seed_frags: list[StreamFragment] = []
                if gop_snapshot and gop_snapshot[0].is_keyframe:
                    seed_frags.extend(gop_snapshot)
                if not seed_frags or seed_frags[-1].seq < frag.seq:
                    if frag.is_keyframe or seed_frags:
                        seed_frags.append(frag)
                if not seed_frags or not seed_frags[0].is_keyframe:
                    return
                self._open_new_segment_locked(
                    tier=tier,
                    init_bytes=init_bytes,
                    first_frags=seed_frags,
                    wall_now=wall_now,
                    mono_now=mono_now,
                    width=width,
                    height=height,
                )
                return

            # Ротация длинного сегмента на ключевом кадре
            elapsed = max(mono_now - open_seg.start_mono, wall_now - open_seg.start_ts)
            if elapsed >= float(self.segment_seconds) and (frag.is_keyframe or elapsed >= float(self.segment_seconds) + 30.0):
                self._close_open_segment_locked(tier, enqueue_remux=True)
                seed_next = [frag] if frag.is_keyframe else (
                    [fr for fr in gop_snapshot if fr.seq >= frag.seq] or [frag]
                )
                self._open_new_segment_locked(
                    tier=tier,
                    init_bytes=init_bytes,
                    first_frags=seed_next,
                    wall_now=wall_now,
                    mono_now=mono_now,
                    width=width,
                    height=height,
                )
                return

            if frag.seq <= open_seg.last_seq:
                return

            patched, eff_base = rebase_moof_tfdt(frag.data, open_seg.base_tfdt)
            if open_seg.base_tfdt is None:
                open_seg.base_tfdt = eff_base
            try:
                open_seg.fh.write(patched)
                open_seg.size_bytes += len(patched)
                open_seg.last_seq = frag.seq
                open_seg.end_ts = max(open_seg.end_ts, wall_now)
                open_seg.last_frag_mono = mono_now
                if (mono_now - open_seg.last_flush_mono) >= 2.0:
                    open_seg.fh.flush()
                    open_seg.last_flush_mono = mono_now
            except OSError as exc:
                logger.warning("Ошибка записи фрагмента в сегмент %s: %s", open_seg.seg_id, exc)
                self._close_open_segment_locked(tier, enqueue_remux=False)

    def _remux_segment_faststart(self, seg_id: str) -> bool:
        ffmpeg_bin = self._explicit_ffmpeg or find_ffmpeg_bin()
        if not ffmpeg_bin:
            return False
        with self._lock:
            entry = next((s for s in self._segments if s["id"] == seg_id), None)
            if entry is None:
                return False
            seg_path = self.segments_dir / entry["rel_path"]
        if not seg_path.is_file():
            return False

        tmp_path = seg_path.with_name(f"{seg_path.stem}.tmp.mp4")
        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(seg_path),
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=CREATE_NO_WINDOW,
            )
            self._active_remux_proc = proc
            _, err_bytes = proc.communicate(timeout=35.0)
            self._active_remux_proc = None
            if proc.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size <= 256:
                if tmp_path.is_file():
                    tmp_path.unlink(missing_ok=True)
                err_txt = err_bytes.decode("utf-8", errors="replace").strip() if err_bytes else ""
                if err_txt:
                    logger.debug("Ремукс DVR %s пропущен (%s)", seg_id, err_txt)
                return False

            os.replace(tmp_path, seg_path)
            new_size = int(seg_path.stat().st_size)
            with self._lock:
                cur = next((s for s in self._segments if s["id"] == seg_id), None)
                if cur is not None:
                    delta = new_size - int(cur.get("size_bytes", 0))
                    self._total_bytes = max(0, self._total_bytes + delta)
                    cur["size_bytes"] = new_size
                    cur["faststart"] = True
                    self._save_index_locked()
            return True
        except Exception:
            self._active_remux_proc = None
            try:
                if tmp_path.is_file():
                    tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _remux_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                seg_id = self._remux_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if seg_id is None or self._stop_event.is_set():
                break
            self._remux_segment_faststart(seg_id)

    def _housekeep_loop(self) -> None:
        while not self._stop_event.wait(self.RETENTION_CHECK_INTERVAL_SEC):
            if self._stop_event.is_set():
                break
            try:
                # Закрываем остывший hi-сегмент, даже если поток перестал присылать кадры на затихании
                now_mono = time.monotonic()
                now_wall = time.time()
                with self._lock:
                    if "hi" in self._open_segments and not self._is_hi_active_locked(now_mono, now_wall):
                        self._close_open_segment_locked("hi", enqueue_remux=True)
                self.enforce_retention(now_ts=now_wall)
            except Exception as exc:
                logger.warning("Ошибка обслуживания архива DVR: %s", exc)

    def _probe_disk_usage(self) -> tuple[int, int]:
        """Возвращает (total_bytes, free_bytes) тома, где расположен dvr_dir (-1 если не удалось прочитать)."""
        probe = self.dvr_dir
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
            disk_total = int(usage.total)
            disk_free = int(usage.free)
        except OSError:
            disk_total, disk_free = 0, -1
        if self._disk_free_fn is not None:
            try:
                disk_free = int(self._disk_free_fn(self.dvr_dir))
            except Exception:
                pass
        return disk_total, disk_free

    def enforce_retention(self, now_ts: float | None = None) -> int:
        ts_now = float(time.time() if now_ts is None else now_ts)
        cutoff = ts_now - (self.retention_days * 86400.0)
        removed = 0

        with self._lock:
            kept: list[dict[str, Any]] = []
            to_delete: list[dict[str, Any]] = []
            for seg in self._segments:
                if float(seg.get("end_ts") or 0.0) < cutoff:
                    to_delete.append(seg)
                else:
                    kept.append(seg)

            # Если превышена квота объёма или порог свободного места — удаляем самые старые сегменты.
            # В пределах одного часа сначала освобождаем тяжёлый «hi», чтобы дольше сохранить «low» 360p.
            _, disk_free = self._probe_disk_usage()
            projected_total = sum(int(s.get("size_bytes") or 0) for s in kept)
            reclaimed = 0

            if (disk_free >= 0 and (disk_free + reclaimed) < self.min_free_bytes) or projected_total > self.max_bytes:
                ordered = sorted(
                    kept,
                    key=lambda s: (
                        int(float(s.get("start_ts") or 0.0) // 3600),
                        0 if s.get("tier") == "hi" else 1,
                        float(s.get("start_ts") or 0.0),
                    ),
                )
                drop_ids: set[str] = set()
                for cand in ordered:
                    need_space = (
                        disk_free >= 0 and (disk_free + reclaimed) < self.min_free_bytes
                    ) or projected_total > self.max_bytes
                    if not need_space:
                        break
                    sz = int(cand.get("size_bytes") or 0)
                    drop_ids.add(str(cand["id"]))
                    to_delete.append(cand)
                    projected_total = max(0, projected_total - sz)
                    reclaimed += sz
                if drop_ids:
                    kept = [s for s in kept if str(s["id"]) not in drop_ids]

            for seg in to_delete:
                seg_path = self.segments_dir / str(seg.get("rel_path") or "")
                try:
                    if seg_path.is_file():
                        seg_path.unlink()
                    removed += 1
                except OSError:
                    pass

            if removed > 0 or to_delete:
                self._segments = kept
                self._total_bytes = sum(int(s.get("size_bytes") or 0) for s in kept)
                self._save_index_locked()

        # Удаляем опустевшие папки дат в segments/
        if self.segments_dir.is_dir():
            try:
                for day_dir in list(self.segments_dir.iterdir()):
                    if day_dir.is_dir():
                        try:
                            if not any(day_dir.iterdir()):
                                day_dir.rmdir()
                        except OSError:
                            pass
            except OSError:
                pass

        if self.motion_log is not None:
            try:
                self.motion_log.prune(self.retention_days, now_ts=ts_now)
            except Exception:
                pass
        return removed

    def _segment_to_public(self, seg: dict[str, Any], is_open: bool = False) -> dict[str, Any]:
        st = float(seg.get("start_ts") or 0.0)
        en = max(st, float(seg.get("end_ts") or st))
        sid = str(seg["id"])
        return {
            "id": sid,
            "tier": str(seg.get("tier") or "low"),
            "date": str(seg.get("date") or ""),
            "start_ts": round(st, 3),
            "end_ts": round(en, 3),
            "duration_sec": round(max(0.0, en - st), 1),
            "width": int(seg.get("width") or 0),
            "height": int(seg.get("height") or 0),
            "rotate": int(seg.get("rotate") or 0),
            "flip_h": bool(seg.get("flip_h", False)),
            "size_bytes": int(seg.get("size_bytes") or 0),
            "faststart": bool(seg.get("faststart", False)),
            "open": bool(is_open),
            "url": f"/api/dvr/segments/{sid}.mp4",
        }

    def _all_segments_snapshot_locked(self) -> list[dict[str, Any]]:
        out = [self._segment_to_public(s, is_open=False) for s in self._segments]
        for oseg in self._open_segments.values():
            try:
                oseg.fh.flush()
            except Exception:
                pass
            out.append(
                self._segment_to_public(
                    {
                        "id": oseg.seg_id,
                        "tier": oseg.tier,
                        "date": oseg.date_str,
                        "start_ts": oseg.start_ts,
                        "end_ts": max(oseg.start_ts, oseg.end_ts),
                        "width": oseg.width,
                        "height": oseg.height,
                        "rotate": oseg.rotate,
                        "flip_h": oseg.flip_h,
                        "size_bytes": oseg.size_bytes,
                        "faststart": False,
                    },
                    is_open=True,
                )
            )
        out.sort(key=lambda s: (float(s["start_ts"]), 0 if s["tier"] == "hi" else 1))
        return out

    def list_segments(self, start_ts: float, end_ts: float) -> list[dict[str, Any]]:
        st = float(start_ts)
        en = float(end_ts)
        if en < st:
            return []
        with self._lock:
            snap = self._all_segments_snapshot_locked()
        return [
            s
            for s in snap
            if float(s["end_ts"]) >= st and float(s["start_ts"]) <= en
        ]

    def resolve_best_segment(self, ts: float, prefer_tier: str = "auto") -> dict[str, Any] | None:
        target = float(ts)
        pref = str(prefer_tier or "auto").strip().lower()
        with self._lock:
            snap = self._all_segments_snapshot_locked()
        if not snap:
            return None

        def _score(s: dict[str, Any]) -> tuple[int, int, float]:
            st = float(s["start_ts"])
            en = float(s["end_ts"])
            inside = 0 if (st <= target <= en) else 1
            dist = 0.0 if inside == 0 else min(abs(target - st), abs(target - en))
            tier = str(s.get("tier") or "low")
            if pref == "low":
                tier_rank = 0 if tier == "low" else 1
            else:
                tier_rank = 0 if tier == "hi" else 1
            return (inside, tier_rank, dist)

        # Сначала ищем точное попадание или ближайший в пределах ±30 с
        candidates = [
            s
            for s in snap
            if (float(s["start_ts"]) - 30.0) <= target <= (float(s["end_ts"]) + 30.0)
        ]
        if not candidates:
            return None

        best = min(candidates, key=_score)
        best_copy = dict(best)
        offset = max(0.0, target - float(best_copy["start_ts"]))
        if best_copy["duration_sec"] > 0.5:
            offset = min(offset, max(0.0, float(best_copy["duration_sec"]) - 0.2))
        best_copy["offset_sec"] = round(offset, 2)

        # Ищем следующий смежный сегмент (для непрерывного авто-перехода в плеере)
        next_cands = [
            s
            for s in snap
            if float(s["start_ts"]) >= float(best_copy["end_ts"]) - 2.0 and s["id"] != best_copy["id"]
        ]
        if next_cands:
            nxt = min(
                next_cands,
                key=lambda s: (abs(float(s["start_ts"]) - float(best_copy["end_ts"])), 0 if s["tier"] == best_copy["tier"] else 1),
            )
            best_copy["next_id"] = nxt["id"]
            best_copy["next_url"] = nxt["url"]
        return best_copy

    def get_segment_path(self, seg_id: str) -> Path | None:
        if not _SAFE_SEG_ID_RE.match(str(seg_id or "")):
            return None
        with self._lock:
            for oseg in self._open_segments.values():
                if oseg.seg_id == seg_id:
                    try:
                        oseg.fh.flush()
                    except Exception:
                        pass
                    return oseg.abs_path if oseg.abs_path.is_file() else None
            entry = next((s for s in self._segments if s["id"] == seg_id), None)
            if entry is None:
                return None
            p = (self.segments_dir / str(entry["rel_path"])).resolve()
            root = self.segments_dir.resolve()
            if root not in p.parents:
                return None
            return p if p.is_file() else None

    def storage_stats(self) -> dict[str, Any]:
        disk_total, disk_free = self._probe_disk_usage()
        motion_bytes = self.motion_log.storage_bytes() if self.motion_log is not None else 0
        with self._lock:
            low_count = sum(1 for s in self._segments if s["tier"] == "low")
            hi_count = sum(1 for s in self._segments if s["tier"] == "hi")
            used_b = self._total_bytes + sum(o.size_bytes for o in self._open_segments.values())
            active_low = "low" in self._open_segments
            active_hi = "hi" in self._open_segments
            oldest_ts = float(self._segments[0]["start_ts"]) if self._segments else None
            newest_ts = float(self._segments[-1]["end_ts"]) if self._segments else None

        return {
            "dvr_dir": str(self.dvr_dir),
            "retention_days": self.retention_days,
            "max_gb": round(self.max_bytes / (1024 ** 3), 2),
            "min_free_gb": round(self.min_free_bytes / (1024 ** 3), 2),
            "used_gb": round(used_b / (1024 ** 3), 3),
            "used_mb": round(used_b / (1024 ** 2), 1),
            "motion_log_mb": round(motion_bytes / (1024 ** 2), 2),
            "disk_free_gb": round(max(0, disk_free) / (1024 ** 3), 2),
            "disk_total_gb": round(disk_total / (1024 ** 3), 2),
            "segments_low": low_count,
            "segments_hi": hi_count,
            "active_low": active_low,
            "active_hi": active_hi,
            "oldest_ts": oldest_ts,
            "newest_ts": newest_ts,
        }

    def status_summary(self) -> dict[str, Any]:
        disk_total, disk_free = self._probe_disk_usage()
        with self._lock:
            used_b = self._total_bytes + sum(o.size_bytes for o in self._open_segments.values())
            active_low = "low" in self._open_segments
            active_hi = "hi" in self._open_segments
            seg_count = len(self._segments) + len(self._open_segments)
        return {
            "enabled": True,
            "retention_days": self.retention_days,
            "used_gb": round(used_b / (1024 ** 3), 3),
            "disk_free_gb": round(max(0, disk_free) / (1024 ** 3), 2),
            "disk_total_gb": round(disk_total / (1024 ** 3), 2),
            "active_low": active_low,
            "active_hi": active_hi,
            "segments_count": seg_count,
        }

    def list_days(self, now_ts: float | None = None) -> dict[str, Any]:
        ts_now = float(time.time() if now_ts is None else now_ts)
        with self._lock:
            snap = self._all_segments_snapshot_locked()

        by_date: dict[str, dict[str, Any]] = {}
        intervals_by_date: dict[str, list[tuple[float, float]]] = {}
        today = datetime.fromtimestamp(ts_now).date()

        def _ensure_day(d_str: str) -> dict[str, Any] | None:
            if d_str in by_date:
                return by_date[d_str]
            try:
                d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()
            except ValueError:
                return None
            day_start = datetime(d_obj.year, d_obj.month, d_obj.day, 0, 0, 0).timestamp()
            day_end = day_start + 86400.0
            has_motion = False
            if self.motion_log is not None:
                m_dir = self.motion_log.root_dir / d_str
                has_motion = m_dir.is_dir() and any(m_dir.iterdir())
            entry = {
                "date": d_str,
                "start_ts": day_start,
                "end_ts": min(ts_now, day_end) if d_obj == today else day_end,
                "low_count": 0,
                "hi_count": 0,
                "covered_sec": 0.0,
                "size_mb": 0.0,
                "has_motion": has_motion,
            }
            by_date[d_str] = entry
            intervals_by_date[d_str] = []
            return entry

        # Включаем последние retention_days дней + все даты с фактическими сегментами
        max_days = max(1, int(round(self.retention_days)))
        for delta in range(max_days):
            d_obj = datetime.fromtimestamp(ts_now - delta * 86400.0).date()
            _ensure_day(d_obj.strftime("%Y-%m-%d"))

        for seg in snap:
            d_str = str(seg.get("date") or "")
            item = _ensure_day(d_str)
            if item is None:
                continue
            if seg["tier"] == "hi":
                item["hi_count"] += 1
            else:
                item["low_count"] += 1
            item["size_mb"] += float(seg.get("size_bytes") or 0) / (1024 ** 2)
            st = float(seg.get("start_ts") or 0.0)
            en = max(st, float(seg.get("end_ts") or st))
            if en > st:
                intervals_by_date[d_str].append((st, en))

        days = sorted(by_date.values(), key=lambda x: x["date"], reverse=True)
        for d in days:
            ivs = sorted(intervals_by_date.get(d["date"], []))
            covered = 0.0
            cur_st: float | None = None
            cur_en: float = 0.0
            for iv_st, iv_en in ivs:
                if cur_st is None:
                    cur_st, cur_en = iv_st, iv_en
                elif iv_st <= cur_en + 1.0:
                    cur_en = max(cur_en, iv_en)
                else:
                    covered += cur_en - cur_st
                    cur_st, cur_en = iv_st, iv_en
            if cur_st is not None:
                covered += cur_en - cur_st
            d["covered_sec"] = round(min(86400.0, covered), 1)
            d["size_mb"] = round(d["size_mb"], 1)

        return {
            "days": days,
            "storage": self.storage_stats(),
        }
