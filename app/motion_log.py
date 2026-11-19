from __future__ import annotations

import gzip
import json
import logging
import shutil
import threading
import time
import zlib
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("fallguard.motion_log")


def _round_kps(kps: Any) -> list[list[float]]:
    if not isinstance(kps, (list, tuple)):
        return []
    out: list[list[float]] = []
    for pt in kps[:17]:
        if isinstance(pt, (list, tuple)) and len(pt) >= 3:
            c = float(pt[2])
            if c <= 0.0:
                out.append([0.0, 0.0, 0.0])
            else:
                out.append([round(float(pt[0]), 4), round(float(pt[1]), 4), round(c, 3)])
    return out


def _round_bbox(bbox: Any) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return []
    return [
        round(float(bbox[0]), 4),
        round(float(bbox[1]), 4),
        round(float(bbox[2]), 4),
        round(float(bbox[3]), 4),
    ]


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    try:
        with gzip.open(path, "rb") as gz:
            while True:
                try:
                    raw_line = gz.readline()
                except (EOFError, gzip.BadGzipFile, zlib.error, OSError):
                    break
                if not raw_line:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                    if isinstance(obj, dict) and "ts" in obj:
                        records.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    return records


class MotionLogStore:
    """
    7-дневный «чёрный ящик» телеметрии движения, поз, ролей и таймеров FSM.
    Пишет сжатые почасовые журналы <dvr_dir>/motion_logs/YYYY-MM-DD/HH.jsonl.gz
    и хранит последние ~120 секунд в кольцевом буфере ОЗУ для мгновенных отчётов об ошибках.
    """

    def __init__(
        self,
        base_dir: Path | str,
        retention_days: float = 7.0,
        flush_interval_sec: float = 5.0,
        flush_batch_size: int = 40,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.root_dir = self.base_dir / "motion_logs"
        self.retention_days = max(0.5, float(retention_days))
        self._flush_interval_sec = max(1.0, float(flush_interval_sec))
        self._flush_batch_size = max(1, int(flush_batch_size))

        self._lock = threading.RLock()
        self._ring: deque[dict[str, Any]] = deque(maxlen=600)
        self._pending: dict[tuple[str, str], list[str]] = {}
        self._pending_count = 0
        self._last_flush_mono = time.monotonic()

        self._last_logged_ts: float = -1e12
        self._last_sig: Any = None
        self._bucket_cache: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    def _hour_path(self, date_str: str, hour_str: str) -> Path:
        return self.root_dir / date_str / f"{hour_str}.jsonl.gz"

    @staticmethod
    def _compact_track(tr: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": int(tr.get("track_id", -1)),
            "r": str(tr.get("role") or "unknown"),
            "rs": str(tr.get("reason") or ""),
            "b": _round_bbox(tr.get("bbox")),
            "kp": _round_kps(tr.get("keypoints")),
            "ang": round(float(tr["angle_deg"]), 1) if tr.get("angle_deg") is not None else None,
            "h": round(float(tr["height_cm"]), 1) if tr.get("height_cm") is not None else None,
            "xy": (
                [round(float(tr["floor_xy_cm"][0]), 1), round(float(tr["floor_xy_cm"][1]), 1)]
                if isinstance(tr.get("floor_xy_cm"), (list, tuple)) and len(tr["floor_xy_cm"]) >= 2
                else None
            ),
            "ob": bool(tr.get("on_bed", False)),
            "hide": bool(tr.get("hide_skeleton", False)),
        }
        return item

    @staticmethod
    def _compact_raw_pose(obs: Any, reason: str) -> dict[str, Any]:
        if isinstance(obs, dict):
            tid = int(obs.get("track_id", -1))
            bbox = _round_bbox(obs.get("bbox") or obs.get("bbox_norm"))
            kps = _round_kps(obs.get("keypoints") or obs.get("keypoints_norm"))
            conf = float(obs.get("conf", 0.0) or 0.0)
        else:
            tid = int(getattr(obs, "track_id", -1))
            bbox = _round_bbox(getattr(obs, "bbox_norm", ()))
            kps = _round_kps(getattr(obs, "keypoints_norm", ()))
            conf = float(getattr(obs, "conf", 0.0) or 0.0)
        return {
            "id": tid,
            "b": bbox,
            "kp": kps,
            "conf": round(conf, 3),
            "why": reason,
        }

    def record(
        self,
        ts: float,
        state: str,
        status_reason: str = "",
        optical_motion: float = 0.0,
        caregiver_nearby: bool = False,
        door_visit_active: bool = False,
        patient_in_bed_latched: bool = False,
        is_night_ir: bool = False,
        timers: dict[str, float] | None = None,
        tracks_ui: Sequence[dict[str, Any]] = (),
        raw_observations: Sequence[Any] = (),
        force: bool = False,
    ) -> bool:
        """
        Добавляет снимок состояния в 7-дневный лог с адаптивной частотой:
        до 4 Гц при движении/смене состояния/фантомах и 0.2 Гц в полном покое.
        """
        ts_val = float(ts)
        real_ui = [
            tr
            for tr in tracks_ui
            if isinstance(tr, dict) and int(tr.get("track_id", -1)) >= 0 and not tr.get("virtual_presence")
        ]
        live_ids = {int(tr.get("track_id", -1)) for tr in real_ui}

        ign_list: list[dict[str, Any]] = []
        for raw_obs in raw_observations:
            raw_tid = int(
                raw_obs.get("track_id", -1)
                if isinstance(raw_obs, dict)
                else getattr(raw_obs, "track_id", -1)
            )
            if raw_tid not in live_ids:
                ign_list.append(self._compact_raw_pose(raw_obs, "filtered_or_unconfirmed"))

        sig = (
            str(state),
            bool(caregiver_nearby),
            bool(door_visit_active),
            bool(patient_in_bed_latched),
            tuple((int(t.get("track_id", -1)), str(t.get("role") or "")) for t in real_ui),
            len(ign_list) > 0,
        )
        is_active = (
            float(optical_motion) >= 0.010
            or str(state) not in ("in_bed", "idle", "paused")
            or bool(ign_list)
            or any(str(t.get("role") or "") == "visitor" for t in real_ui)
        )
        min_dt = 0.25 if is_active else 5.0

        with self._lock:
            dt = ts_val - self._last_logged_ts
            sig_changed = sig != self._last_sig
            if not force and not sig_changed and abs(dt) < min_dt:
                return False

            tm_clean = {
                k: round(float(v), 2)
                for k, v in (timers or {}).items()
                if v is not None and float(v) > 0.0
            }
            entry: dict[str, Any] = {
                "ts": round(ts_val, 3),
                "st": str(state),
                "sr": str(status_reason or ""),
                "opt": round(float(optical_motion), 4),
                "cg": bool(caregiver_nearby),
                "dv": bool(door_visit_active),
                "bl": bool(patient_in_bed_latched),
                "ir": bool(is_night_ir),
                "tm": tm_clean,
                "tr": [self._compact_track(t) for t in real_ui],
            }
            if ign_list:
                entry["ign"] = ign_list

            self._last_logged_ts = ts_val
            self._last_sig = sig
            self._ring.append(entry)

            dt_obj = datetime.fromtimestamp(ts_val)
            key = (dt_obj.strftime("%Y-%m-%d"), dt_obj.strftime("%H"))
            line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
            self._pending.setdefault(key, []).append(line)
            self._pending_count += 1

            now_mono = time.monotonic()
            if (
                self._pending_count >= self._flush_batch_size
                or (now_mono - self._last_flush_mono) >= self._flush_interval_sec
                or sig_changed
            ):
                self._flush_locked()
            return True

    def _flush_locked(self) -> None:
        if not self._pending:
            self._last_flush_mono = time.monotonic()
            return
        pending = self._pending
        self._pending = {}
        self._pending_count = 0
        self._last_flush_mono = time.monotonic()

        for (date_str, hour_str), lines in pending.items():
            if not lines:
                continue
            for stale_key in [k for k in self._bucket_cache if k[0] == date_str and k[1] == hour_str]:
                self._bucket_cache.pop(stale_key, None)
            path = self._hour_path(date_str, hour_str)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                raw_bytes = "".join(lines).encode("utf-8")
                gz_chunk = gzip.compress(raw_bytes, compresslevel=5)
                with open(path, "ab") as f:
                    f.write(gz_chunk)
            except OSError as exc:
                logger.warning("Не удалось дописать лог движения %s: %s", path, exc)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _collect_pending_records_locked(
        self,
        start_ts: float,
        end_ts: float,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for lines in self._pending.values():
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    ts_v = float(obj.get("ts") or 0.0)
                    if start_ts <= ts_v <= end_ts:
                        out.append(obj)
                except Exception:
                    continue
        return out

    def _iter_hour_keys(self, start_ts: float, end_ts: float) -> list[tuple[str, str]]:
        if end_ts < start_ts:
            return []
        cur = datetime.fromtimestamp(start_ts).replace(minute=0, second=0, microsecond=0)
        end_dt = datetime.fromtimestamp(end_ts)
        keys: list[tuple[str, str]] = []
        while cur <= end_dt:
            keys.append((cur.strftime("%Y-%m-%d"), cur.strftime("%H")))
            cur += timedelta(hours=1)
        return keys

    def query_range(
        self,
        start_ts: float,
        end_ts: float,
        max_points: int = 2400,
    ) -> list[dict[str, Any]]:
        st = float(start_ts)
        en = float(end_ts)
        if en < st:
            return []

        with self._lock:
            self._flush_locked()
            ring_hits = [
                dict(rec)
                for rec in self._ring
                if st <= float(rec.get("ts") or 0.0) <= en
            ]
            hour_keys = self._iter_hour_keys(st, en)

        records_by_ts: dict[float, dict[str, Any]] = {}
        for d_str, h_str in hour_keys:
            h_path = self._hour_path(d_str, h_str)
            if not h_path.is_file():
                continue
            for rec in _read_gzip_jsonl(h_path):
                ts_v = float(rec.get("ts") or 0.0)
                if st <= ts_v <= en:
                    records_by_ts[ts_v] = rec

        for rec in ring_hits:
            ts_v = float(rec.get("ts") or 0.0)
            records_by_ts[ts_v] = rec

        all_recs = [records_by_ts[k] for k in sorted(records_by_ts.keys())]
        limit = max(10, int(max_points))
        if len(all_recs) <= limit:
            return all_recs

        # Умное прореживание: обязательно оставляем смены состояний и отсеянных кандидатов (ign)
        essential_idx: set[int] = {0, len(all_recs) - 1}
        prev_st = all_recs[0].get("st")
        for idx, rec in enumerate(all_recs):
            if rec.get("st") != prev_st or rec.get("ign"):
                essential_idx.add(idx)
            prev_st = rec.get("st")

        step = max(1, len(all_recs) // limit)
        for idx in range(0, len(all_recs), step):
            essential_idx.add(idx)

        selected = [all_recs[i] for i in sorted(essential_idx)]
        if len(selected) > limit:
            thin_step = max(1, len(selected) // limit)
            selected = selected[::thin_step]
        return selected

    def recent_slice(
        self,
        window_sec: float = 25.0,
        around_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        win = max(1.0, float(window_sec))
        if around_ts is None:
            now_t = time.time()
            with self._lock:
                if self._ring:
                    latest = float(self._ring[-1].get("ts") or now_t)
                    cutoff = max(now_t, latest) - win
                    items = [dict(r) for r in self._ring if float(r.get("ts") or 0.0) >= cutoff]
                    if items:
                        return items
            return self.query_range(now_t - win, now_t + 1.0, max_points=600)

        target = float(around_ts)
        half = win / 2.0
        return self.query_range(target - half, target + half, max_points=600)

    def motion_buckets(
        self,
        start_ts: float,
        end_ts: float,
        bucket_sec: float = 300.0,
    ) -> list[dict[str, Any]]:
        """
        Агрегирует пиковую интенсивность движения (opt), фантомов (ign) и присутствия людей по корзинам,
        используя кэш завершённых прошлых часов для мгновенного ответа в UI архива.
        """
        st = float(start_ts)
        en = float(end_ts)
        bsec = max(30, int(bucket_sec))
        if en <= st:
            return []

        now_ts = time.time()
        with self._lock:
            self._flush_locked()
            hour_keys = self._iter_hour_keys(st, en)

        buckets_map: dict[int, dict[str, Any]] = {}
        for d_str, h_str in hour_keys:
            cache_key = (d_str, h_str, bsec)
            try:
                hour_dt = datetime.strptime(f"{d_str} {h_str}", "%Y-%m-%d %H")
                hour_end_ts = (hour_dt + timedelta(hours=1)).timestamp()
            except ValueError:
                hour_end_ts = now_ts

            can_cache = hour_end_ts < (now_ts - 90.0)
            with self._lock:
                cached = self._bucket_cache.get(cache_key) if can_cache else None

            if cached is not None:
                hour_buckets = cached
            else:
                h_path = self._hour_path(d_str, h_str)
                local_map: dict[int, dict[str, Any]] = {}
                if h_path.is_file():
                    for rec in _read_gzip_jsonl(h_path):
                        ts_v = float(rec.get("ts") or 0.0)
                        b_idx = int(ts_v // bsec)
                        b = local_map.get(b_idx)
                        if b is None:
                            b = {
                                "start_ts": b_idx * bsec,
                                "end_ts": (b_idx + 1) * bsec,
                                "max_opt": 0.0,
                                "samples": 0,
                                "ign_hits": 0,
                                "max_tracks": 0,
                                "alert_states": 0,
                            }
                            local_map[b_idx] = b
                        b["samples"] += 1
                        opt_v = float(rec.get("opt") or 0.0)
                        if opt_v > b["max_opt"]:
                            b["max_opt"] = round(opt_v, 4)
                        if rec.get("ign"):
                            b["ign_hits"] += 1
                        tr_len = len(rec.get("tr") or ())
                        if tr_len > b["max_tracks"]:
                            b["max_tracks"] = tr_len
                        if str(rec.get("st") or "") in ("sitting_up", "left_bed", "fallen", "help_call", "missing"):
                            b["alert_states"] += 1
                hour_buckets = [local_map[k] for k in sorted(local_map.keys())]
                if can_cache and hour_buckets:
                    with self._lock:
                        if len(self._bucket_cache) > 220:
                            self._bucket_cache.clear()
                        self._bucket_cache[cache_key] = hour_buckets

            for hb in hour_buckets:
                if float(hb["end_ts"]) < st or float(hb["start_ts"]) > en:
                    continue
                b_idx = int(hb["start_ts"] // bsec)
                existing = buckets_map.get(b_idx)
                if existing is None:
                    buckets_map[b_idx] = dict(hb)
                else:
                    existing["samples"] += hb["samples"]
                    existing["max_opt"] = max(existing["max_opt"], hb["max_opt"])
                    existing["ign_hits"] += hb["ign_hits"]
                    existing["max_tracks"] = max(existing["max_tracks"], hb["max_tracks"])
                    existing["alert_states"] += hb["alert_states"]

        return [buckets_map[k] for k in sorted(buckets_map.keys())]

    def storage_bytes(self) -> int:
        if not self.root_dir.is_dir():
            return 0
        total = 0
        try:
            for day_dir in self.root_dir.iterdir():
                if not day_dir.is_dir():
                    continue
                for f in day_dir.iterdir():
                    if f.is_file():
                        with_stat = f.stat()
                        total += int(with_stat.st_size)
        except OSError:
            pass
        return total

    def prune(self, retention_days: float | None = None, now_ts: float | None = None) -> int:
        days = float(self.retention_days if retention_days is None else retention_days)
        ts_now = float(time.time() if now_ts is None else now_ts)
        cutoff_dt = datetime.fromtimestamp(ts_now - days * 86400.0)
        cutoff_date = cutoff_dt.strftime("%Y-%m-%d")

        if not self.root_dir.is_dir():
            return 0
        removed = 0
        try:
            for day_dir in list(self.root_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                name = day_dir.name
                # Удаляем только папки YYYY-MM-DD, строго старше cutoff_date
                if len(name) == 10 and name[4] == "-" and name[7] == "-" and name < cutoff_date:
                    shutil.rmtree(day_dir, ignore_errors=True)
                    removed += 1
                    with self._lock:
                        stale_keys = [k for k in self._bucket_cache if k[0] == name]
                        for k in stale_keys:
                            self._bucket_cache.pop(k, None)
        except OSError as exc:
            logger.warning("Ошибка очистки старых логов движения: %s", exc)
        return removed
