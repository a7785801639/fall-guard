from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

STATE_MAP: dict[str, tuple[str, str, str]] = {
    "paused": ("paused", "paused", "Пауза"),
    "fallen": ("fallen", "alarm", "Падение"),
    "help_call": ("help_call", "warn", "Зовёт рукой"),
    "sitting_up": ("sitting_up", "warn", "Села в кровати"),
    "left_bed": ("left_bed", "warn", "Встала одна"),
    "missing": ("missing", "warn", "Вышла из кадра"),
    "caregiver": ("caregiver", "caregiver", "Рядом свой"),
    "in_bed": ("in_bed", "calm", "В кровати"),
    "idle": ("idle", "calm", "Спокойно"),
}


def _map_state(state_val: str | Any, cam_online: bool) -> tuple[str, str, str]:
    if not cam_online:
        return ("offline", "offline", "Нет сигнала")
    raw = str(getattr(state_val, "value", state_val) or "idle").strip().lower()
    return STATE_MAP.get(raw, ("idle", "calm", "Спокойно"))


class TimelineStore:
    """
    Потокобезопасное хранилище хроники состояний (6 ч / 24 ч) с атомарной записью на диск.
    Хранит интервалы {"start": float, "end": float, "state": str, "tone": str, "label": str}
    в секундах Unix epoch (time.time()).
    """

    MERGE_GAP_SEC: float = 30.0
    DISK_FLUSH_INTERVAL_SEC: float = 30.0
    MIN_CAREGIVER_SPAN_SEC: float = 3.5
    SCHEMA_VERSION: int = 6

    def __init__(self, path: Path | str, window_sec: float = 21600.0) -> None:
        self._path = Path(path)
        self.window_sec = max(60.0, float(window_sec))
        self._lock = threading.RLock()
        self._spans: list[dict[str, Any]] = []
        self._last_saved_ts: float = -1e12
        self._last_saved_mono: float = -1e12
        self._load_locked()

    @property
    def path(self) -> Path:
        return self._path

    def _coalesce_spans_locked(self) -> None:
        if len(self._spans) <= 1:
            return
        merged: list[dict[str, Any]] = [dict(self._spans[0])]
        for sp in self._spans[1:]:
            prev = merged[-1]
            st = float(sp["start"])
            en = float(sp["end"])
            gap = st - float(prev["end"])
            if sp["state"] == prev["state"] and gap <= self.MERGE_GAP_SEC * 2.0:
                prev["end"] = max(float(prev["end"]), en)
                prev["tone"] = sp["tone"]
                prev["label"] = sp["label"]
            else:
                merged.append(dict(sp))
        self._spans = merged

    def _load_locked(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        schema_ver = int(raw.get("schema_version", 0)) if isinstance(raw, dict) else 0
        items = raw.get("spans") if isinstance(raw, dict) else (raw if isinstance(raw, list) else None)
        if not isinstance(items, list):
            return
        loaded: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                st = float(item["start"])
                en = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if en < st:
                en = st
            state_str = str(item.get("state") or "idle")
            if state_str == "offline":
                mapped_state, tone, label = ("offline", "offline", "Нет сигнала")
            else:
                mapped_state, default_tone, default_label = _map_state(state_str, True)
                tone = str(item.get("tone") or default_tone)
                label = str(item.get("label") or default_label)
            loaded.append(
                {
                    "start": st,
                    "end": en,
                    "state": mapped_state,
                    "tone": tone,
                    "label": label,
                }
            )
        loaded.sort(key=lambda sp: (sp["start"], sp["end"]))
        self._spans = loaded

        if schema_ver < self.SCHEMA_VERSION and self._spans:
            # Однократная миграция до-фиксовой хроники v5 -> v6: очистка коротких всплесков и ложных ИК-дублей caregiver
            latest_ts = max(float(self._spans[-1]["end"]), time.time())
            self.sanitize_glitches(
                min_caregiver_sec=self.MIN_CAREGIVER_SPAN_SEC,
                replace_caregiver_after_ts=latest_ts - 7200.0,
                now_ts=latest_ts,
            )
        else:
            self._coalesce_spans_locked()

    def _prune_locked(self, now_ts: float) -> None:
        cutoff = now_ts - self.window_sec
        kept: list[dict[str, Any]] = []
        for sp in self._spans:
            en = float(sp["end"])
            st = float(sp["start"])
            if en < cutoff or (en == cutoff and st < cutoff):
                continue
            if st < cutoff:
                sp = {**sp, "start": cutoff}
            kept.append(sp)
        self._spans = kept

    def _save_locked(self, now_ts: float) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "window_sec": self.window_sec,
                "updated_ts": now_ts,
                "spans": self._spans,
            }
            tmp_path = self._path.with_name(f"{self._path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._path)
            self._last_saved_ts = now_ts
            self._last_saved_mono = time.monotonic()
        except OSError:
            pass

    def sanitize_glitches(
        self,
        min_caregiver_sec: float = 3.5,
        min_transient_sec: float = 1.2,
        replace_caregiver_after_ts: float | None = None,
        keep_clip_timestamps: list[float] | None = None,
        now_ts: float | None = None,
    ) -> int:
        """
        Очищает мимолётные ложные всплески хроники (короткие фантомы caregiver < min_caregiver_sec,
        нулевые сбои без клипов и при миграции — до-фиксовые ночные ИК-дубли), склеивая соседние спокойные интервалы.
        Возвращает количество устранённых ложных сегментов.
        """
        clip_ts = [float(t) for t in (keep_clip_timestamps or [])]

        def _has_clip_near(st: float, en: float) -> bool:
            return any((st - 25.0) <= ct <= (en + 25.0) for ct in clip_ts)

        with self._lock:
            if not self._spans:
                return 0
            ts = float(time.time() if now_ts is None else now_ts)
            removed_count = 0
            cleaned: list[dict[str, Any]] = []
            total = len(self._spans)

            for idx, sp in enumerate(self._spans):
                is_last = idx == (total - 1)
                st = float(sp["start"])
                en = float(sp["end"])
                dur = max(0.0, en - st)
                state = str(sp.get("state") or "idle")

                # 1. До-фиксовые ложные ночные ИК-сегменты caregiver (без реальных клипов) переводим в in_bed
                if (
                    replace_caregiver_after_ts is not None
                    and state == "caregiver"
                    and en >= float(replace_caregiver_after_ts)
                    and not _has_clip_near(st, en)
                ):
                    mapped_state, tone, label = _map_state("in_bed", True)
                    cleaned.append(
                        {
                            "start": st,
                            "end": en,
                            "state": mapped_state,
                            "tone": tone,
                            "label": label,
                        }
                    )
                    removed_count += 1
                    continue

                # 2. Завершённые мимолётные всплески caregiver короче порога
                if not is_last and state == "caregiver" and dur < float(min_caregiver_sec):
                    removed_count += 1
                    if cleaned and cleaned[-1]["state"] in ("in_bed", "idle"):
                        cleaned[-1]["end"] = max(float(cleaned[-1]["end"]), en)
                    continue

                # 3. Завершённые субсекундные сбои (< min_transient_sec) без подтверждённого клипа внутри спокойного сна
                if (
                    not is_last
                    and state in ("idle", "offline", "sitting_up", "left_bed", "missing")
                    and dur < float(min_transient_sec)
                    and not _has_clip_near(st, en)
                    and cleaned
                    and cleaned[-1]["state"] == "in_bed"
                ):
                    removed_count += 1
                    cleaned[-1]["end"] = max(float(cleaned[-1]["end"]), en)
                    continue

                cleaned.append(dict(sp))

            self._spans = cleaned
            before_coalesce = len(self._spans)
            self._coalesce_spans_locked()
            if removed_count > 0 or len(self._spans) != before_coalesce:
                self._save_locked(ts)
            return removed_count

    def flush(self, now_ts: float | None = None) -> None:
        with self._lock:
            ts = float(time.time() if now_ts is None else now_ts)
            self._save_locked(ts)

    def dismiss_false_alarm_around(
        self,
        wall_ts: float,
        fallback_state: str = "in_bed",
        max_span_sec: float = 45.0,
    ) -> bool:
        """
        Находит тревожный сегмент около wall_ts (±30 с, длительностью <= max_span_sec)
        и переводит его в спокойный fallback_state (in_bed или caregiver) со склейкой соседних интервалов.
        """
        alarm_states = ("sitting_up", "left_bed", "fallen", "missing", "help_call")
        target_ts = float(wall_ts)
        safe_fallback = (
            fallback_state if fallback_state in ("in_bed", "caregiver", "idle") else "in_bed"
        )
        mapped_state, tone, label = _map_state(safe_fallback, True)
        limit_sec = max(1.0, float(max_span_sec))

        with self._lock:
            if not self._spans:
                return False
            changed = False
            for sp in self._spans:
                st = float(sp["start"])
                en = float(sp["end"])
                dur = max(0.0, en - st)
                if (
                    str(sp.get("state") or "") in alarm_states
                    and dur <= limit_sec
                    and st <= (target_ts + 30.0)
                    and en >= (target_ts - 30.0)
                ):
                    sp["state"] = mapped_state
                    sp["tone"] = tone
                    sp["label"] = label
                    changed = True
            if not changed:
                return False
            self._coalesce_spans_locked()
            save_ts = max(target_ts, float(self._spans[-1]["end"]), time.time())
            self._save_locked(save_ts)
            return True

    def clear(self) -> None:
        with self._lock:
            self._spans = []
            self._save_locked(time.time())

    def record(
        self,
        state_val: str | Any,
        cam_online: bool = True,
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        explicit_ts = now_ts is not None
        ts = float(time.time() if now_ts is None else now_ts)
        mapped_state, tone, label = _map_state(state_val, cam_online)

        with self._lock:
            state_changed = False
            if self._spans:
                last = self._spans[-1]
                gap = ts - float(last["end"])
                if last["state"] == mapped_state and -1.0 <= gap <= self.MERGE_GAP_SEC:
                    last["end"] = max(float(last["end"]), ts)
                    last["tone"] = tone
                    last["label"] = label
                    active_span = dict(last)
                else:
                    # Поглощаем мимолётный ложный всплеск caregiver (< 3.5 с) при возврате в спокойный режим
                    last_dur = max(0.0, float(last["end"]) - float(last["start"]))
                    if (
                        last["state"] == "caregiver"
                        and last_dur < self.MIN_CAREGIVER_SPAN_SEC
                        and mapped_state in ("in_bed", "idle")
                        and gap >= -1.0
                    ):
                        popped = self._spans.pop()
                        if (
                            self._spans
                            and self._spans[-1]["state"] == mapped_state
                            and (ts - float(self._spans[-1]["end"])) <= self.MERGE_GAP_SEC * 2.0
                        ):
                            self._spans[-1]["end"] = max(float(self._spans[-1]["end"]), ts)
                            self._spans[-1]["tone"] = tone
                            self._spans[-1]["label"] = label
                            active_span = dict(self._spans[-1])
                            state_changed = True
                        else:
                            state_changed = True
                            new_span = {
                                "start": float(popped["start"]),
                                "end": max(float(popped["start"]), ts),
                                "state": mapped_state,
                                "tone": tone,
                                "label": label,
                            }
                            self._spans.append(new_span)
                            active_span = dict(new_span)
                    else:
                        state_changed = True
                        new_span = {
                            "start": ts,
                            "end": ts,
                            "state": mapped_state,
                            "tone": tone,
                            "label": label,
                        }
                        self._spans.append(new_span)
                        active_span = dict(new_span)
            else:
                state_changed = True
                new_span = {
                    "start": ts,
                    "end": ts,
                    "state": mapped_state,
                    "tone": tone,
                    "label": label,
                }
                self._spans.append(new_span)
                active_span = dict(new_span)

            self._prune_locked(ts)

            mono_elapsed = time.monotonic() - self._last_saved_mono
            sim_elapsed = abs(ts - self._last_saved_ts)
            if (
                state_changed
                or explicit_ts
                or sim_elapsed >= self.DISK_FLUSH_INTERVAL_SEC
                or mono_elapsed >= self.DISK_FLUSH_INTERVAL_SEC
            ):
                self._save_locked(ts)

            return active_span

    def to_dict(
        self,
        now_ts: float | None = None,
        view_window_sec: float | None = None,
    ) -> dict[str, Any]:
        ts = float(time.time() if now_ts is None else now_ts)
        if view_window_sec is None:
            win = self.window_sec
        else:
            win = max(60.0, min(self.window_sec, float(view_window_sec)))
        cutoff = ts - win
        with self._lock:
            visible: list[dict[str, Any]] = []
            for sp in self._spans:
                en = float(sp["end"])
                st = float(sp["start"])
                if en < cutoff or (en == cutoff and st < cutoff):
                    continue
                clamped_start = cutoff if st < cutoff else st
                visible.append(
                    {
                        "start": clamped_start,
                        "end": max(clamped_start, en),
                        "state": sp["state"],
                        "tone": sp["tone"],
                        "label": sp["label"],
                    }
                )
            return {
                "window_sec": win,
                "start_ts": cutoff,
                "now_ts": ts,
                "spans": visible,
            }

    def spans_between(self, start_ts: float, end_ts: float) -> list[dict[str, Any]]:
        """Копии интервалов, обрезанные по окну [start_ts, end_ts) — для ночной сводки."""
        out: list[dict[str, Any]] = []
        with self._lock:
            for sp in self._spans:
                st = max(float(start_ts), float(sp["start"]))
                en = min(float(end_ts), float(sp["end"]))
                if en <= st:
                    continue
                item = dict(sp)
                item["start"], item["end"] = st, en
                out.append(item)
        return out

    def slice_range(
        self,
        start_ts: float,
        end_ts: float,
    ) -> dict[str, Any]:
        """Возвращает интервалы хроники строго в заданном диапазоне [start_ts, end_ts]."""
        st_bound = float(start_ts)
        en_bound = max(st_bound, float(end_ts))
        with self._lock:
            visible: list[dict[str, Any]] = []
            for sp in self._spans:
                en = float(sp["end"])
                st = float(sp["start"])
                if en < st_bound or st > en_bound:
                    continue
                if en == st_bound and st < st_bound:
                    continue
                clamped_st = max(st_bound, st)
                clamped_en = min(en_bound, max(clamped_st, en))
                visible.append(
                    {
                        "start": clamped_st,
                        "end": clamped_en,
                        "state": sp["state"],
                        "tone": sp["tone"],
                        "label": sp["label"],
                    }
                )
            return {
                "window_sec": en_bound - st_bound,
                "start_ts": st_bound,
                "now_ts": en_bound,
                "spans": visible,
            }

    def _summarize_interval_locked(
        self,
        cutoff: float,
        ts: float,
        win: float,
    ) -> dict[str, Any]:
        in_bed_sec = 0.0
        calm_sec = 0.0
        caregiver_sec = 0.0
        paused_sec = 0.0
        offline_sec = 0.0
        tracked_sec = 0.0

        sit_up_count = 0
        raw_left_bed_count = 0
        missing_count = 0
        out_of_bed_episodes = 0
        fall_count = 0
        help_call_count = 0
        caregiver_visits_count = 0

        prev_out_end: float | None = None

        for sp in self._spans:
            en = float(sp["end"])
            st = float(sp["start"])
            if en < cutoff or (en == cutoff and st < cutoff) or st > ts:
                continue
            clamped_st = max(cutoff, st)
            clamped_en = min(ts, max(clamped_st, en))
            dur = max(0.0, clamped_en - clamped_st)
            state = str(sp.get("state") or "idle")

            if state == "in_bed":
                in_bed_sec += dur
                calm_sec += dur
                tracked_sec += dur
                prev_out_end = None
            elif state == "idle":
                calm_sec += dur
                tracked_sec += dur
            elif state == "caregiver":
                caregiver_sec += dur
                tracked_sec += dur
                caregiver_visits_count += 1
                prev_out_end = None
            elif state == "paused":
                paused_sec += dur
                tracked_sec += dur
                prev_out_end = None
            elif state == "offline":
                offline_sec += dur
            else:
                tracked_sec += dur
                if state == "sitting_up":
                    sit_up_count += 1
                elif state == "left_bed":
                    raw_left_bed_count += 1
                    if prev_out_end is None or (clamped_st - prev_out_end) > 60.0:
                        out_of_bed_episodes += 1
                    prev_out_end = clamped_en
                elif state == "missing":
                    missing_count += 1
                    if prev_out_end is None or (clamped_st - prev_out_end) > 60.0:
                        out_of_bed_episodes += 1
                    prev_out_end = clamped_en
                elif state == "fallen":
                    fall_count += 1
                elif state == "help_call":
                    help_call_count += 1

        return {
            "window_sec": win,
            "in_bed_sec": round(in_bed_sec, 1),
            "calm_sec": round(calm_sec, 1),
            "caregiver_sec": round(caregiver_sec, 1),
            "paused_sec": round(paused_sec, 1),
            "offline_sec": round(offline_sec, 1),
            "tracked_sec": round(tracked_sec, 1),
            "sit_up_count": sit_up_count,
            "left_bed_count": out_of_bed_episodes,
            "raw_left_bed_count": raw_left_bed_count,
            "missing_count": missing_count,
            "fall_count": fall_count,
            "help_call_count": help_call_count,
            "caregiver_visits_count": caregiver_visits_count,
        }

    def summary_range(
        self,
        start_ts: float,
        end_ts: float,
    ) -> dict[str, Any]:
        st = float(start_ts)
        en = max(st + 1.0, float(end_ts))
        with self._lock:
            return self._summarize_interval_locked(st, en, en - st)

    def summary_stats(
        self,
        now_ts: float | None = None,
        window_sec: float = 86400.0,
    ) -> dict[str, Any]:
        ts = float(time.time() if now_ts is None else now_ts)
        win = max(60.0, float(window_sec))
        cutoff = ts - win
        with self._lock:
            return self._summarize_interval_locked(cutoff, ts, win)
