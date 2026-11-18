"""
История для графиков и отчёта врачу: сводка за каждую ночь и список падений.

Хроника состояний (timeline.json) хранит только последние сутки, поэтому каждую закончившуюся
ночь сворачиваем в одну запись и храним долго. Падения пишем отдельно, по идентификатору тревоги:
архив записей можно очистить, а журнал событий ограничен по длине.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

# Ночь, где камера видела меньше этой доли времени, на графике помечается и не идёт в «обычное»
MIN_COVERAGE = 0.6
# Тренд: сколько последних ночей сравниваем с какими «обычными»
RECENT_NIGHTS = 7
BASELINE_NIGHTS = 28
MIN_RECENT_VALID = 4
MIN_BASELINE_VALID = 7
KEEP_NIGHTS = 400
KEEP_FALLS = 2000
OUT_STATES = ("left_bed", "missing")
EPISODE_MERGE_SEC = 60.0


def night_window(day: str, start_hour: int, end_hour: int) -> tuple[float, float]:
    """Окно ночи по местному времени сервера. day — дата вечера, с которого ночь начинается."""
    d = datetime.strptime(day, "%Y-%m-%d")
    start = d.replace(hour=int(start_hour) % 24)
    end = d.replace(hour=int(end_hour) % 24)
    if end <= start:
        end += timedelta(days=1)
    return start.timestamp(), end.timestamp()


def last_finished_night(now_ts: float, start_hour: int, end_hour: int) -> str:
    """Дата вечера последней полностью закончившейся ночи."""
    now = datetime.fromtimestamp(now_ts)
    for back in range(0, 3):
        day = (now - timedelta(days=back)).strftime("%Y-%m-%d")
        _, end = night_window(day, start_hour, end_hour)
        if end <= now_ts:
            return day
    return (now - timedelta(days=3)).strftime("%Y-%m-%d")


def summarize_spans(spans: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    """Сводка ночи по интервалам хроники (уже обрезанным по окну)."""
    length = max(1.0, end - start)
    in_bed = tracked = caregiver = paused = 0.0
    sit_ups = falls = help_calls = 0
    getup_times: list[float] = []
    last_out_end: float | None = None
    prev_state = ""
    for sp in sorted(spans, key=lambda x: float(x["start"])):
        st, en = float(sp["start"]), float(sp["end"])
        dur = max(0.0, en - st)
        state = str(sp.get("state") or "idle")
        if state == "offline":
            prev_state = state
            continue
        tracked += dur
        if state == "in_bed":
            in_bed += dur
        elif state == "caregiver":
            caregiver += dur
        elif state == "paused":
            paused += dur
        elif state == "sitting_up":
            if prev_state != "sitting_up":
                sit_ups += 1
        elif state in OUT_STATES:
            # Вышла и вернулась в течение минуты — это один и тот же выход
            if last_out_end is None or st - last_out_end > EPISODE_MERGE_SEC:
                getup_times.append(round(st, 1))
            last_out_end = en
        elif state == "fallen":
            if prev_state != "fallen":
                falls += 1
        elif state == "help_call":
            if prev_state != "help_call":
                help_calls += 1
        prev_state = state
    coverage = tracked / length
    return {
        "start": round(start, 1),
        "end": round(end, 1),
        "in_bed_sec": round(in_bed, 1),
        "tracked_sec": round(tracked, 1),
        "caregiver_sec": round(caregiver, 1),
        "paused_sec": round(paused, 1),
        "coverage": round(coverage, 3),
        "valid": coverage >= MIN_COVERAGE,
        "getups": len(getup_times),
        "getup_times": getup_times,
        "sit_ups": sit_ups,
        "falls": falls,
        "help_calls": help_calls,
    }


def care_trend_window() -> int:
    """Сколько ночей нужно для тренда: последняя неделя и обычные ночи до неё."""
    return RECENT_NIGHTS + BASELINE_NIGHTS


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def trend(nights: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Сравнение последней недели с обычными ночами до неё. Предупреждаем, только если данных
    достаточно и разница заметная и по разам, и по абсолютному числу: «втрое чаще» от 0,3 — шум.
    """
    valid = [n for n in sorted(nights, key=lambda x: x["day"]) if n.get("valid")]
    recent = valid[-RECENT_NIGHTS:]
    baseline = valid[-(RECENT_NIGHTS + BASELINE_NIGHTS):-RECENT_NIGHTS] if len(valid) > RECENT_NIGHTS else []
    out: dict[str, Any] = {
        "ready": len(recent) >= MIN_RECENT_VALID and len(baseline) >= MIN_BASELINE_VALID,
        "valid_nights": len(valid),
        "need_nights": MIN_RECENT_VALID + MIN_BASELINE_VALID,
        "level": "calm",
        "items": [],
    }
    if not recent:
        return out
    r_get = _avg([n["getups"] for n in recent])
    r_bed = _avg([n["in_bed_sec"] for n in recent]) / 3600.0
    out["recent"] = {"getups": round(r_get, 2), "in_bed_h": round(r_bed, 2), "nights": len(recent)}
    if not out["ready"]:
        return out
    b_get = _avg([n["getups"] for n in baseline])
    b_bed = _avg([n["in_bed_sec"] for n in baseline]) / 3600.0
    out["baseline"] = {"getups": round(b_get, 2), "in_bed_h": round(b_bed, 2), "nights": len(baseline)}
    ratio = r_get / max(b_get, 0.5)
    out["getups_ratio"] = round(ratio, 2)
    if r_get - b_get >= 1.0 and ratio >= 2.0:
        out["level"] = "warn"
        out["items"].append({
            "kind": "getups",
            "text": f"За неделю вставала в среднем {r_get:.1f} раза за ночь — в {ratio:.1f} раза чаще обычного ({b_get:.1f})".replace(".", ","),
        })
    if b_bed - r_bed >= 1.5:
        out["level"] = "warn"
        out["items"].append({
            "kind": "in_bed",
            "text": f"Ночью стала меньше лежать в кровати: {r_bed:.1f} ч вместо обычных {b_bed:.1f} ч".replace(".", ","),
        })
    return out


class CareHistory:
    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time) -> None:
        self._path = Path(path)
        self._clock = clock
        self._lock = threading.RLock()
        self._nights: dict[str, dict[str, Any]] = {}
        self._falls: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------ диск
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        for n in raw.get("nights", []) if isinstance(raw, dict) else []:
            if isinstance(n, dict) and n.get("day"):
                self._nights[str(n["day"])] = n
        self._falls = [f for f in (raw.get("falls", []) if isinstance(raw, dict) else []) if isinstance(f, dict) and f.get("id")]

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        days = sorted(self._nights)[-KEEP_NIGHTS:]
        self._nights = {d: self._nights[d] for d in days}
        self._falls = self._falls[-KEEP_FALLS:]
        data = {"schema_version": 1, "nights": [self._nights[d] for d in days], "falls": self._falls}
        tmp = self._path.with_name(f"{self._path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    # ------------------------------------------------------------ падения
    def record_fall(self, alert_id: str, ts: float, title: str = "", description: str = "", clip_id: str | None = None) -> None:
        with self._lock:
            if any(f["id"] == alert_id for f in self._falls):
                return
            self._falls.append({
                "id": str(alert_id), "ts": round(float(ts), 1), "title": str(title)[:200],
                "description": str(description)[:400], "clip_id": clip_id or "",
            })
            self._save_locked()

    def falls(self, since_ts: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(f) for f in self._falls if f["ts"] >= since_ts]

    # ------------------------------------------------------------ ночи
    def rollup(self, timeline: Any, start_hour: int, end_hour: int, now_ts: float | None = None) -> bool:
        """
        Сворачивает закончившиеся ночи, которые ещё есть в хронике (она хранит сутки), в записи.
        Дёшево, если всё уже записано. Возвращает True, если что-то добавилось.
        """
        now = float(self._clock() if now_ts is None else now_ts)
        last = last_finished_night(now, start_hour, end_hour)
        added = False
        with self._lock:
            for back in range(0, 2):
                day = (datetime.strptime(last, "%Y-%m-%d") - timedelta(days=back)).strftime("%Y-%m-%d")
                if day in self._nights:
                    continue
                start, end = night_window(day, start_hour, end_hour)
                if end < now - 86400.0:  # хроника этого уже не помнит
                    continue
                spans = timeline.spans_between(start, end)
                if not spans:
                    continue
                rec = summarize_spans(spans, start, end)
                rec["day"] = day
                rec["hours"] = [int(start_hour), int(end_hour)]
                self._nights[day] = rec
                added = True
            if added:
                self._save_locked()
        return added

    def current_night(self, timeline: Any, start_hour: int, end_hour: int, now_ts: float | None = None) -> dict[str, Any] | None:
        """Идущая сейчас ночь — «пока что», для графика (не хранится)."""
        now = float(self._clock() if now_ts is None else now_ts)
        for back in (0, 1):
            day = (datetime.fromtimestamp(now) - timedelta(days=back)).strftime("%Y-%m-%d")
            start, end = night_window(day, start_hour, end_hour)
            if start <= now < end:
                rec = summarize_spans(timeline.spans_between(start, now), start, end)
                rec["day"] = day
                rec["partial"] = True
                rec["valid"] = False
                return rec
        return None

    def nights(self, days: int = 28) -> list[dict[str, Any]]:
        with self._lock:
            keys = sorted(self._nights)[-max(1, int(days)):]
            return [dict(self._nights[k]) for k in keys]
