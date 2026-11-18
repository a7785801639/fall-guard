from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app.care_history import CareHistory, night_window, summarize_spans, trend
from app.engine import FallGuardEngine
from app.timeline import TimelineStore
from app.web import create_web_app
from test_web_api import _build_test_settings


def _ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp()


def test_night_window_wraps_midnight_and_handles_same_day_hours() -> None:
    st, en = night_window("2026-03-10", 22, 7)
    assert st == _ts("2026-03-10 22:00") and en == _ts("2026-03-11 07:00")
    st, en = night_window("2026-03-10", 0, 6)
    assert st == _ts("2026-03-10 00:00") and en == _ts("2026-03-10 06:00")


def test_summary_counts_getups_as_episodes_and_coverage() -> None:
    start, end = _ts("2026-03-10 22:00"), _ts("2026-03-11 07:00")
    spans = [
        {"start": start, "end": start + 3600 * 2, "state": "in_bed"},
        {"start": start + 7200, "end": start + 7230, "state": "sitting_up"},
        {"start": start + 7230, "end": start + 7300, "state": "left_bed"},
        {"start": start + 7320, "end": start + 7400, "state": "missing"},  # вернулась через 20 с — тот же выход
        {"start": start + 7400, "end": start + 3600 * 5, "state": "in_bed"},
        {"start": start + 3600 * 5, "end": start + 3600 * 5 + 60, "state": "left_bed"},  # второй выход
        {"start": start + 3600 * 5 + 60, "end": start + 3600 * 6, "state": "in_bed"},
        {"start": start + 3600 * 6, "end": end, "state": "offline"},  # камера не работала
    ]
    rec = summarize_spans(spans, start, end)
    assert rec["getups"] == 2 and rec["sit_ups"] == 1
    assert abs(rec["coverage"] - 6 / 9) < 0.01 and rec["valid"] is True
    assert rec["in_bed_sec"] > 5 * 3600


def _nights(getups: list[int], bed_h: float = 7.0, valid: bool = True) -> list[dict]:
    return [{"day": f"2026-01-{i + 1:02d}", "getups": g, "in_bed_sec": bed_h * 3600, "valid": valid} for i, g in enumerate(getups)]


def test_trend_needs_enough_nights_and_a_real_difference() -> None:
    assert trend(_nights([1] * 5))["ready"] is False
    calm = trend(_nights([1] * 14 + [1, 2, 1, 1, 2, 1, 1]))
    assert calm["ready"] is True and calm["level"] == "calm"
    worse = trend(_nights([1] * 14 + [3, 4, 3, 3, 4, 3, 3]))
    assert worse["level"] == "warn" and "чаще обычного" in worse["items"][0]["text"]
    # «Втрое» от почти нуля — шум, не тревожим
    tiny = trend(_nights([0] * 14 + [1, 0, 1, 0, 1, 0, 1]))
    assert tiny["level"] == "calm"
    # Ночи, где камера почти не работала, не считаются
    assert trend(_nights([5] * 20, valid=False))["valid_nights"] == 0


def test_rollup_stores_finished_night_and_falls_are_persistent(tmp_path: Path) -> None:
    tl = TimelineStore(tmp_path / "tl.json", window_sec=86400.0)
    start, end = night_window("2026-03-10", 22, 7)
    # Движок пишет состояние постоянно — имитируем запись каждые 20 с
    t = start
    while t < end:
        state = "left_bed" if start + 3600 <= t < start + 3660 else "in_bed"
        tl.record(state, True, now_ts=t)
        t += 20
    hist = CareHistory(tmp_path / "care.json")
    assert hist.rollup(tl, 22, 7, now_ts=end + 600) is True
    assert hist.rollup(tl, 22, 7, now_ts=end + 900) is False  # уже записано
    night = hist.nights()[0]
    assert night["day"] == "2026-03-10" and night["getups"] == 1 and night["valid"]
    hist.record_fall("a1", end + 100, "Падение", "Сползла с кровати")
    hist.record_fall("a1", end + 100, "Падение", "повтор не дублируется")
    again = CareHistory(tmp_path / "care.json")
    assert len(again.falls()) == 1 and again.nights()[0]["getups"] == 1


def test_stats_endpoint_returns_nights_trend_and_falls(tmp_path: Path) -> None:
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    engine.care.record_fall("f1", __import__("time").time() - 3600, "Падение", "На полу у кровати")
    data = TestClient(create_web_app(engine)).get("/api/stats/nights?days=28").json()
    assert data["ok"] and data["night_hours"] == [22, 7]
    assert data["falls"][0]["id"] == "f1" and data["falls"][0]["false_alarm"] is False
    assert "trend" in data and "nights" in data
