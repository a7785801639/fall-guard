from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app.devices import duty_covers, duty_on, normalize_duty
from app.engine import FallGuardEngine
from app.state_machine import AlertEvent, AlertSeverity, PatientState
from app.web import create_web_app
from test_web_api import _build_test_settings

EXTERNAL = {"host": "nofall.example.com", "x-forwarded-proto": "https"}


def test_duty_hours_wrap_midnight_and_follow_night_settings() -> None:
    night = {"mode": "night"}
    assert duty_on(night, 23) and duty_on(night, 3) and not duty_on(night, 12)
    assert duty_on({"mode": "day"}, 12) and not duty_on({"mode": "day"}, 2)
    custom = {"mode": "custom", "from": 20, "to": 8}
    assert duty_on(custom, 21) and duty_on(custom, 7) and not duty_on(custom, 9)
    assert duty_on({"mode": "custom", "from": 9, "to": 18}, 10) and not duty_on({"mode": "custom", "from": 9, "to": 18}, 19)
    # Ночь по настройкам ночного режима: 23–6
    assert duty_on(night, 23, 23, 6) and not duty_on(night, 22, 23, 6)
    # Без расписания — всегда; мусор — тоже всегда
    assert duty_on(None, 4) and normalize_duty({"mode": "?"})["mode"] == "always"


def test_falls_reach_off_duty_people_unless_they_opted_out() -> None:
    day = {"mode": "day"}
    assert duty_covers(day, "fallen", 3) is True
    assert duty_covers(day, "left_bed", 3) is False
    assert duty_covers({**day, "falls_always": False}, "fallen", 3) is False


def _engine_with_phones(tmp_path: Path) -> tuple[FallGuardEngine, TestClient, dict[str, str], dict[str, str]]:
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    client = TestClient(create_web_app(engine))
    tokens = {}
    for name in ("Анна", "Борис"):
        code = client.post("/api/devices/pair-code").json()["code"]
        client.cookies.clear()
        tokens[name] = client.post("/api/devices/claim", json={"code": code, "name": name}, headers=EXTERNAL).json()["token"]
        client.cookies.clear()
    ids = {d["name"]: d["id"] for d in engine.devices.list()}
    for name, tok in tokens.items():  # оба телефона «на связи»
        client.get("/api/alerts/poll?after=0&wait=0", headers={**EXTERNAL, "authorization": f"Bearer {tok}"})
    # Телеграм-получатель из настроек тестов не дежурит ночью, чтобы не мешать проверке телефонов
    for sub in engine.notifier.list_subscribers():
        engine.notifier.set_subscriber_duty(str(sub["chat_id"]), {"mode": "day", "falls_always": False})
    return engine, client, tokens, ids


def _hour_ts(hour: int) -> float:
    return datetime.now().replace(hour=hour, minute=30, second=0, microsecond=0).timestamp()


def test_only_the_on_duty_phone_gets_night_events_but_falls_go_to_all(tmp_path: Path) -> None:
    engine, _, _, ids = _engine_with_phones(tmp_path)
    engine.devices.update(ids["Анна"], duty={"mode": "night"})
    engine.devices.update(ids["Борис"], duty={"mode": "day"})
    night = _hour_ts(3)
    assert engine.duty_allows("device", ids["Анна"], "left_bed", "warning", night) is True
    assert engine.duty_allows("device", ids["Борис"], "left_bed", "warning", night) is False
    assert engine.duty_allows("device", ids["Борис"], "fallen", "critical", night) is True  # падения — всем
    # Днём наоборот
    assert engine.duty_allows("device", ids["Борис"], "left_bed", "warning", _hour_ts(13)) is True
    assert engine.duty_allows("device", ids["Анна"], "left_bed", "warning", _hour_ts(13)) is False


def test_nobody_on_duty_means_everyone_gets_it(tmp_path: Path) -> None:
    engine, _, _, ids = _engine_with_phones(tmp_path)
    engine.devices.update(ids["Анна"], duty={"mode": "custom", "from": 9, "to": 12})
    engine.devices.update(ids["Борис"], duty={"mode": "custom", "from": 14, "to": 17})
    assert engine.duty_allows("device", ids["Анна"], "left_bed", "warning", _hour_ts(3)) is True
    assert engine.duty_allows("device", ids["Борис"], "left_bed", "warning", _hour_ts(3)) is True


def test_a_silent_on_duty_phone_does_not_swallow_alerts(tmp_path: Path) -> None:
    engine, _, _, ids = _engine_with_phones(tmp_path)
    engine.devices.update(ids["Анна"], duty={"mode": "night"})
    engine.devices.update(ids["Борис"], duty={"mode": "day"})
    # Анна дежурит, но её телефон давно молчит — тогда ночное событие получит и Борис
    with engine.devices._lock:
        engine.devices._devices[ids["Анна"]]["last_seen"] = time.time() - 3600
    assert engine.duty_allows("device", ids["Борис"], "left_bed", "warning", _hour_ts(3)) is True


def test_poll_filters_off_duty_events_and_reports_duty(tmp_path: Path) -> None:
    engine, client, tokens, ids = _engine_with_phones(tmp_path)
    hour = datetime.now().hour
    on, off = {"mode": "custom", "from": hour, "to": (hour + 1) % 24}, {"mode": "custom", "from": (hour + 2) % 24, "to": (hour + 3) % 24}
    engine.devices.update(ids["Анна"], duty=on)
    engine.devices.update(ids["Борис"], duty=off)
    engine.alert_feed.push("a1", "left_bed", "warning", "Встала", "Ушла от кровати")
    get = lambda n: client.get("/api/alerts/poll?after=0&wait=0", headers={**EXTERNAL, "authorization": f"Bearer {tokens[n]}"}).json()
    a, b = get("Анна"), get("Борис")
    assert [x["id"] for x in a["alerts"]] == ["a1"] and a["on_duty"] is True
    assert b["alerts"] == [] and b["on_duty"] is False
    me = client.get("/api/me", headers={**EXTERNAL, "authorization": f"Bearer {tokens['Борис']}"}).json()["me"]
    assert me["duty"]["mode"] == "custom" and me["on_duty"] is False


def test_telegram_gate_applies_duty(tmp_path: Path) -> None:
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    cid = str(engine.notifier.list_subscribers()[0]["chat_id"])
    hour = datetime.now().hour
    engine.notifier.set_subscriber_duty(cid, {"mode": "custom", "from": (hour + 2) % 24, "to": (hour + 3) % 24})
    # Ещё один дежурный, чтобы не сработало «никого нет — всем»
    engine.notifier.register_subscriber("777", name="Дежурный")
    ev = AlertEvent(state=PatientState.LEFT_BED, severity=AlertSeverity.WARNING, title="Встала", description="", timestamp=time.time(), patient_track_id=None)
    assert cid not in engine.notifier._target_chats(event=ev)
    assert "777" in engine.notifier._target_chats(event=ev)
    fall = AlertEvent(state=PatientState.FALLEN, severity=AlertSeverity.CRITICAL, title="Падение", description="", timestamp=time.time(), patient_track_id=None)
    assert cid in engine.notifier._target_chats(event=fall)
