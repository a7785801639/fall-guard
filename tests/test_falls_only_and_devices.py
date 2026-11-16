from __future__ import annotations

from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.devices import AlertFeed, DeviceStore
from app.engine import FallGuardEngine
from app.notifier import TelegramNotifier, _build_reply_keyboard
from app.patient_id import Role, TrackIdentity
from app.state_machine import AlertEvent, AlertSeverity, FallStateMachine, PatientState
from app.web import create_web_app

from test_fallguard import BED_ZONE, make_pose
from test_web_api import _build_test_settings

EXTERNAL = {"host": "nofall.example.com", "x-forwarded-proto": "https"}


def _ident(tid: int, role: Role) -> TrackIdentity:
    return TrackIdentity(track_id=tid, role=role, has_moved=True)


def _in_bed(tid: int = 1):
    return make_pose(tid, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))


def _on_floor(tid: int, dx: float = 0.0):
    return make_pose(tid, (0.30 + dx, 0.76, 0.65 + dx, 0.94), (0.36 + dx, 0.85), (0.56 + dx, 0.86))


def _standing(tid: int):
    return make_pose(tid, (0.80, 0.22, 0.92, 0.86), (0.86, 0.32), (0.86, 0.56))


# --- Страховка падения от путаницы ролей ---------------------------------------------

def test_fall_is_alerted_when_visit_is_only_latched_and_nobody_is_visible() -> None:
    """Визит «защёлкнут» (caregiver_nearby), но в кадре помощника нет — падение не глушится."""
    fsm = FallStateMachine(fall_confirm_seconds=1.0)
    roles = {1: _ident(1, Role.PATIENT)}
    t = 0.0
    for _ in range(8):
        t += 0.25
        assert fsm.step([_in_bed()], roles, False, BED_ZONE, now=t) is None

    events = []
    for _ in range(16):
        t += 0.25
        ev = fsm.step([_on_floor(1)], roles, True, BED_ZONE, now=t, caregiver_visible=False)
        if ev is not None:
            events.append(ev)
    assert len(events) == 1
    assert events[0].state == PatientState.FALLEN
    assert events[0].severity == AlertSeverity.CRITICAL
    assert fsm.fall_unacknowledged is True


def test_fall_stays_suppressed_while_caregiver_is_really_visible() -> None:
    fsm = FallStateMachine(fall_confirm_seconds=1.0)
    roles = {1: _ident(1, Role.PATIENT), 2: _ident(2, Role.VISITOR)}
    t = 0.0
    for _ in range(20):
        t += 0.25
        ev = fsm.step([_on_floor(1), _standing(2)], roles, True, BED_ZONE, now=t, caregiver_visible=True)
        assert ev is None
    assert fsm.state == PatientState.CAREGIVER_ASSIST


def test_person_down_safety_net_alerts_even_when_role_is_not_patient() -> None:
    """Бабушку приняли за «своего»: она неподвижно лежит на полу одна — тревога всё равно есть."""
    fsm = FallStateMachine(fall_confirm_seconds=1.0)
    roles = {5: _ident(5, Role.VISITOR)}
    t, events = 0.0, []
    for _ in range(40):
        t += 0.25
        ev = fsm.step([_on_floor(5)], roles, True, BED_ZONE, now=t, caregiver_visible=True)
        if ev is not None:
            events.append((t, ev))
    assert len(events) == 1
    first_t, ev = events[0]
    assert first_t >= 6.0
    assert ev.state == PatientState.FALLEN and "на полу" in ev.title
    assert fsm.state == PatientState.FALLEN

    # «Ты ошибся» — по тому же месту 10 минут не тревожим
    fsm.dismiss_false_alert()
    for _ in range(60):
        t += 0.25
        assert fsm.step([_on_floor(5)], roles, False, BED_ZONE, now=t) is None


def test_person_down_safety_net_ignores_helped_moving_or_disabled_cases() -> None:
    roles = {5: _ident(5, Role.VISITOR), 6: _ident(6, Role.VISITOR)}

    # Рядом стоит второй человек — помощь уже есть
    fsm = FallStateMachine(fall_confirm_seconds=1.0)
    t = 0.0
    for _ in range(48):
        t += 0.25
        assert fsm.step([_on_floor(5), _standing(6)], roles, True, BED_ZONE, now=t) is None

    # Человек на полу перемещается (моет пол) — отсчёт неподвижности сбрасывается
    fsm = FallStateMachine(fall_confirm_seconds=1.0)
    t = 0.0
    for i in range(48):
        t += 0.25
        assert fsm.step([_on_floor(5, dx=0.09 * ((i // 8) % 2))], roles, False, BED_ZONE, now=t) is None

    # Страховка выключена в настройках
    fsm = FallStateMachine(fall_confirm_seconds=1.0, floor_safety_net=False)
    t = 0.0
    for _ in range(48):
        t += 0.25
        assert fsm.step([_on_floor(5)], roles, False, BED_ZONE, now=t) is None


# --- Режим «только падение / сползание» у каждого получателя -------------------------

def _event(state: PatientState, severity: AlertSeverity = AlertSeverity.WARNING) -> AlertEvent:
    return AlertEvent(state, severity, "Заголовок", "Описание", 1.0, 1)


def test_falls_only_subscriber_gets_only_fall_alerts_and_can_toggle_from_bot(tmp_path: Path) -> None:
    photos: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        body = request.read().decode("utf-8", "ignore")
        if endpoint == "sendPhoto":
            for cid in ("1001", "2002"):
                if f'name="chat_id"\r\n\r\n{cid}' in body:
                    photos.append(cid)
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})
        return httpx.Response(200, json={"ok": True, "result": True})

    transport = httpx.MockTransport(handler)
    notifier = TelegramNotifier(
        bot_token="123456:MOCK",
        chat_ids=("1001", "2002"),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: b"\xff\xd8\xff\xd9",
        is_fall_unacknowledged=lambda: False,
        subscribers_path=tmp_path / "subs.json",
        transport=transport,
    )

    # 2002 нажимает кнопку «Только падения» в боте
    with httpx.Client(transport=transport) as client:
        notifier._handle_update(
            client,
            {"update_id": 1, "message": {"chat": {"id": 2002}, "from": {"id": 2002, "first_name": "Борис"}, "text": "🚨 Только падения"}},
        )
    assert notifier.get_subscriber_mode("2002") == "falls"
    assert notifier.get_subscriber_mode("1001") == "all"
    assert "📣 Все события" in _build_reply_keyboard(True, "falls")

    notifier.send_alert_photo(_event(PatientState.SITTING_UP), b"jpg", alert_id="a1")
    assert photos == ["1001"]
    photos.clear()
    notifier.send_alert_photo(_event(PatientState.LEFT_BED, AlertSeverity.CRITICAL), b"jpg", alert_id="a2")
    assert photos == ["1001"]
    photos.clear()
    notifier.send_alert_photo(_event(PatientState.FALLEN, AlertSeverity.CRITICAL), b"jpg", alert_id="a3")
    assert photos == ["1001", "2002"]

    # Режим переживает перезапуск
    reloaded = TelegramNotifier(
        bot_token="123456:MOCK",
        chat_ids=("1001", "2002"),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: None,
        is_fall_unacknowledged=lambda: False,
        subscribers_path=tmp_path / "subs.json",
        transport=transport,
    )
    assert reloaded.get_subscriber_mode("2002") == "falls"
    assert reloaded.set_subscriber_mode("2002", "all")["mode"] == "all"


# --- Телефоны: привязка, токены, лента тревог -----------------------------------------

def test_device_store_pairing_token_hashing_and_lockout(tmp_path: Path) -> None:
    clock = {"t": 1000.0}
    store = DeviceStore(tmp_path / "devices.json", wall_clock=lambda: clock["t"])

    assert store.claim("WRONG123") is None
    code, ttl = store.create_pair_code()
    assert len(code) == 8 and ttl == 600.0
    device, token = store.claim(code.lower(), name="Pixel")
    assert token.startswith("fgd_") and device["mode"] == "all"
    assert store.claim(code) is None, "код одноразовый"
    assert token not in (tmp_path / "devices.json").read_text(encoding="utf-8")

    assert store.authenticate(token)["id"] == device["id"]
    assert store.authenticate(token + "x") is None
    assert DeviceStore(tmp_path / "devices.json").authenticate(token) is not None

    expired, _ = store.create_pair_code()
    clock["t"] += 601.0
    assert store.claim(expired) is None

    for _ in range(12):
        store.claim("BADBAD00")
    assert store.claims_locked_out() is True
    fresh, _ = store.create_pair_code()
    assert store.claim(fresh) is None, "перебор кодов блокируется"
    clock["t"] += 700.0
    assert store.claim(fresh) is None and store.claims_locked_out() is False

    assert store.update(device["id"], mode="falls")["mode"] == "falls"
    assert store.remove(device["id"]) is True and store.authenticate(token) is None


def test_alert_feed_filters_by_mode_and_cursor() -> None:
    feed = AlertFeed()
    a = feed.push("a", "sitting_up", "warning", "Садится", "")
    b = feed.push("b", "fallen", "critical", "Падение", "")
    assert b["seq"] > a["seq"] and feed.cursor == b["seq"]
    assert [x["id"] for x in feed.since(0)] == ["a", "b"]
    assert [x["id"] for x in feed.since(0, mode="falls")] == ["b"]
    assert feed.since(b["seq"]) == []


def test_device_pairing_api_external_auth_and_per_device_mode(tmp_path: Path) -> None:
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    client = TestClient(create_web_app(engine))

    # Снаружи без входа пульт закрыт, а привязка без кода невозможна
    assert client.get("/api/status", headers=EXTERNAL).status_code == 401
    assert client.post("/api/devices/claim", json={"code": ""}, headers=EXTERNAL).status_code == 403
    assert client.post("/api/devices/pair-code", headers=EXTERNAL).status_code == 401

    pair = client.post("/api/devices/pair-code").json()
    assert pair["link"].startswith("fallguard://pair?") and "<svg" in pair["qr_svg"]
    client.cookies.clear()
    claimed = client.post("/api/devices/claim", json={"code": pair["code"], "name": "Pixel"}, headers=EXTERNAL)
    assert claimed.status_code == 200
    token = claimed.json()["token"]
    client.cookies.clear()
    auth = {**EXTERNAL, "authorization": f"Bearer {token}"}

    assert client.get("/api/status", headers=auth).status_code == 200
    assert client.get("/api/status", headers={**EXTERNAL, "cookie": f"fg_device={token}"}).status_code == 200
    session = client.get("/api/tg/session", headers=auth).json()
    assert session["authenticated"] is True and session["mode"] == "device"

    me = client.get("/api/me", headers=auth).json()["me"]
    assert me["kind"] == "device" and me["mode"] == "all"
    assert client.post("/api/me/mode", json={"mode": "falls"}, headers=auth).json()["me"]["mode"] == "falls"
    assert client.post("/api/me/mode", json={"mode": "falls"}).status_code == 400, "LAN-браузер не получатель"
    assert engine.get_status()["devices"][0]["mode"] == "falls"

    # В режиме «только падения» лента отдаёт телефону только падение
    engine.alert_feed.push("s1", "sitting_up", "warning", "Садится", "")
    polled = client.get("/api/alerts/poll?after=1&wait=0", headers=auth).json()
    assert polled["alerts"] == [] and polled["mode"] == "falls"
    engine.alert_feed.push("f1", "fallen", "critical", "Падение", "")
    polled = client.get("/api/alerts/poll?after=1&wait=0", headers=auth).json()
    assert [a["id"] for a in polled["alerts"]] == ["f1"] and polled["cursor"] == polled["alerts"][0]["seq"]
    assert client.get(f"/api/alerts/poll?after={polled['cursor']}&wait=0", headers=auth).json()["alerts"] == []

    # Отвязанный телефон теряет доступ
    assert client.delete(f"/api/devices/{me['id']}").status_code == 200
    assert client.get("/api/status", headers=auth).status_code == 401


def test_floor_safety_net_pref_is_persisted_and_applied(tmp_path: Path) -> None:
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    assert engine.get_prefs()["floor_safety_net"] is True
    assert engine.update_prefs({"floor_safety_net": False})["floor_safety_net"] is False
    assert engine.fsm._floor_safety_net is False
