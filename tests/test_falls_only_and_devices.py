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


# --- Пауза «Я рядом» не отключает контроль падения --------------------------------

def test_pause_still_catches_a_new_fall_when_nobody_is_around() -> None:
    fsm = FallStateMachine(fall_confirm_seconds=1.0)
    roles = {1: _ident(1, Role.PATIENT)}
    t = 0.0
    for _ in range(8):
        t += 0.25
        fsm.step([_in_bed()], roles, False, BED_ZONE, now=t)
    fsm.acknowledge_and_pause(now=t, pause_seconds=900.0)

    # Пока лежит в кровати — пауза, тишина
    for _ in range(20):
        t += 0.25
        assert fsm.step([_in_bed()], roles, False, BED_ZONE, now=t, caregiver_visible=False) is None
    assert fsm.state == PatientState.PAUSED

    # Упала, своих в кадре нет — тревога, пауза снята
    events = []
    for _ in range(16):
        t += 0.25
        ev = fsm.step([_on_floor(1)], roles, False, BED_ZONE, now=t, caregiver_visible=False)
        if ev is not None:
            events.append(ev)
    assert len(events) == 1 and events[0].state == PatientState.FALLEN
    assert fsm.pause_remaining_sec(t) == 0.0 and fsm.fall_unacknowledged is True


def test_pause_does_not_repeat_the_fall_that_was_just_acknowledged_or_when_family_is_there() -> None:
    fsm = FallStateMachine(fall_confirm_seconds=1.0)
    roles = {1: _ident(1, Role.PATIENT)}
    t = 0.0
    for _ in range(12):
        t += 0.25
        fsm.step([_on_floor(1)], roles, False, BED_ZONE, now=t, caregiver_visible=False)
    assert fsm.state == PatientState.FALLEN
    fsm.acknowledge_and_pause(now=t, pause_seconds=900.0)  # «Я иду» — человек ещё на полу
    for _ in range(80):
        t += 0.25
        assert fsm.step([_on_floor(1)], roles, False, BED_ZONE, now=t, caregiver_visible=False) is None

    # Новый эпизод при родственнике рядом — тоже тишина
    for _ in range(30):
        t += 0.25
        fsm.step([_in_bed()], roles, False, BED_ZONE, now=t, caregiver_visible=False)
    for _ in range(30):
        t += 0.25
        assert fsm.step([_on_floor(1)], roles, True, BED_ZONE, now=t, caregiver_visible=True) is None
    assert fsm.state == PatientState.PAUSED


# --- Внешний доступ: прокси без Host не открывает пульт без входа -------------------

def test_proxied_requests_from_internet_require_login_even_without_public_host() -> None:
    from app.tg_auth import is_external_proxy_request as ext

    docker = "172.18.0.5"
    assert ext({"host": "fall-guard:8090", "x-forwarded-for": "8.8.8.8"}, docker) is True
    # Клиент подставил «домашний» адрес — ближайший прокси всё равно дописал настоящий
    assert ext({"host": "fall-guard:8090", "x-forwarded-for": "192.168.1.30, 8.8.8.8"}, docker) is True
    assert ext({"host": "fall-guard:8090", "x-forwarded-proto": "https"}, docker) is True
    assert ext({"host": "fall-guard:8090", "x-forwarded-for": "192.168.1.30"}, docker) is False
    assert ext({"host": "192.168.1.50:8090"}, "192.168.1.30") is False


# --- Контроль камеры: застывшее изображение и сдвиг -----------------------------------

def _room(seed: int = 0):
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), 70, np.uint8)
    for _ in range(40):  # «мебель»: прямоугольники разной яркости
        x, y = int(rng.integers(0, 560)), int(rng.integers(0, 300))
        w, h = int(rng.integers(20, 120)), int(rng.integers(20, 90))
        c = int(rng.integers(20, 230))
        cv2.rectangle(img, (x, y), (x + w, y + h), (c, c, c), -1)
    return img


def test_camera_health_detects_frozen_picture_and_moved_camera(tmp_path: Path) -> None:
    import numpy as np

    from app.camera_health import CameraHealthMonitor

    rng = np.random.default_rng(3)
    base = _room()

    def noisy(img):
        return np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

    mon = CameraHealthMonitor(tmp_path)
    t, changes = 0.0, []
    for _ in range(200):
        t += 1.0
        changes += mon.update(noisy(base), t, False)
    assert changes == [] and mon.issue == ""
    assert (tmp_path / "camera_ref_day.npy").exists()

    # Камеру задели: картинка уехала на 12% кадра
    moved = np.roll(base, 80, axis=1)
    for _ in range(120):
        t += 1.0
        changes += mon.update(noisy(moved), t, False)
    assert [(c.kind, c.active) for c in changes] == [("moved", True)] and mon.issue == "moved"

    # Перерисовали разметку — новый вид становится эталоном, тревога снимается
    mon.reset_reference()
    assert mon.issue == ""

    # Застывшее изображение: поток идёт, а кадр не меняется
    frozen = CameraHealthMonitor(None)
    t, changes = 0.0, []
    for _ in range(300):
        t += 0.5
        changes += frozen.update(base, t, False)
    assert ("frozen", True) in [(c.kind, c.active) for c in changes]
    changes = frozen.update(noisy(base), t + 0.5, False)
    assert [(c.kind, c.active) for c in changes] == [("frozen", False)]


# --- Тревога снимается сама, когда подходит свой --------------------------------------

def test_alarm_auto_resolves_when_family_member_arrives_and_telegram_post_is_marked(tmp_path: Path) -> None:
    edits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        body = request.read().decode("utf-8", "ignore")
        if endpoint == "sendPhoto":
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 11}})
        if endpoint == "editMessageCaption":
            from urllib.parse import parse_qs

            edits.append(parse_qs(body).get("caption", [""])[0])
        return httpx.Response(200, json={"ok": True, "result": True})

    notifier = TelegramNotifier(
        bot_token="123456:MOCK",
        chat_ids=("1001",),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: None,
        is_fall_unacknowledged=lambda: False,
        subscribers_path=tmp_path / "subs.json",
        transport=httpx.MockTransport(handler),
    )
    notifier.send_alert_photo(_event(PatientState.FALLEN, AlertSeverity.CRITICAL), b"jpg", alert_id="f1")
    assert notifier.mark_latest_resolved("Рядом свой — тревога снята") is True
    assert edits and "Рядом свой" in edits[-1]
    assert notifier.mark_latest_resolved("ещё раз") is False  # одно сообщение помечается один раз

    # Машина состояний: свой подошёл к упавшей — тревога больше не «неподтверждённая»
    fsm = FallStateMachine(fall_confirm_seconds=1.0)
    roles = {1: _ident(1, Role.PATIENT), 2: _ident(2, Role.VISITOR)}
    t = 0.0
    for _ in range(12):
        t += 0.25
        fsm.step([_on_floor(1)], {1: roles[1]}, False, BED_ZONE, now=t, caregiver_visible=False)
    assert fsm.fall_unacknowledged is True
    t += 0.25
    fsm.step([_on_floor(1), _standing(2)], roles, True, BED_ZONE, now=t, caregiver_visible=True)
    assert fsm.fall_unacknowledged is False and fsm.state == PatientState.CAREGIVER_ASSIST


def test_fonts_are_public_but_everything_else_stays_behind_login(tmp_path: Path) -> None:
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    client = TestClient(create_web_app(engine))
    font = client.get("/static/fonts/onest-cyrillic.woff2", headers=EXTERNAL)
    assert font.status_code == 200 and font.headers["content-type"].startswith("font/")
    assert client.get("/static/index.html", headers=EXTERNAL).status_code == 401
    assert client.get("/api/status", headers=EXTERNAL).status_code == 401


def test_phone_alarm_screen_media_reachable_with_device_token(tmp_path: Path) -> None:
    """Экран тревоги в приложении грузит живые кадры и снимок тревоги по токену телефона снаружи."""
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    client = TestClient(create_web_app(engine))
    code = client.post("/api/devices/pair-code").json()["code"]
    client.cookies.clear()
    token = client.post("/api/devices/claim", json={"code": code, "name": "Pixel"}, headers=EXTERNAL).json()["token"]
    client.cookies.clear()
    auth = {**EXTERNAL, "authorization": f"Bearer {token}"}

    assert client.get("/api/snapshot.jpg?overlay=1", headers=EXTERNAL).status_code == 401
    assert client.get("/api/snapshot.jpg?overlay=1", headers=auth).status_code in (200, 503)
    assert client.get("/api/clips/nope.jpg", headers=EXTERNAL).status_code == 401
    assert client.get("/api/clips/nope.jpg", headers=auth).status_code in (200, 404)


def test_status_is_gzipped_and_lite_drops_heavy_parts_but_streams_are_not(tmp_path: Path) -> None:
    """Пульт опрашивает статус постоянно: ответ сжат, «лёгкий» опрос без превью и журнала."""
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    for i in range(80):
        engine._log_event("warning", f"Пытается встать с кровати — событие {i}", state="sitting_up")
    client = TestClient(create_web_app(engine))
    full = client.get("/api/status", headers={"accept-encoding": "gzip"})
    assert full.status_code == 200 and full.headers.get("content-encoding") == "gzip"
    data = full.json()
    assert "events" in data and "thumbs" in data and "recent_clips" in data
    lite = client.get("/api/status?lite=1", headers={"accept-encoding": "gzip"}).json()
    assert lite["lite"] is True and "events" not in lite and "thumbs" not in lite
    assert "recent_clips" in lite and "state" in lite and "timeline" in lite
    plain = client.get("/api/status", headers={"accept-encoding": "identity"})
    assert plain.headers.get("content-encoding") is None and plain.json()["state"] == data["state"]
    snap = client.get("/api/snapshot.jpg", headers={"accept-encoding": "gzip"})
    assert snap.headers.get("content-encoding") is None


def test_camera_health_telegram_push_respects_notify_camera_offline(tmp_path: Path, monkeypatch) -> None:
    """Проблемы камеры всегда пишутся в журнал пульта, а пуши в Telegram идут только при NOTIFY_CAMERA_OFFLINE=true."""
    import dataclasses
    import numpy as np
    from app.camera_health import CameraIssueChange

    settings = _build_test_settings(tmp_path)
    assert settings.notify_camera_offline is False
    engine = FallGuardEngine(settings)

    sent: list[str] = []
    monkeypatch.setattr(engine.notifier, "send_text", lambda text: sent.append(text) or True)
    monkeypatch.setattr(
        engine.camera_health,
        "update",
        lambda frame, now, night_ir: [CameraIssueChange(kind="frozen", active=True, detail="95 с")],
    )

    dummy_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    engine._handle_camera_health(dummy_frame, False)
    assert sent == []
    assert any("застыло" in ev["text"] for ev in engine.get_status()["events"])

    engine.settings = dataclasses.replace(settings, notify_camera_offline=True)
    engine._handle_camera_health(dummy_frame, False)
    assert len(sent) == 1 and "застыло" in sent[0]

