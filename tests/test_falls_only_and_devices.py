from __future__ import annotations

from pathlib import Path

import time
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
    # Часы сервера — в каждом ответе: клиенты считают «сколько прошло» от них
    assert abs(lite["server_ts"] - time.time()) < 5
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


# --- Режим «Только вставание» (standup) и защита /api/ignore-track --------------------


def test_standup_mode_normalization_matching_and_alert_feed() -> None:
    from app.devices import ALERT_MODE_ALL, ALERT_MODE_FALLS, ALERT_MODE_STANDUP, alert_matches_mode, normalize_alert_mode

    for syn in ("standup", "stand", "stand_only", "standup_only", "getting_up", "left_bed", "  STANDUP  "):
        assert normalize_alert_mode(syn) == ALERT_MODE_STANDUP
    for syn in ("falls", "fall", "falls_only", "fall_only"):
        assert normalize_alert_mode(syn) == ALERT_MODE_FALLS
    for syn in ("all", "", None, "unknown"):
        assert normalize_alert_mode(syn) == ALERT_MODE_ALL

    assert alert_matches_mode(PatientState.LEFT_BED, AlertSeverity.WARNING, "standup") is True
    assert alert_matches_mode(PatientState.LEFT_BED, AlertSeverity.CRITICAL, "standup") is False
    assert alert_matches_mode(PatientState.SITTING_UP, AlertSeverity.WARNING, "standup") is False
    assert alert_matches_mode(PatientState.HELP_CALL, AlertSeverity.WARNING, "standup") is False
    assert alert_matches_mode(PatientState.MISSING, AlertSeverity.WARNING, "standup") is False
    assert alert_matches_mode(PatientState.IN_BED, "info", "standup") is False
    assert alert_matches_mode(PatientState.FALLEN, AlertSeverity.CRITICAL, "standup") is True
    assert alert_matches_mode(PatientState.FALLEN, AlertSeverity.CRITICAL, "falls") is True
    assert alert_matches_mode(PatientState.LEFT_BED, AlertSeverity.WARNING, "falls") is False

    feed = AlertFeed()
    feed.push("s1", "sitting_up", "warning", "Садится", "")
    feed.push("h1", "help_call", "warning", "Зовёт рукой", "")
    feed.push("m1", "missing", "warning", "Ушла из кадра", "")
    feed.push("l_ok", "left_bed", "warning", "Встала с кровати одна", "")
    feed.push("l_crit", "left_bed", "critical", "Долго не возвращается в кровать!", "")
    feed.push("f1", "fallen", "critical", "Падение!", "")

    assert [x["id"] for x in feed.since(0, mode="all")] == ["s1", "h1", "m1", "l_ok", "l_crit", "f1"]
    assert [x["id"] for x in feed.since(0, mode="standup")] == ["l_ok", "f1"]
    assert [x["id"] for x in feed.since(0, mode="falls")] == ["f1"]


def test_standup_mode_telegram_delivery_escalation_and_cyclic_keyboard(tmp_path: Path) -> None:
    import time
    from urllib.parse import parse_qs

    from app.notifier import (
        ALERT_MODE_ALL,
        ALERT_MODE_FALLS,
        ALERT_MODE_STANDUP,
        BOT_COMMANDS_RU,
        BTN_ALL_EVENTS_TEXT,
        BTN_FALLS_ONLY_TEXT,
        BTN_STANDUP_ONLY_TEXT,
        alert_matches_mode,
        normalize_alert_mode,
    )

    assert normalize_alert_mode("stand_only") == ALERT_MODE_STANDUP
    assert alert_matches_mode("left_bed", "warning", ALERT_MODE_STANDUP) is True
    assert any(c["command"] == "standonly" for c in BOT_COMMANDS_RU)

    # Циклическая кнопка режима: all -> standup -> falls -> all
    assert BTN_STANDUP_ONLY_TEXT in _build_reply_keyboard(True, ALERT_MODE_ALL)
    assert BTN_FALLS_ONLY_TEXT in _build_reply_keyboard(True, ALERT_MODE_STANDUP)
    assert BTN_ALL_EVENTS_TEXT in _build_reply_keyboard(True, ALERT_MODE_FALLS)

    photos: list[str] = []
    messages: list[tuple[str, str]] = []
    events_log: list[tuple[str, str]] = []
    unack = {"val": False}

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        body = request.read().decode("utf-8", "ignore")
        if endpoint == "sendPhoto":
            for cid in ("1001", "2002", "3003"):
                if f'name="chat_id"\r\n\r\n{cid}' in body:
                    photos.append(cid)
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 15}})
        if endpoint == "sendMessage":
            parsed = parse_qs(body)
            cid = (parsed.get("chat_id") or [""])[0]
            txt = (parsed.get("text") or [""])[0]
            if cid:
                messages.append((cid, txt))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 16}})
        return httpx.Response(200, json={"ok": True, "result": True})

    transport = httpx.MockTransport(handler)
    notifier = TelegramNotifier(
        bot_token="123456:MOCK",
        chat_ids=("1001", "2002", "3003"),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: b"\xff\xd8\xff\xd9",
        is_fall_unacknowledged=lambda: unack["val"],
        repeat_interval_sec=10.0,
        max_repeats=1,
        subscribers_path=tmp_path / "subs.json",
        on_event=lambda lvl, txt: events_log.append((lvl, txt)),
        transport=transport,
    )

    with httpx.Client(transport=transport) as client:
        # 2002 включает режим «Только вставание» кнопкой, 3003 пробует /standonly и затем «Только падения»
        notifier._handle_update(
            client,
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 2002},
                    "from": {"id": 2002, "first_name": "Анна"},
                    "text": BTN_STANDUP_ONLY_TEXT,
                },
            },
        )
        assert notifier.get_subscriber_mode("2002") == "standup"
        assert any("только вставание" in txt for _, txt in events_log)

        notifier._handle_update(
            client,
            {
                "update_id": 2,
                "message": {
                    "chat": {"id": 3003},
                    "from": {"id": 3003, "first_name": "Иван"},
                    "text": "/standonly",
                },
            },
        )
        assert notifier.get_subscriber_mode("3003") == "standup"
        notifier._handle_update(
            client,
            {
                "update_id": 3,
                "message": {
                    "chat": {"id": 3003},
                    "from": {"id": 3003, "first_name": "Иван"},
                    "text": BTN_FALLS_ONLY_TEXT,
                },
            },
        )
        assert notifier.get_subscriber_mode("3003") == "falls"

    assert notifier.get_subscriber_mode("1001") == "all"

    # 1. Посадка -> только all (1001)
    photos.clear()
    notifier.send_alert_photo(_event(PatientState.SITTING_UP, AlertSeverity.WARNING), b"jpg", alert_id="s1")
    assert photos == ["1001"]

    # 2. Штатное вставание (left_bed + warning) -> all (1001) и standup (2002)
    photos.clear()
    notifier.send_alert_photo(_event(PatientState.LEFT_BED, AlertSeverity.WARNING), b"jpg", alert_id="l1")
    assert photos == ["1001", "2002"]

    # 3. Критическое напоминание prolonged_absence (left_bed + critical) -> ТОЛЬКО all (1001), и повторы ТОЛЬКО в all!
    photos.clear()
    messages.clear()
    unack["val"] = True
    notifier._stop_event.wait = lambda timeout=None: False  # type: ignore[method-assign]
    notifier.send_alert_photo(
        AlertEvent(PatientState.LEFT_BED, AlertSeverity.CRITICAL, "Долго не возвращается в кровать!", "Описание", 2.0, 1),
        b"jpg",
        alert_id="pa1",
    )
    assert photos == ["1001"]
    deadline = time.monotonic() + 2.0
    while notifier._repeat_active and time.monotonic() < deadline:
        time.sleep(0.01)
    escalation_cids = [cid for cid, txt in messages if "Повтор #1" in txt]
    assert escalation_cids == ["1001"]

    # 4. Реальное падение (fallen + critical) -> во ВСЕ 3 режима, и повторы во ВСЕ 3 режима!
    photos.clear()
    messages.clear()
    notifier.send_alert_photo(
        AlertEvent(PatientState.FALLEN, AlertSeverity.CRITICAL, "Падение!", "Описание", 3.0, 1),
        b"jpg",
        alert_id="fall1",
    )
    assert photos == ["1001", "2002", "3003"]
    deadline = time.monotonic() + 2.0
    while notifier._repeat_active and time.monotonic() < deadline:
        time.sleep(0.01)
    fall_escalation_cids = [cid for cid, txt in messages if "Повтор #1" in txt]
    assert fall_escalation_cids == ["1001", "2002", "3003"]
    unack["val"] = False

    # Перезагрузка сохраняет standup
    reloaded = TelegramNotifier(
        bot_token="123456:MOCK",
        chat_ids=("1001", "2002", "3003"),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: None,
        is_fall_unacknowledged=lambda: False,
        subscribers_path=tmp_path / "subs.json",
        transport=transport,
    )
    assert reloaded.get_subscriber_mode("2002") == "standup"


def test_standup_mode_http_api_poll_alarm_active_and_ignore_track_conflict(tmp_path: Path, monkeypatch) -> None:
    settings = _build_test_settings(tmp_path)
    engine = FallGuardEngine(settings)
    engine.notifier.register_subscriber("7700", name="Мария", username="maria", force_enable=True)
    client = TestClient(create_web_app(engine))

    # /api/subscribers/{id}/mode принимает standup
    sub_res = client.post("/api/subscribers/7700/mode", json={"mode": "standup"})
    assert sub_res.status_code == 200 and sub_res.json()["subscriber"]["mode"] == "standup"

    # Привязка телефона и переключение через /api/me/mode и /api/devices/{id}
    pair = client.post("/api/devices/pair-code").json()
    client.cookies.clear()
    claimed = client.post("/api/devices/claim", json={"code": pair["code"], "name": "Galaxy"}, headers=EXTERNAL).json()
    client.cookies.clear()
    token = claimed["token"]
    dev_id = claimed["device"]["id"]
    auth = {**EXTERNAL, "authorization": f"Bearer {token}"}

    me_res = client.post("/api/me/mode", json={"mode": "standup"}, headers=auth)
    assert me_res.status_code == 200 and me_res.json()["me"]["mode"] == "standup"
    assert client.post(f"/api/devices/{dev_id}", json={"mode": "falls"}).json()["device"]["mode"] == "falls"
    assert client.post(f"/api/devices/{dev_id}", json={"mode": "stand_only"}).json()["device"]["mode"] == "standup"

    # Лента /api/alerts/poll для телефона в режиме standup
    s_seq = engine.alert_feed.push("s1", "sitting_up", "warning", "Садится", "")["seq"]
    engine.alert_feed.push("l_ok", "left_bed", "warning", "Встала с кровати одна", "")
    engine.alert_feed.push("l_crit", "left_bed", "critical", "Долго не возвращается в кровать!", "")
    polled = client.get(f"/api/alerts/poll?after={s_seq - 1}&wait=0", headers=auth).json()
    assert polled["mode"] == "standup"
    assert [a["id"] for a in polled["alerts"]] == ["l_ok"]

    # Сирена alarm_active в режиме standup НЕ звенит при неподтверждённом left_bed, но звенит при fallen
    engine.fsm._state = PatientState.LEFT_BED
    engine.fsm._fall_unacknowledged = True
    poll_standup = client.get(f"/api/alerts/poll?after={polled['cursor']}&wait=0", headers=auth).json()
    assert poll_standup["fall_active"] is False and poll_standup["alarm_active"] is False
    poll_lan_all = client.get(f"/api/alerts/poll?after={polled['cursor']}&wait=0&mode=all").json()
    assert poll_lan_all["alarm_active"] is True

    engine.fsm._state = PatientState.FALLEN
    engine.fsm._fall_unacknowledged = True
    poll_fall = client.get(f"/api/alerts/poll?after={polled['cursor']}&wait=0", headers=auth).json()
    assert poll_fall["fall_active"] is True and poll_fall["alarm_active"] is True

    # /api/ignore-track переводит ValueError от движка в 409 Conflict с текстом причины
    def _deny_bed_ignore(_tid: int):
        raise ValueError("Нельзя добавить игнор-зону поверх кровати или двери — используйте кнопку «Свой»")

    monkeypatch.setattr(engine, "ignore_track", _deny_bed_ignore)
    conflict = client.post("/api/ignore-track", json={"track_id": 1})
    assert conflict.status_code == 409
    assert "поверх кровати" in conflict.json()["detail"]




def test_phone_receives_events_by_mode_matrix(tmp_path: Path) -> None:
    """Что получает телефон в каждом режиме: все / только вставание / только падения."""
    engine = FallGuardEngine(_build_test_settings(tmp_path))
    client = TestClient(create_web_app(engine))
    code = client.post("/api/devices/pair-code").json()["code"]
    client.cookies.clear()
    token = client.post("/api/devices/claim", json={"code": code, "name": "Pixel"}, headers=EXTERNAL).json()["token"]
    client.cookies.clear()
    auth = {**EXTERNAL, "authorization": f"Bearer {token}"}
    events = [("sitting_up", "warning"), ("left_bed", "warning"), ("help_call", "warning"), ("fallen", "critical")]
    expected = {
        "all": {"sitting_up", "left_bed", "help_call", "fallen"},
        "standup": {"left_bed", "fallen"},
        "falls": {"fallen"},
    }
    for mode, want in expected.items():
        assert client.post("/api/me/mode", json={"mode": mode}, headers=auth).json()["me"]["mode"] == mode
        start = engine.alert_feed.cursor
        for i, (state, sev) in enumerate(events):
            engine.alert_feed.push(f"{mode}{i}", state, sev, state, "")
        got = client.get(f"/api/alerts/poll?after={start}&wait=0", headers=auth).json()
        assert {a["state"] for a in got["alerts"]} == want, mode
        assert all(a["is_fall"] == (a["state"] == "fallen") for a in got["alerts"])

    # Непринятая тревога: в режиме «все» сирена держится для любого критического события,
    # в режимах «вставание»/«падения» — только пока это само падение
    engine.fsm._fall_unacknowledged = True
    for mode, active in (("all", True), ("standup", False), ("falls", False)):
        client.post("/api/me/mode", json={"mode": mode}, headers=auth)
        assert client.get("/api/alerts/poll?after=0&wait=0", headers=auth).json()["alarm_active"] is active, mode


def test_camera_health_masks_dynamic_zones_lamp_transition_and_exposes_status_snapshot(
    tmp_path: Path,
) -> None:
    """
    Включение потолочной люстры (верхний засвет y < 0.08 + потемневшее окно), открытие двери
    и смена одеяла/позы на кровати НЕ вызывают ложное «Камеру сдвинули» при неподвижных стенах,
    а реальный сдвиг на 8 px (в 160x90) или перекрытие объектива по-прежнему поднимают issue='moved'.
    """
    import cv2
    import numpy as np

    from app.camera_health import CameraHealthMonitor

    h, w = 1080, 1920
    bed_poly = [
        [0.59805, 0.79135],
        [0.60234, 0.65649],
        [0.66389, 0.64504],
        [0.69395, 0.56997],
        [0.66890, 0.41858],
        [0.30964, 0.40458],
        [0.25740, 0.57888],
        [0.27815, 0.78880],
    ]
    door_poly = [
        [0.20659, 0.19211],
        [0.26885, 0.82188],
        [0.21947, 0.99746],
        [0.18297, 0.99746],
        [0.06847, 0.26845],
    ]

    # Строим комнату с неподвижными деревянными планками стен и мебелью
    wall_base = np.full((h, w, 3), 95, dtype=np.uint8)
    for y_plank in range(90, h, 38):
        cv2.line(wall_base, (0, y_plank), (w - 1, y_plank), (58, 72, 88), 3)
    for x_post in (360, 760, 1180, 1520):
        cv2.line(wall_base, (x_post, 90), (x_post, h - 1), (48, 60, 74), 4)
    cv2.rectangle(wall_base, (620, 130), (960, 340), (140, 155, 168), -1)
    cv2.rectangle(wall_base, (620, 130), (960, 340), (42, 48, 55), 5)

    bed_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in bed_poly], dtype=np.int32)
    door_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in door_poly], dtype=np.int32)

    # 1. Сумеречный эталон (16:09): выключена люстра, яркое окно справа, закрытая дверь, тёмный плед на кровати
    dusk_room = (wall_base.astype(np.float32) * 0.58).astype(np.uint8)
    cv2.rectangle(dusk_room, (1580, 110), (1890, 560), (235, 240, 245), -1)  # светлое окно
    cv2.fillPoly(dusk_room, [door_pts], (125, 132, 140))  # закрытое полотно двери
    cv2.fillPoly(dusk_room, [bed_pts], (70, 82, 110))

    # 2. Вечер с люстрой (17:57): яркий засвет потолка y < 0.08, чёрное окно, распахнутая дверь и новое пёстрое одеяло
    lamp_room = np.clip(wall_base.astype(np.float32) * 1.28 + 18.0, 0, 255).astype(np.uint8)
    lamp_room[: int(0.08 * h), :] = 250  # засвет потолочного плафона
    cv2.rectangle(lamp_room, (1580, 110), (1890, 560), (18, 20, 24), -1)  # почерневшее ночное окно
    cv2.fillPoly(lamp_room, [door_pts], (22, 26, 30))  # тёмный проём открытой двери
    for stripe_x in range(140, 510, 24):
        cv2.line(lamp_room, (stripe_x, 220), (stripe_x, 980), (200, 180, 150), 4)
    cv2.fillPoly(lamp_room, [bed_pts], (190, 210, 225))
    for bx in range(540, 1280, 32):
        cv2.circle(lamp_room, (bx, 640), 14, (45, 65, 160), -1)

    rng = np.random.default_rng(17)

    def with_sensor_noise(img: np.ndarray) -> np.ndarray:
        noise = rng.normal(0.0, 2.5, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    mon = CameraHealthMonitor(tmp_path)
    mon.set_dynamic_polygons(bed_polygon=bed_poly, door_polygon=door_poly)
    assert 0.25 <= float(mon.snapshot()["static_ratio"]) < 0.80

    t = 0.0
    for _ in range(5):
        t += 20.0
        assert mon.update(with_sensor_noise(dusk_room), t, False) == []
    assert mon.snapshot()["has_reference"] == ["day"]

    # Переход «окно -> люстра + открытая дверь + смена одеяла» при неподвижных стенах НЕ поднимает тревогу!
    lamp_changes = []
    for _ in range(6):
        t += 20.0
        lamp_changes.extend(mon.update(with_sensor_noise(lamp_room), t, False))
    snap_lamp = mon.snapshot()
    assert lamp_changes == []
    assert mon.issue == ""
    assert float(snap_lamp["shift_px"]) < 1.0

    # Безопасный фолбэк на полный кадр, если статическая площадь < 25% кадра
    huge_bed = [[0.01, 0.01], [0.99, 0.01], [0.99, 0.99], [0.01, 0.99]]
    mon_fallback = CameraHealthMonitor(None)
    mon_fallback.set_dynamic_polygons(bed_polygon=huge_bed, door_polygon=[])
    assert mon_fallback.snapshot()["static_ratio"] == 1.0

    # Реальный физический сдвиг камеры на 96 px в 1080p (= 8 px в 160x90) -> issue == "moved"
    shifted_room = np.roll(lamp_room, 96, axis=1)
    shift_changes = []
    for _ in range(4):
        t += 20.0
        shift_changes.extend(mon.update(with_sensor_noise(shifted_room), t, False))
    assert ("moved", True) in [(c.kind, c.active) for c in shift_changes]
    assert mon.issue == "moved"
    assert float(mon.snapshot()["shift_px"]) >= 6.0

    # Перекрытие объектива тряпкой (однородный серый фон с шумом матрицы, corr ~ 0 < 0.16) -> issue == "moved"
    mon.reset_reference()
    for _ in range(4):
        t += 20.0
        mon.update(with_sensor_noise(lamp_room), t, False)
    assert mon.issue == ""
    cloth_frame = np.full((h, w, 3), 42, dtype=np.uint8)
    cloth_changes = []
    for _ in range(4):
        t += 20.0
        cloth_changes.extend(mon.update(with_sensor_noise(cloth_frame), t, False))
    assert ("moved", True) in [(c.kind, c.active) for c in cloth_changes]
    assert mon.issue == "moved"

    # Проверка экспозиции camera_health.snapshot() в engine.get_status() и /api/status
    engine = FallGuardEngine(_build_test_settings(tmp_path / "eng"))
    engine.zones.update(bed_polygon=bed_poly, door_polygon=door_poly)
    engine.on_zones_updated()
    st = engine.get_status()
    assert isinstance(st.get("camera_health"), dict)
    assert st["camera_health"]["issue"] == ""
    assert "shift_px" in st["camera_health"] and "correlation" in st["camera_health"]
    assert 0.25 <= float(st["camera_health"]["static_ratio"]) < 0.80





def test_device_list_flags_phones_that_went_silent(tmp_path: Path) -> None:
    """Телефон, который давно не опрашивал сервер, помечен: тревоги до него не дойдут."""
    clock = [1_000_000.0]
    store = DeviceStore(tmp_path / "devices.json", wall_clock=lambda: clock[0])
    code, _ = store.create_pair_code()
    dev, _tok = store.claim(code, name="Pixel")
    code2, _ = store.create_pair_code()
    dev2, _tok2 = store.claim(code2, name="Galaxy")
    store.update(dev2["id"], enabled=False)
    clock[0] += 20 * 60
    items = {d["name"]: d for d in store.list()}
    assert items["Pixel"]["stale"] is True and items["Pixel"]["offline_sec"] >= 1200
    assert items["Galaxy"]["stale"] is False, "выключенный получатель не считается пропавшим"
    clock[0] -= 19 * 60
    assert {d["name"]: d for d in store.list()}["Pixel"]["stale"] is False
