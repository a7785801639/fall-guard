from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import cv2
import httpx
import numpy as np
import pytest

from app.capture import (
    BACKEND_CPU,
    BACKEND_NVDEC,
    VideoStreamCapture,
    _build_cuvid_ffmpeg_cmd,
    _redact_url,
)
from app.notifier import (
    BOT_COMMANDS_RU,
    BTN_MUTE_TEXT,
    BTN_PHOTO_TEXT,
    BTN_STATUS_TEXT,
    BTN_UNMUTE_TEXT,
    TelegramNotifier,
    _build_alert_inline_keyboard,
    _build_reply_keyboard,
)
from app.state_machine import AlertEvent, AlertSeverity, PatientState
from app.thumbs import TimelineThumbnailStore


def _extract_multipart_field(raw_body: bytes, field_name: str) -> str:
    """Извлекает строковое значение поля из multipart/form-data или x-www-form-urlencoded тела."""
    text = raw_body.decode("utf-8", errors="ignore")
    if f'name="{field_name}"' in text:
        marker = f'name="{field_name}"'
        after = text.split(marker, 1)[1]
        parts = after.split("\r\n\r\n", 1)
        if len(parts) == 2:
            val = parts[1].split("\r\n--", 1)[0]
            return val.strip()
    parsed = parse_qs(text)
    if field_name in parsed and parsed[field_name]:
        return parsed[field_name][0]
    return ""


def _make_room_frame(width: int = 640, height: int = 360, seed_val: int = 40) -> np.ndarray:
    """Генерирует детализированный тестовый BGR-кадр комнаты без однородных битых полос."""
    yy, xx = np.meshgrid(
        np.linspace(20, 180, height, dtype=np.int32),
        np.linspace(30, 210, width, dtype=np.int32),
        indexing="ij",
    )
    b = ((yy + xx + seed_val) % 255).astype(np.uint8)
    g = ((yy * 2 + xx // 2 + seed_val) % 255).astype(np.uint8)
    r = ((xx * 2 + yy // 2 + seed_val) % 255).astype(np.uint8)
    return np.dstack([b, g, r])


def _bgr_to_nv12_bytes(frame_bgr: np.ndarray) -> bytes:
    """Кодирует BGR-кадр чётного размера в сырой буфер NV12 (Y + interleaved UV)."""
    h, w = frame_bgr.shape[:2]
    assert h % 2 == 0 and w % 2 == 0
    yuv_i420 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
    y_plane = yuv_i420[:h, :]
    u_plane = yuv_i420[h : h + h // 4, :].reshape((h // 2, w // 2))
    v_plane = yuv_i420[h + h // 4 :, :].reshape((h // 2, w // 2))
    uv_interleaved = np.empty((h // 2, w), dtype=np.uint8)
    uv_interleaved[:, 0::2] = u_plane
    uv_interleaved[:, 1::2] = v_plane
    nv12 = np.vstack([y_plane, uv_interleaved])
    return nv12.tobytes()


# ==============================================================================
# 1. Тесты TimelineThumbnailStore (app/thumbs.py)
# ==============================================================================


def test_thumbs_periodic_60s_state_change_and_debounce(tmp_path: Path) -> None:
    thumbs_dir = tmp_path / "thumbs"
    store = TimelineThumbnailStore(thumbs_dir)
    frame = _make_room_frame(1280, 720, seed_val=15)

    t0 = 1_765_000_000.0
    # 1. Первый кадр сразу после старта (_last_periodic_ts == 0)
    first = store.maybe_record(frame, "in_bed", now_ts=t0)
    assert first is not None
    assert first["state"] == "in_bed"
    assert first["ts"] == t0

    jpg_path = store.get_by_id(first["id"])
    assert jpg_path is not None and jpg_path.is_file()
    decoded = cv2.imdecode(np.frombuffer(jpg_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (180, 320, 3)

    # 2. До истечения 60 с без смены state кадры пропускаются
    assert store.maybe_record(frame, "in_bed", now_ts=t0 + 12.0) is None
    assert store.maybe_record(frame, "in_bed", now_ts=t0 + 59.5) is None

    # 3. Ровно через 60 с пишется плановый поминутный снимок
    second = store.maybe_record(frame, "in_bed", now_ts=t0 + 60.0)
    assert second is not None
    assert second["ts"] == t0 + 60.0

    # 4. Антидребезг min_gap_sec = 5.0 при смене состояния
    assert store.maybe_record(frame, "sitting_up", now_ts=t0 + 62.0) is None
    # Через 6 с после предыдущего кадра смена состояния фиксируется внеочередно
    third = store.maybe_record(frame, "sitting_up", now_ts=t0 + 66.0)
    assert third is not None
    assert third["state"] == "sitting_up"

    # Проверка фильтрации list_index(since_ts) и перезагрузки с диска
    reloaded = TimelineThumbnailStore(thumbs_dir)
    all_items = reloaded.list_index()
    assert [it["id"] for it in all_items] == [first["id"], second["id"], third["id"]]
    recent_items = reloaded.list_index(since_ts=t0 + 60.0)
    assert [it["id"] for it in recent_items] == [second["id"], third["id"]]


def test_thumbs_retention_bisect_nearest_and_path_traversal(tmp_path: Path) -> None:
    thumbs_dir = tmp_path / "thumbs"
    store = TimelineThumbnailStore(thumbs_dir, max_thumbs=5, retention_sec=86400.0)
    frame = _make_room_frame(640, 360, seed_val=77)

    base_ts = 1_765_100_000.0
    very_old = store.maybe_record(frame, "in_bed", now_ts=base_ts)
    assert very_old is not None

    # Перешагиваем 24 часа -> старый снимок обязан удалиться и с диска, и из индекса
    after_24h = base_ts + 86410.0
    created_ids: list[str] = []
    for i in range(6):
        rec = store.maybe_record(frame, "in_bed", now_ts=after_24h + i * 65.0)
        assert rec is not None
        created_ids.append(rec["id"])

    assert store.get_by_id(very_old["id"]) is None
    assert not (thumbs_dir / f"{very_old['id']}.jpg").exists()

    # Из 6 новых при max_thumbs=5 остаётся ровно 5 последних
    kept = store.list_index()
    assert len(kept) == 5
    assert [it["id"] for it in kept] == created_ids[1:]
    assert not (thumbs_dir / f"{created_ids[0]}.jpg").exists()

    # Проверка бинарного поиска get_nearest (bisect)
    target_ts = float(kept[2]["ts"]) + 14.0
    nearest = store.get_nearest(target_ts, max_delta_sec=900.0)
    assert nearest is not None
    nearest_path, nearest_meta = nearest
    assert nearest_meta["id"] == kept[2]["id"]
    assert nearest_path.is_file()

    # За пределами max_delta_sec возвращается None
    assert store.get_nearest(after_24h + 50_000.0, max_delta_sec=900.0) is None

    # Защита от path traversal в get_by_id
    (tmp_path / "secret.jpg").write_bytes(b"SECRET")
    for bad_id in ("", "../secret", "..\\secret", "20261112_120000_000_idle_ffff", "a/b"):
        assert store.get_by_id(bad_id) is None


# ==============================================================================
# 2. Тесты персональной подписки и ReplyKeyboardMarkup в TelegramNotifier
# ==============================================================================


def test_notifier_backward_compat_and_env_seeding(tmp_path: Path) -> None:
    subs_path = tmp_path / "subscribers.json"
    # Старый формат subscribers.json без ключа "enabled"
    legacy_payload = {
        "subscribers": [
            {
                "chat_id": "2002",
                "name": "Мария",
                "username": "maria_rel",
                "joined_at": "2026-11-11 20:00:00",
            }
        ]
    }
    subs_path.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")

    notifier = TelegramNotifier(
        bot_token="123456:MOCK_TOKEN_NO_NET",
        chat_ids=("1001",),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: b"\xff\xd8\xff\xd9",
        is_fall_unacknowledged=lambda: False,
        subscribers_path=subs_path,
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True})),
    )

    subs = notifier.list_subscribers()
    by_id = {str(s["chat_id"]): s for s in subs}
    assert set(by_id.keys()) == {"1001", "2002"}
    assert by_id["1001"]["enabled"] is True
    assert by_id["1001"]["source"] == "env"
    assert by_id["2002"]["enabled"] is True
    assert by_id["2002"]["source"] == "bot"
    assert notifier.get_recipient_chat_ids() == ["1001", "2002"]

    # Проверяем, что чат из .env тоже может отключить и включить себе уведомления
    toggled = notifier.set_subscriber_enabled("1001", False)
    assert toggled is not None and toggled["enabled"] is False
    assert notifier.get_recipient_chat_ids() == ["2002"]


def test_notifier_multi_user_mute_unmute_and_persistent_reply_keyboard(tmp_path: Path) -> None:
    sent_requests: list[tuple[str, bytes]] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        body = request.read()
        sent_requests.append((endpoint, body))
        if endpoint == "sendPhoto":
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 101}})
        return httpx.Response(200, json={"ok": True, "result": True})

    transport = httpx.MockTransport(mock_handler)
    subs_path = tmp_path / "subscribers.json"

    notifier = TelegramNotifier(
        bot_token="123456:MOCK_TOKEN_MULTI_USER",
        chat_ids=("1001",),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: b"\xff\xd8\xff\xe0MOCK_JPEG",
        is_fall_unacknowledged=lambda: False,
        subscribers_path=subs_path,
        get_status_text=lambda: "Статус: В кровати · Камера: онлайн",
        get_report_text=lambda: "Сводка ухода FallGuard за 24 часа\n• Падений: 0",
        transport=transport,
    )

    # Проверяем наличие /mute и /unmute в BOT_COMMANDS_RU
    cmds = {c["command"]: c["description"] for c in BOT_COMMANDS_RU}
    assert "mute" in cmds and "unmute" in cmds

    # Проверяем структуру ReplyKeyboardMarkup для обоих состояний
    kb_on = json.loads(_build_reply_keyboard(enabled=True))
    assert kb_on["resize_keyboard"] is True
    assert kb_on["is_persistent"] is True
    assert kb_on["keyboard"] == [
        [{"text": BTN_MUTE_TEXT}],
        [{"text": BTN_PHOTO_TEXT}, {"text": BTN_STATUS_TEXT}],
    ]
    kb_off = json.loads(_build_reply_keyboard(enabled=False))
    assert kb_off["keyboard"][0] == [{"text": BTN_UNMUTE_TEXT}]

    with httpx.Client(transport=transport) as client:
        # Подключаются Анна (2002) и Борис (3003) через /start
        for cid, fname in ((2002, "Анна"), (3003, "Борис")):
            notifier._handle_update(
                client,
                {
                    "update_id": cid,
                    "message": {
                        "chat": {"id": cid, "first_name": fname},
                        "from": {"id": cid, "first_name": fname},
                        "text": "/start",
                    },
                },
            )
        assert notifier.get_recipient_chat_ids() == ["1001", "2002", "3003"]

        # Анна (2002) нажимает постоянную кнопку «🔕 Отключить уведомления»
        sent_requests.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 10,
                "message": {
                    "chat": {"id": 2002, "first_name": "Анна"},
                    "from": {"id": 2002, "first_name": "Анна"},
                    "text": BTN_MUTE_TEXT,
                },
            },
        )
        assert len(sent_requests) == 1
        assert sent_requests[0][0] == "sendMessage"
        mute_kb = json.loads(_extract_multipart_field(sent_requests[0][1], "reply_markup"))
        assert mute_kb["keyboard"][0][0]["text"] == BTN_UNMUTE_TEXT
        assert notifier.is_subscriber_enabled("2002") is False
        assert notifier.get_recipient_chat_ids() == ["1001", "3003"]

        # Боевая тревога уходит ТОЛЬКО 1001 и 3003 (Анне 2002 НЕ отправляется!) + с inline_keyboard
        sent_requests.clear()
        alert_ev = AlertEvent(
            state=PatientState.SITTING_UP,
            severity=AlertSeverity.WARNING,
            title="Бабушка садится",
            description="Подъём корпуса в кровати.",
            timestamp=500.0,
            patient_track_id=1,
        )
        msg_ids = notifier.send_alert_photo(alert_ev, b"\xff\xd8ALERT_JPG")
        assert set(msg_ids.keys()) == {"1001", "3003"}
        assert len(sent_requests) == 2
        for ep, raw_body in sent_requests:
            assert ep == "sendPhoto"
            alert_markup = json.loads(_extract_multipart_field(raw_body, "reply_markup"))
            assert "inline_keyboard" in alert_markup

        # Пока у Анны уведомления выключены, она жмёт «📸 Снимок» и «📊 Статус» —
        # ответы приходят с её ReplyKeyboardMarkup(enabled=False), а флаг enabled остаётся False!
        sent_requests.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 11,
                "message": {"chat": {"id": 2002}, "from": {"id": 2002}, "text": BTN_PHOTO_TEXT},
            },
        )
        assert sent_requests[0][0] == "sendPhoto"
        snap_kb = json.loads(_extract_multipart_field(sent_requests[0][1], "reply_markup"))
        assert snap_kb["keyboard"][0][0]["text"] == BTN_UNMUTE_TEXT
        assert notifier.is_subscriber_enabled("2002") is False

        sent_requests.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 12,
                "message": {"chat": {"id": 2002}, "from": {"id": 2002}, "text": BTN_STATUS_TEXT},
            },
        )
        assert sent_requests[0][0] == "sendMessage"
        st_kb = json.loads(_extract_multipart_field(sent_requests[0][1], "reply_markup"))
        assert st_kb["keyboard"][0][0]["text"] == BTN_UNMUTE_TEXT
        assert notifier.is_subscriber_enabled("2002") is False

        # Анна снова включает уведомления кнопкой «🔔 Включить уведомления»
        sent_requests.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 13,
                "message": {"chat": {"id": 2002}, "from": {"id": 2002}, "text": BTN_UNMUTE_TEXT},
            },
        )
        assert notifier.is_subscriber_enabled("2002") is True
        assert notifier.get_recipient_chat_ids() == ["1001", "2002", "3003"]
        unmute_kb = json.loads(_extract_multipart_field(sent_requests[0][1], "reply_markup"))
        assert unmute_kb["keyboard"][0][0]["text"] == BTN_MUTE_TEXT


# ==============================================================================
# 3. Тесты аппаратного декодирования NVDEC (h264_cuvid) и CPU-отката (app/capture.py)
# ==============================================================================


class _ChunkedPipeReader:
    """Эмулирует небуферизованный stdout пайп FFmpeg, отдающий данные неполными кусками в readinto."""

    def __init__(self, payload: bytes, chunk_size: int = 1024, hold_open_event: Any = None) -> None:
        self._bio = io.BytesIO(payload)
        self._chunk_size = chunk_size
        self._hold_open_event = hold_open_event
        self.closed = False

    def readinto(self, b: Any) -> int:
        if self.closed:
            return 0
        mv = memoryview(b)
        to_read = min(len(mv), self._chunk_size)
        chunk = self._bio.read(to_read)
        if chunk:
            mv[: len(chunk)] = chunk
            return len(chunk)
        if self._hold_open_event is not None:
            self._hold_open_event.wait(timeout=2.0)
        return 0

    def close(self) -> None:
        self.closed = True
        if self._hold_open_event is not None:
            self._hold_open_event.set()


class _FakePopen:
    def __init__(self, stdout_bytes: bytes, stderr_bytes: bytes = b"", hold_after_eof: bool = False) -> None:
        import threading

        self._release_ev = threading.Event() if hold_after_eof else None
        self.stdout = _ChunkedPipeReader(stdout_bytes, chunk_size=900, hold_open_event=self._release_ev)
        self.stderr = io.BytesIO(stderr_bytes)
        self.killed = False
        self.waited = False
        self._returncode: int | None = None if hold_after_eof else 0

    def poll(self) -> int | None:
        return self._returncode

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9
        self.stdout.close()

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self._returncode is None:
            self._returncode = 0
        return self._returncode


def test_capture_cuvid_nv12_pipeline_and_guaranteed_kill_on_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    # Проверяем команду и маскирование учётных данных в URL
    sample_url = "rtsp://admin:SuperSecret123@192.168.1.50:554/Streaming/Channels/101"
    assert "SuperSecret123" not in _redact_url(sample_url)
    cmd = _build_cuvid_ffmpeg_cmd("/usr/bin/ffmpeg", sample_url)
    assert "-hwaccel" in cmd and "cuda" in cmd
    assert "-c:v" in cmd and "h264_cuvid" in cmd
    assert "-timeout" in cmd and "5000000" in cmd
    assert "-pix_fmt" in cmd and "nv12" in cmd

    w, h = 64, 48
    src_frame = _make_room_frame(w, h, seed_val=90)
    nv12_packet = _bgr_to_nv12_bytes(src_frame)
    # 4 кадра подряд в потоке
    stream_bytes = nv12_packet * 4
    spawned_procs: list[_FakePopen] = []

    def fake_popen(args: list[str], **kwargs: Any) -> _FakePopen:
        proc = _FakePopen(
            stdout_bytes=stream_bytes,
            stderr_bytes=b"[rtsp] test warning rtsp://admin:Pass@192.168.1.50/stream\n",
            hold_after_eof=True,
        )
        spawned_procs.append(proc)
        return proc

    cap = VideoStreamCapture(
        rtsp_url="rtsp://user:testpass@127.0.0.1:8554/live",
        capture_fps=10.0,
        hwaccel="nvdec",
    )
    monkeypatch.setattr(cap, "_find_ffmpeg_bin", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cap, "_has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cap, "_probe_rtsp_dimensions", lambda url, ff: (w, h))
    monkeypatch.setattr("app.capture.subprocess.Popen", fake_popen)

    cap.start()
    try:
        deadline = time.monotonic() + 2.5
        got_frame: np.ndarray | None = None
        while time.monotonic() < deadline:
            frm, ts = cap.get_latest()
            if frm is not None and ts > 0:
                got_frame = frm
                break
            time.sleep(0.03)

        assert got_frame is not None
        assert got_frame.shape == (h, w, 3)
        assert cap.is_connected is True
        assert cap.decoder_backend == BACKEND_NVDEC
        assert len(spawned_procs) == 1
    finally:
        cap.stop()

    assert spawned_procs[0].killed is True
    assert spawned_procs[0].waited is True
    assert cap.decoder_backend == BACKEND_CPU


def test_capture_cuvid_falls_back_to_cpu_after_two_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    w, h = 64, 48
    fallback_frame = _make_room_frame(w, h, seed_val=120)
    cuvid_attempts: list[_FakePopen] = []

    def failing_popen(args: list[str], **kwargs: Any) -> _FakePopen:
        # FFmpeg CUVID мгновенно падает без единого кадра
        proc = _FakePopen(stdout_bytes=b"", stderr_bytes=b"Cannot load nvcuvid.dll\n", hold_after_eof=False)
        cuvid_attempts.append(proc)
        return proc

    class _FakeCvCapture:
        def __init__(self) -> None:
            self._opened = True

        def isOpened(self) -> bool:
            return self._opened

        def read(self) -> tuple[bool, np.ndarray]:
            return True, fallback_frame.copy()

        def grab(self) -> bool:
            time.sleep(0.02)
            return True

        def retrieve(self) -> tuple[bool, np.ndarray]:
            return True, fallback_frame.copy()

        def release(self) -> None:
            self._opened = False

    cap = VideoStreamCapture(
        rtsp_url="rtsp://user:testpass@127.0.0.1:8554/live",
        capture_fps=8.0,
        hwaccel="auto",
    )
    monkeypatch.setattr(cap, "_find_ffmpeg_bin", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cap, "_has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cap, "_probe_rtsp_dimensions", lambda url, ff: (w, h))
    monkeypatch.setattr("app.capture.subprocess.Popen", failing_popen)
    monkeypatch.setattr(cap, "_open_with_transport", lambda url, tr: _FakeCvCapture())

    cap.start()
    try:
        deadline = time.monotonic() + 3.0
        got_frame: np.ndarray | None = None
        while time.monotonic() < deadline:
            frm, ts = cap.get_latest()
            if frm is not None and ts > 0:
                got_frame = frm
                break
            time.sleep(0.04)

        assert got_frame is not None
        assert got_frame.shape == (h, w, 3)
        # Ровно 2 неудачные попытки CUVID -> автоматический откат на штатный CPU декодер
        assert len(cuvid_attempts) == 2
        assert cap.decoder_backend == BACKEND_CPU
        assert cap.is_connected is True
    finally:
        cap.stop()


# ==============================================================================
# 4. Волна 8: 1 пост фото->видео (editMessageMedia) + 2 кнопки + кэш file_id
# ==============================================================================


def test_wave8_alert_2buttons_edit_message_media_file_id_cache_and_early_button_race(
    tmp_path: Path,
) -> None:
    # 1. Проверка структуры компактной 1-рядной клавиатуры из 2 кнопок и лимита <= 64 байт UTF-8
    long_id = "20261113_004500_very_long_custom_alert_identifier_exceeding_sixty_four_bytes_total_abcd1234"
    kb_long = json.loads(_build_alert_inline_keyboard(long_id))
    assert len(kb_long["inline_keyboard"]) == 1
    assert len(kb_long["inline_keyboard"][0]) == 2
    for btn in kb_long["inline_keyboard"][0]:
        assert len(btn["callback_data"].encode("utf-8")) <= 64

    api_calls: list[tuple[str, bytes]] = []
    next_msg_seq = {"val": 500}

    def mock_tg_api(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        body = request.read()
        api_calls.append((endpoint, body))
        if endpoint == "sendPhoto":
            next_msg_seq["val"] += 1
            return httpx.Response(200, json={"ok": True, "result": {"message_id": next_msg_seq["val"]}})
        if endpoint == "editMessageMedia":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "result": {
                        "message_id": 501,
                        "video": {"file_id": "BAACAgIAAx0CTEST_CACHED_FILE_ID_V8"},
                    },
                },
            )
        return httpx.Response(200, json={"ok": True, "result": True})

    transport = httpx.MockTransport(mock_tg_api)
    paused_calls: list[float] = []
    false_alert_calls: list[tuple[str, str, str]] = []

    notifier = TelegramNotifier(
        bot_token="123456:WAVE8_MOCK_TOKEN",
        chat_ids=("1001", "2002"),
        on_pause_cmd=lambda sec: paused_calls.append(sec),
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: b"\xff\xd8SNAP",
        is_fall_unacknowledged=lambda: False,
        subscribers_path=tmp_path / "subscribers.json",
        on_false_alert_cmd=lambda aid, cid, who: false_alert_calls.append((aid, cid, who)),
        transport=transport,
    )

    alert_id = "20261113_004500_left_bed_a1b2"
    ev = AlertEvent(
        state=PatientState.LEFT_BED,
        severity=AlertSeverity.WARNING,
        title="Встала с кровати одна",
        description="Самостоятельный выход без сопровождения.",
        timestamp=777.0,
        patient_track_id=5,
    )

    # 2. Первичная отправка фото тревоги 2 подписчикам
    msg_ids = notifier.send_alert_photo(ev, b"\xff\xd8ALERT_PHOTO", alert_id=alert_id)
    assert msg_ids == {"1001": 501, "2002": 502}
    assert [ep for ep, _ in api_calls] == ["sendPhoto", "sendPhoto"]
    init_cap = _extract_multipart_field(api_calls[0][1], "caption")
    assert "⏳ Видеоклип загружается…" in init_cap
    init_kb = json.loads(_extract_multipart_field(api_calls[0][1], "reply_markup"))
    assert init_kb["inline_keyboard"] == [
        [
            {"text": "✅ Я рядом", "callback_data": f"near:{alert_id}"},
            {"text": "❌ Ты ошибся", "callback_data": f"wrong:{alert_id}"},
        ]
    ]

    # 3. Гонка ДО прихода MP4: на 1-й секунде Анна (1001) жмёт «✅ Я рядом», а Борис (2002) — «❌ Ты ошибся»
    with httpx.Client(transport=transport) as client:
        api_calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 101,
                "callback_query": {
                    "id": "cb_near_1",
                    "data": f"near:{alert_id}",
                    "from": {"id": 1001, "first_name": "Анна"},
                    "message": {"message_id": 501, "chat": {"id": 1001}},
                },
            },
        )
        assert paused_calls == [900.0]
        # Никаких новых sendMessage — только editMessageCaption у обоих подписчиков + answerCallbackQuery!
        assert [ep for ep, _ in api_calls] == [
            "editMessageCaption",
            "editMessageCaption",
            "answerCallbackQuery",
        ]
        near_ans = _extract_multipart_field(api_calls[2][1], "text")
        assert near_ans == "Принято — охрана на паузе 15 мин"

        api_calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 102,
                "callback_query": {
                    "id": "cb_wrong_1",
                    "data": f"wrong:{alert_id}",
                    "from": {"id": 2002, "first_name": "Борис"},
                    "message": {"message_id": 502, "chat": {"id": 2002}},
                },
            },
        )
        assert false_alert_calls == [(alert_id, "2002", "Борис")]
        assert [ep for ep, _ in api_calls] == [
            "editMessageCaption",
            "editMessageCaption",
            "answerCallbackQuery",
        ]
        wrong_ans = _extract_multipart_field(api_calls[2][1], "text")
        assert wrong_ans == "Ошибка сохранена для разбора, тревога снята"

        # Повторный клик по «✓ Ошибка записана» (noop)
        api_calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 103,
                "callback_query": {
                    "id": "cb_noop_1",
                    "data": "noop",
                    "from": {"id": 2002, "first_name": "Борис"},
                    "message": {"message_id": 502, "chat": {"id": 2002}},
                },
            },
        )
        assert [ep for ep, _ in api_calls] == ["answerCallbackQuery"]
        assert _extract_multipart_field(api_calls[0][1], "text") == "Уже отмечено"

    # 4. Через 4 с готов MP4: фото заменяется на видео через editMessageMedia!
    api_calls.clear()
    mp4_payload = b"\x00\x00\x00\x18ftypmp42WAVE8_VIDEO_BINARY_STREAM"
    notifier.send_followup_video(mp4_payload, msg_ids, ev.title, alert_id=alert_id)

    assert [ep for ep, _ in api_calls] == ["editMessageMedia", "editMessageMedia"]

    # Первый подписчик (1001) получил multipart-заливку видео attach://video
    first_body = api_calls[0][1]
    assert mp4_payload in first_body
    first_media = json.loads(_extract_multipart_field(first_body, "media"))
    assert first_media["type"] == "video"
    assert first_media["media"] == "attach://video"
    assert first_media["supports_streaming"] is True

    # Второй подписчик (2002) получил замену по кэшированному file_id БЕЗ повторной заливки MP4!
    second_body = api_calls[1][1]
    assert mp4_payload not in second_body
    second_media = json.loads(_extract_multipart_field(second_body, "media"))
    assert second_media["type"] == "video"
    assert second_media["media"] == "BAACAgIAAx0CTEST_CACHED_FILE_ID_V8"

    # В обоих постах строка загрузки исчезла, а статусы и кнопки от раннего клика сохранились!
    for _, raw_body in api_calls:
        med = json.loads(_extract_multipart_field(raw_body, "media"))
        cap = med["caption"]
        assert "⏳ Видеоклип загружается…" not in cap
        assert "Встала с кровати одна" in cap
        assert "Принято (Анна)" in cap
        assert "Ошибка записана (Борис)" in cap
        mk = json.loads(_extract_multipart_field(raw_body, "reply_markup"))
        assert mk["inline_keyboard"] == [
            [
                {"text": "✓ Я рядом", "callback_data": f"near:{alert_id}"},
                {"text": "✓ Ошибка записана", "callback_data": "noop"},
            ]
        ]


def test_wave8_edit_message_media_fallback_to_send_video_when_deleted(tmp_path: Path) -> None:
    calls: list[tuple[str, bytes]] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        ep = request.url.path.rsplit("/", 1)[-1]
        body = request.read()
        calls.append((ep, body))
        if ep == "sendPhoto":
            cid = _extract_multipart_field(body, "chat_id")
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 701 if cid == "1001" else 702}})
        if ep == "editMessageMedia":
            cid = _extract_multipart_field(body, "chat_id")
            if cid == "1001":
                return httpx.Response(400, json={"ok": False, "description": "Bad Request: message to edit not found"})
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 702}})
        if ep == "sendVideo":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "result": {
                        "message_id": 799,
                        "video": {"file_id": "BAACAgIAAx0CFALLBACK_FID"},
                    },
                },
            )
        return httpx.Response(200, json={"ok": True, "result": True})

    notifier = TelegramNotifier(
        bot_token="123456:FALLBACK_TOKEN",
        chat_ids=("1001", "2002"),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: None,
        is_fall_unacknowledged=lambda: False,
        subscribers_path=tmp_path / "subs.json",
        transport=httpx.MockTransport(mock_handler),
    )

    ev = AlertEvent(
        state=PatientState.SITTING_UP,
        severity=AlertSeverity.WARNING,
        title="Садится в кровати",
        description="Проверка фоллбэка.",
        timestamp=12.0,
        patient_track_id=1,
    )
    msg_ids = notifier.send_alert_photo(ev, b"\xff\xd8JPG", alert_id="clip_fb_01")
    calls.clear()

    mp4_data = b"\x00\x00\x00\x18ftypmp42FALLBACK_MP4"
    ok = notifier.replace_alert_photo_with_video(mp4_data, msg_ids, ev.title, alert_id="clip_fb_01")
    assert ok is True
    # Для 1001: editMessageMedia (400) -> фоллбэк sendVideo (200, кэширует file_id) -> для 2002 editMessageMedia по file_id!
    assert [ep for ep, _ in calls] == ["editMessageMedia", "sendVideo", "editMessageMedia"]
    assert mp4_data not in calls[2][1]
    med_2002 = json.loads(_extract_multipart_field(calls[2][1], "media"))
    assert med_2002["media"] == "BAACAgIAAx0CFALLBACK_FID"


# ==============================================================================
# 4. Волна 9: Криптография Telegram Mini App (tg_auth.py) и кнопка «Пульт» в боте
# ==============================================================================


def _sign_webapp_init_data(
    bot_token: str,
    user: dict[str, Any],
    auth_date: int,
    start_param: str = "",
    query_id: str = "AAHdF6IQAAAAAN0XohDhrOrc",
) -> str:
    import hashlib
    import hmac
    from urllib.parse import urlencode

    fields: dict[str, str] = {
        "auth_date": str(auth_date),
        "query_id": query_id,
        "user": json.dumps(user, ensure_ascii=False, separators=(",", ":")),
    }
    if start_param:
        fields["start_param"] = start_param

    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    sig = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": sig})


def test_wave9_tg_auth_init_data_session_tokens_and_proxy_detector() -> None:
    from app.tg_auth import (
        create_session_token,
        is_external_proxy_request,
        verify_session_token,
        verify_telegram_init_data,
    )

    bot_token = "987654:TEST_WAVE9_CRYPTO_BOT_SECRET"
    now_fixed = 1_765_500_000.0
    user_info = {
        "id": 550011,
        "first_name": "Елена",
        "last_name": "Романова",
        "username": "elena_family",
    }

    valid_init = _sign_webapp_init_data(
        bot_token=bot_token,
        user=user_info,
        auth_date=int(now_fixed) - 30,
        start_param="clip_20261113_001",
    )
    parsed = verify_telegram_init_data(valid_init, bot_token, max_age_sec=86400, now_ts=now_fixed)
    assert parsed == {
        "chat_id": "550011",
        "name": "Елена Романова",
        "username": "elena_family",
        "auth_date": int(now_fixed) - 30,
        "start_param": "clip_20261113_001",
    }

    # Допуск рассинхрона часов до -60 с проходит, а -120 с отвергается
    skew_ok = _sign_webapp_init_data(bot_token, user_info, auth_date=int(now_fixed) + 45)
    assert verify_telegram_init_data(skew_ok, bot_token, now_ts=now_fixed) is not None
    skew_too_far = _sign_webapp_init_data(bot_token, user_info, auth_date=int(now_fixed) + 120)
    assert verify_telegram_init_data(skew_too_far, bot_token, now_ts=now_fixed) is None

    # Истёкший auth_date, чужой токен и подмена параметра отвергаются
    expired_init = _sign_webapp_init_data(bot_token, user_info, auth_date=int(now_fixed) - 90_000)
    assert verify_telegram_init_data(expired_init, bot_token, max_age_sec=86400, now_ts=now_fixed) is None
    assert verify_telegram_init_data(valid_init, "987654:OTHER_BOT_SECRET", now_ts=now_fixed) is None
    tampered_init = valid_init.replace("550011", "999999")
    assert verify_telegram_init_data(tampered_init, bot_token, now_ts=now_fixed) is None
    assert verify_telegram_init_data(valid_init + "&hash=" + ("0" * 64), bot_token, now_ts=now_fixed) is None
    assert verify_telegram_init_data("", bot_token, now_ts=now_fixed) is None

    # Подписанный сессионный токен Mini App
    tok = create_session_token(
        chat_id="550011",
        name="Елена Романова",
        bot_token=bot_token,
        ttl_sec=3600,
        now_ts=now_fixed,
        username="elena_family",
    )
    assert "." in tok
    verified_tok = verify_session_token(tok, bot_token, now_ts=now_fixed + 120)
    assert verified_tok == {
        "chat_id": "550011",
        "name": "Елена Романова",
        "username": "elena_family",
        "iat": int(now_fixed),
        "exp": int(now_fixed) + 3600,
    }
    assert verify_session_token(tok, bot_token, now_ts=now_fixed + 3600) is None
    assert verify_session_token(tok, "987654:WRONG_SECRET", now_ts=now_fixed) is None
    b64_part, sig_part = tok.split(".")
    flipped_sig = ("0" if sig_part[0] != "0" else "1") + sig_part[1:]
    assert verify_session_token(f"{b64_part}.{flipped_sig}", bot_token, now_ts=now_fixed) is None

    # Детектор внешнего прокси vs локальной сети
    assert is_external_proxy_request({"host": "testserver"}, "testclient") is False
    assert is_external_proxy_request({"host": "localhost:8090"}, "127.0.0.1") is False
    assert is_external_proxy_request({"host": "192.168.1.100:8090"}, "192.168.1.101") is False
    assert is_external_proxy_request({"host": "192.168.1.100:8090"}, "10.8.0.15") is False
    assert is_external_proxy_request({"host": "fallguard.local:8090"}, "172.20.1.5") is False

    assert is_external_proxy_request({"host": "nofall.example.com"}, "192.168.1.101") is True
    assert is_external_proxy_request({"x-forwarded-host": "nofall.example.com"}, "testclient") is True
    assert is_external_proxy_request({"host": "192.168.1.100:8090"}, "192.168.1.6") is True
    assert is_external_proxy_request({"host": "192.168.1.100:8090"}, "192.168.211.2") is True
    assert is_external_proxy_request({"host": "192.168.1.100:8090"}, "95.165.10.20") is True


def test_wave9_notifier_menu_button_app_command_and_webapp_whitelist(tmp_path: Path) -> None:
    calls: list[tuple[str, bytes]] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        ep = request.url.path.rsplit("/", 1)[-1]
        body = request.read()
        calls.append((ep, body))
        if ep == "getMe":
            return httpx.Response(
                200,
                json={"ok": True, "result": {"id": 99, "username": "NoFallenAi"}},
            )
        if ep == "sendPhoto":
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 801}})
        return httpx.Response(200, json={"ok": True, "result": True})

    notifier = TelegramNotifier(
        bot_token="123456:WAVE9_BOT_TOKEN",
        chat_ids=("1001",),
        on_pause_cmd=lambda _: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: b"\xff\xd8\xff\xd9SNAP",
        is_fall_unacknowledged=lambda: False,
        subscribers_path=tmp_path / "subscribers.json",
        webapp_url="https://nofall.example.com",
        transport=httpx.MockTransport(mock_handler),
    )

    notifier.register_subscriber("2002", name="Алексей", username="alex_rel")
    notifier.set_subscriber_enabled("2002", False)
    # Даже с выключенными уведомлениями родственник остаётся авторизованным в Mini App!
    assert notifier.is_authorized_webapp_user("1001") is True
    assert notifier.is_authorized_webapp_user("2002") is True
    assert notifier.is_authorized_webapp_user("9999") is False

    removed_ok, _ = notifier.remove_subscriber("2002")
    assert removed_ok is True
    assert notifier.is_authorized_webapp_user("2002") is False

    with httpx.Client(transport=notifier._transport) as client:
        assert notifier.sync_bot_profile_and_commands(client=client) is True
        ep_names = [ep for ep, _ in calls]
        assert ep_names == ["getMe", "setMyCommands", "setMyCommands", "setChatMenuButton"]

        cmd_payload = json.loads(calls[1][1].decode("utf-8"))
        assert cmd_payload["commands"][0] == {
            "command": "app",
            "description": "Открыть видеопульт Mini App",
        }
        menu_payload = json.loads(calls[3][1].decode("utf-8"))
        assert menu_payload == {
            "menu_button": {
                "type": "web_app",
                "text": "Пульт",
                "web_app": {"url": "https://nofall.example.com"},
            }
        }

        # Команда /app отправляет инлайн-кнопку web_app
        calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 501,
                "message": {
                    "chat": {"id": 3003, "first_name": "Ольга"},
                    "from": {"id": 3003, "first_name": "Ольга"},
                    "text": "/app",
                },
            },
        )
        assert len(calls) == 1 and calls[0][0] == "sendMessage"
        app_reply_mk = json.loads(_extract_multipart_field(calls[0][1], "reply_markup"))
        assert app_reply_mk == {
            "inline_keyboard": [
                [
                    {
                        "text": "📱 Открыть пульт FallGuard",
                        "web_app": {"url": "https://nofall.example.com"},
                    }
                ]
            ]
        }

        # Приветствие /start упоминает кнопку «Пульт», а посты тревог хранят ровно 2 кнопки
        calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 502,
                "message": {
                    "chat": {"id": 3003, "first_name": "Ольга"},
                    "from": {"id": 3003, "first_name": "Ольга"},
                    "text": "/start",
                },
            },
        )
        assert calls[0][0] == "sendPhoto"
        welcome_cap = _extract_multipart_field(calls[0][1], "caption")
        assert "Пульт" in welcome_cap

    calls.clear()
    ev = AlertEvent(
        state=PatientState.LEFT_BED,
        severity=AlertSeverity.WARNING,
        title="Встала с кровати одна",
        description="Проверка неизменности тревожных кнопок.",
        timestamp=50.0,
        patient_track_id=1,
    )
    notifier.send_alert_photo(ev, b"\xff\xd8\xff\xd9ALERT", alert_id="w9_alert_1")
    send_photo_calls = [body for ep, body in calls if ep == "sendPhoto"]
    assert len(send_photo_calls) >= 1
    alert_mk = json.loads(_extract_multipart_field(send_photo_calls[0], "reply_markup"))
    assert len(alert_mk["inline_keyboard"]) == 1
    assert len(alert_mk["inline_keyboard"][0]) == 2


