from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs

import cv2
import httpx
import numpy as np
import pytest

from app.archive import ClipArchiveStore
from app.notifier import BOT_COMMANDS_RU, TelegramNotifier, _build_inline_keyboard
from app.qr_svg import encode_qr_matrix, render_qr_svg
from app.runtime_prefs import RuntimePrefsStore
from app.state_machine import AlertEvent, AlertSeverity, PatientState


def _make_event(
    state: PatientState = PatientState.FALLEN,
    severity: AlertSeverity = AlertSeverity.CRITICAL,
    title: str = "Падение / сползание вне кровати!",
    desc: str = "Обнаружено горизонтальное положение на полу.",
    ts: float = 100.0,
) -> AlertEvent:
    return AlertEvent(
        state=state,
        severity=severity,
        title=title,
        description=desc,
        timestamp=ts,
        patient_track_id=1,
    )


def _rasterize_qr_svg(svg_text: str, scale: int = 12) -> np.ndarray:
    """Растеризует непосредственно сгенерированный <svg> QR-кода в BGR-массив для cv2.QRCodeDetector."""
    vb_match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_text)
    assert vb_match is not None
    w_mod = int(vb_match.group(1))
    h_mod = int(vb_match.group(2))

    canvas = np.full((h_mod * scale, w_mod * scale, 3), 255, dtype=np.uint8)
    d_match = re.search(r'd="([^"]+)"', svg_text)
    assert d_match is not None
    d_str = d_match.group(1)

    for mx, my, run in re.findall(r"M(\d+) (\d+)h(\d+)v1h-\d+z", d_str):
        x0 = int(mx) * scale
        y0 = int(my) * scale
        x1 = (int(mx) + int(run)) * scale
        y1 = (int(my) + 1) * scale
        canvas[y0:y1, x0:x1] = 0
    return canvas


def test_clip_archive_save_attach_reload_and_latest_mp4(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    store = ClipArchiveStore(clips_dir=clips_dir, max_clips=10, max_total_mb=50.0)

    jpg_data = b"\xff\xd8\xff\xe0" + b"J" * 512
    mp4_data = b"\x00\x00\x00\x18ftypmp42" + b"V" * 2048

    ev1 = _make_event(state=PatientState.SITTING_UP, severity=AlertSeverity.WARNING, title="Села в кровати")
    rec1 = store.save_event_snapshot(ev1, jpg_data, unix_ts=1760000100.0)
    assert re.fullmatch(r"\d{8}_\d{6}_sitting_up_[0-9a-f]{4}", rec1["id"])
    assert rec1["has_jpg"] is True
    assert rec1["has_mp4"] is False
    assert rec1["jpg_url"] == f"/api/clips/{rec1['id']}.jpg"
    assert rec1["mp4_url"] is None
    assert rec1["size_bytes"] == len(jpg_data)
    assert store.get_latest_mp4() is None

    updated = store.attach_mp4(rec1["id"], mp4_data)
    assert updated is not None
    assert updated["has_mp4"] is True
    assert updated["mp4_url"] == f"/api/clips/{rec1['id']}.mp4"
    assert updated["size_bytes"] == len(jpg_data) + len(mp4_data)

    # Перезагрузка с диска новым инстансом
    reloaded = ClipArchiveStore(clips_dir=clips_dir, max_clips=10, max_total_mb=50.0)
    items = reloaded.list_clips()
    assert len(items) == 1
    assert items[0]["id"] == rec1["id"]
    assert items[0]["has_mp4"] is True

    latest = reloaded.get_latest_mp4()
    assert latest is not None
    assert latest[0] == mp4_data
    assert "Села в кровати" in latest[1]

    stats = reloaded.storage_stats()
    assert stats["count"] == 1
    assert stats["mp4_count"] == 1
    assert stats["used_bytes"] == len(jpg_data) + len(mp4_data)


def test_clip_archive_quota_rotation_and_path_traversal_guard(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    # Квота по количеству (max_clips=3) и по объёму (~0.02 МБ = ~20.9 КБ)
    store = ClipArchiveStore(clips_dir=clips_dir, max_clips=3, max_total_mb=0.02)

    ids: list[str] = []
    for idx in range(4):
        ev = _make_event(title=f"Событие #{idx}")
        rec = store.save_event_snapshot(ev, b"X" * 4000, unix_ts=1760000200.0 + idx * 10)
        ids.append(rec["id"])

    # Первый самый старый клип должен быть автоматически вытеснен и удалён с диска
    assert store.get_clip(ids[0]) is None
    assert not (clips_dir / f"{ids[0]}.jpg").exists()
    assert len(store.list_clips()) == 3

    # Прикрепляем крупный MP4 (16 КБ) к свежему клипу -> старые вытесняются по квоте МБ
    store.attach_mp4(ids[3], b"M" * 16000)
    remaining = store.list_clips()
    assert remaining[0]["id"] == ids[3]
    assert store.storage_stats()["used_bytes"] <= int(0.02 * 1024 * 1024)

    # Строгая защита от path traversal и посторонних расширений
    secret_outside = tmp_path / "secret.jpg"
    secret_outside.write_bytes(b"secret")
    assert store.resolve_media_path("../secret", "jpg") is None
    assert store.resolve_media_path("..\\secret", "jpg") is None
    assert store.resolve_media_path("sub/clip", "mp4") is None
    assert store.resolve_media_path(ids[3], "json") is None
    assert store.resolve_media_path("index", "json") is None
    valid_mp4 = store.resolve_media_path(ids[3], ".mp4")
    assert valid_mp4 is not None and valid_mp4.is_file()

    # Удаление и полная очистка
    assert store.delete_clip("../secret") is False
    cleared = store.clear_all()
    assert cleared >= 1
    assert store.list_clips() == []
    assert store.storage_stats()["used_bytes"] == 0


def test_runtime_prefs_presets_clamping_and_persistence(tmp_path: Path) -> None:
    prefs_path = tmp_path / "runtime_prefs.json"
    store = RuntimePrefsStore(prefs_path)

    initial = store.get()
    assert initial["sensitivity_preset"] == "standard"
    assert (
        initial["sit_confirm_seconds"],
        initial["leave_confirm_seconds"],
        initial["fall_confirm_seconds"],
    ) == (1.8, 2.0, 1.8)
    assert initial["enable_help_gesture"] is True
    assert initial["prolonged_absence_minutes"] == 20
    assert initial["immobility_reminder_minutes"] == 0
    assert initial["night_strict_mode"] is True
    assert (initial["night_start_hour"], initial["night_end_hour"]) == (22, 7)

    # Переключение пресета синхронизирует таймеры (даже если фронтенд прислал весь старый объект)
    high = store.update({**initial, "sensitivity_preset": "high"})
    assert high["sensitivity_preset"] == "high"
    assert (
        high["sit_confirm_seconds"],
        high["leave_confirm_seconds"],
        high["fall_confirm_seconds"],
    ) == (1.2, 1.4, 1.3)

    gentle = store.update({"sensitivity_preset": "gentle"})
    assert gentle["sensitivity_preset"] == "gentle"
    assert (
        gentle["sit_confirm_seconds"],
        gentle["leave_confirm_seconds"],
        gentle["fall_confirm_seconds"],
    ) == (2.8, 3.0, 2.2)

    # Ручная правка таймера переводит пресет в custom + проверка клэмпинга границ
    clamped = store.update(
        {
            "sit_confirm_seconds": 0.1,          # clamp -> 0.8
            "leave_confirm_seconds": 99.0,       # clamp -> 6.0
            "fall_confirm_seconds": 9.5,         # clamp -> 5.0
            "missing_confirm_seconds": 2.0,      # clamp -> 5.0
            "help_confirm_seconds": 20.0,        # clamp -> 8.0
            "prolonged_absence_minutes": 2,      # clamp -> 5 (но 0 остаётся 0)
            "immobility_reminder_minutes": 15,   # clamp -> 30
            "night_start_hour": 29,              # clamp -> 23
            "night_end_hour": -4,                # clamp -> 0
            "enable_help_gesture": False,
        }
    )
    assert clamped["sensitivity_preset"] == "custom"
    assert clamped["sit_confirm_seconds"] == 0.8
    assert clamped["leave_confirm_seconds"] == 6.0
    assert clamped["fall_confirm_seconds"] == 5.0
    assert clamped["missing_confirm_seconds"] == 5.0
    assert clamped["help_confirm_seconds"] == 8.0
    assert clamped["prolonged_absence_minutes"] == 5
    assert clamped["immobility_reminder_minutes"] == 30
    assert (clamped["night_start_hour"], clamped["night_end_hour"]) == (23, 0)
    assert clamped["enable_help_gesture"] is False

    # Отключение напоминаний нулём сохраняется как 0
    off_timers = store.update({"prolonged_absence_minutes": 0, "immobility_reminder_minutes": 0})
    assert off_timers["prolonged_absence_minutes"] == 0
    assert off_timers["immobility_reminder_minutes"] == 0

    # Проверка чтения с диска новым инстансом
    reloaded = RuntimePrefsStore(prefs_path)
    assert reloaded.get() == off_timers


@pytest.mark.parametrize(
    "payload",
    [
        "https://t.me/NoFallenAi",
        "https://t.me/NoFallenAi?start=family_guard",
        "OK",
        "FallGuard: уход 24/7",
        "https://192.168.1.101:8090/?room=grandma&mode=night-watch&ref=qr_test_version_5_or_6_long_url_check",
    ],
)
def test_qr_svg_zero_dep_decoded_by_opencv(payload: str) -> None:
    svg = render_qr_svg(payload, border=3)
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ')
    assert 'shape-rendering="crispEdges"' in svg
    assert '<path fill="currentColor" d="M' in svg

    # 1) Проверяем декодирование матрицы напрямую
    mat = encode_qr_matrix(payload)
    n = len(mat)
    pad = 4
    scale = 10
    img_mat = np.full(((n + 2 * pad) * scale, (n + 2 * pad) * scale, 3), 255, dtype=np.uint8)
    for r, row in enumerate(mat):
        for c, dark in enumerate(row):
            if dark:
                y0, y1 = (r + pad) * scale, (r + pad + 1) * scale
                x0, x1 = (c + pad) * scale, (c + pad + 1) * scale
                img_mat[y0:y1, x0:x1] = 0

    detector = cv2.QRCodeDetector()
    decoded_mat, _, _ = detector.detectAndDecode(img_mat)
    assert decoded_mat == payload

    # 2) Проверяем декодирование непосредственно сгенерированного SVG-пути
    img_svg = _rasterize_qr_svg(svg, scale=12)
    decoded_svg, _, _ = detector.detectAndDecode(img_svg)
    assert decoded_svg == payload


def test_telegram_notifier_report_clip_photo_via_mock_transport(tmp_path: Path) -> None:
    calls: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        body_bytes = request.read()
        calls.append((request.method, endpoint, body_bytes))
        if endpoint == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {"id": 1, "username": "TestCareBot"}})
        if endpoint == "sendPhoto":
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 77}})
        return httpx.Response(200, json={"ok": True, "result": True})

    mock_transport = httpx.MockTransport(handler)
    subs_file = tmp_path / "subscribers.json"

    clip_holder: dict[str, tuple[bytes, str] | None] = {
        "clip": (b"\x00\x00\x00\x18ftypmp42MOCKVIDEO", "Села в кровати (2026-11-12 05:30:00)")
    }
    paused_vals: list[float] = []

    notifier = TelegramNotifier(
        bot_token="123456:TEST_MOCK_TOKEN_DO_NOT_SEND",
        chat_ids=(),
        on_pause_cmd=lambda s: paused_vals.append(s),
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: b"\xff\xd8\xff\xd9JPEG_SNAP",
        is_fall_unacknowledged=lambda: False,
        subscribers_path=subs_file,
        get_status_text=lambda: "Состояние: В кровати",
        get_report_text=lambda: "📊 Сводка за 24 ч: сон 7 ч 40 мин, посадок: 2, падений: 0.",
        get_latest_clip=lambda: clip_holder["clip"],
        transport=mock_transport,
    )

    # 1. Проверяем меню команд и клавиатуру
    cmd_names = [c["command"] for c in BOT_COMMANDS_RU]
    assert cmd_names == [
        "status",
        "photo",
        "report",
        "clip",
        "pause15",
        "pause60",
        "resume",
        "fallsonly",
        "allevents",
        "mute",
        "unmute",
    ]

    kb = json.loads(_build_inline_keyboard())
    assert len(kb["inline_keyboard"]) == 3
    row3_cb = [btn["callback_data"] for btn in kb["inline_keyboard"][2]]
    assert row3_cb == ["report", "clip"]

    with httpx.Client(transport=mock_transport) as client:
        assert notifier.sync_bot_profile_and_commands(client=client) is True

        # 2. Команда /photo (автопривязка чата 5001 + отправка свежего кадра)
        calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 1,
                "message": {
                    "chat": {"id": 5001, "first_name": "Анна"},
                    "from": {"id": 5001, "first_name": "Анна", "username": "anna_care"},
                    "text": "/photo",
                },
            },
        )
        assert "5001" in notifier.get_recipient_chat_ids()
        assert any(ep == "sendPhoto" for _, ep, _ in calls)

        # 3. Команда /report и callback_query "report"
        calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 2,
                "message": {"chat": {"id": 5001}, "from": {"id": 5001}, "text": "/report"},
            },
        )
        assert len(calls) == 1 and calls[0][1] == "sendMessage"
        parsed_msg = parse_qs(calls[0][2].decode("utf-8"))
        assert "сон 7 ч 40 мин" in parsed_msg["text"][0]

        calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 3,
                "callback_query": {
                    "id": "cb_rep_1",
                    "data": "report",
                    "from": {"id": 5001},
                    "message": {"chat": {"id": 5001}},
                },
            },
        )
        assert [ep for _, ep, _ in calls] == ["sendMessage", "answerCallbackQuery"]

        # 4. Команда /clip и callback_query "clip" при наличии MP4 и при пустом архиве
        calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 4,
                "message": {"chat": {"id": 5001}, "from": {"id": 5001}, "text": "/clip@TestCareBot"},
            },
        )
        assert len(calls) == 1 and calls[0][1] == "sendVideo"
        assert b"ftypmp42MOCKVIDEO" in calls[0][2]

        clip_holder["clip"] = None
        calls.clear()
        notifier._handle_update(
            client,
            {
                "update_id": 5,
                "callback_query": {
                    "id": "cb_clip_empty",
                    "data": "clip",
                    "from": {"id": 5001},
                    "message": {"chat": {"id": 5001}},
                },
            },
        )
        assert [ep for _, ep, _ in calls] == ["sendMessage", "answerCallbackQuery"]
        empty_ans = parse_qs(calls[1][2].decode("utf-8"))["text"][0]
        assert "пуст" in empty_ans.lower()


def test_qr_svg_validation_and_archive_state_filtering(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        render_qr_svg("")
    with pytest.raises(ValueError):
        render_qr_svg("A" * 160)

    store = ClipArchiveStore(tmp_path / "clips")
    r_fall = store.save_event_snapshot(
        _make_event(state=PatientState.FALLEN, title="Падение"),
        b"\xff\xd8fall",
        unix_ts=1760000500.0,
    )
    r_sit = store.save_event_snapshot(
        _make_event(state=PatientState.SITTING_UP, severity=AlertSeverity.WARNING, title="Посадка"),
        b"\xff\xd8sit",
        unix_ts=1760000510.0,
    )
    only_fall = store.list_clips(state_filter="fallen")
    assert [c["id"] for c in only_fall] == [r_fall["id"]]
    assert store.delete_clip(r_sit["id"]) is True
    assert store.get_clip(r_sit["id"]) is None
    assert [c["id"] for c in store.list_clips()] == [r_fall["id"]]

