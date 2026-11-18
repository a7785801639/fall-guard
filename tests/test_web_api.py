from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engine import FallGuardEngine
from app.patient_id import ANCHOR_SCHEMA_VERSION
from app.state_machine import AlertEvent, AlertSeverity, PatientState
from app.web import create_web_app


def _rasterize_qr_svg(svg_text: str, scale: int = 12) -> np.ndarray:
    """Растеризует SVG с QR-кодом в BGR-матрицу для реального декодирования через OpenCV."""
    vb_match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_text)
    assert vb_match is not None, "В SVG отсутствует корректный viewBox"
    w_mod = int(vb_match.group(1))
    h_mod = int(vb_match.group(2))

    canvas = np.full((h_mod * scale, w_mod * scale, 3), 255, dtype=np.uint8)
    d_match = re.search(r'd="([^"]+)"', svg_text)
    assert d_match is not None, "В SVG отсутствует контур <path d=...>"
    d_str = d_match.group(1)

    for mx, my, run in re.findall(r"M(\d+) (\d+)h(\d+)v1h-\d+z", d_str):
        x0 = int(mx) * scale
        y0 = int(my) * scale
        x1 = (int(mx) + int(run)) * scale
        y1 = (int(my) + 1) * scale
        canvas[y0:y1, x0:x1] = 0
    return canvas


def _build_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        rtsp_url="",
        tg_bot_token="123456:TEST_ISOLATED_TOKEN",
        tg_chat_ids=("100001",),
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        web_host="127.0.0.1",
        web_port=8090,
        infer_fps=5.0,
        capture_fps=6.0,
        pre_event_seconds=5.0,
        post_event_seconds=4.0,
        pose_model="yolo11s-pose.pt",
        device="cpu",
        pose_conf=0.48,
        bed_lock_seconds=20.0,
        initial_bed_lock_seconds=5.0,
        sit_confirm_seconds=1.8,
        leave_confirm_seconds=2.0,
        fall_confirm_seconds=1.8,
        missing_confirm_seconds=12.0,
        alert_cooldown_seconds=45.0,
        repeat_fall_alert_seconds=15.0,
        max_fall_repeats=3,
        suppress_sitting_when_caregiver=True,
        face_match_threshold=0.38,
        tg_bot_username="NoFallenAi",
        frame_rotate=0,
        frame_flip_h=False,
        low_light_enhance=False,
    )


@pytest.fixture()
def isolated_api(tmp_path: Path) -> tuple[TestClient, FallGuardEngine, Path]:
    """
    Создаёт полностью изолированное FastAPI-приложение в tmp_path без запуска фоновых потоков
    и с жёсткой блокировкой любых внешних HTTP-запросов.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Подкладываем валидный суточный якорь внешности на диск для проверки /api/reset-anchor
    anchor_payload = {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "updated_wall_ts": round(time.time(), 3),
        "anchor_vec": [1.0] + [0.0] * 47,
        "last_patient_pos": [0.52, 0.48],
    }
    (data_dir / "patient_anchor.json").write_text(
        json.dumps(anchor_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    settings = _build_test_settings(tmp_path)
    engine = FallGuardEngine(settings)

    # Гарантия 0 сетевых вызовов даже при случайном обращении к TelegramNotifier
    def _forbid_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Неожиданный внешний сетевой запрос в тесте: {request.method} {request.url}")

    engine.notifier._transport = httpx.MockTransport(_forbid_network)

    app = create_web_app(engine)
    client = TestClient(app)
    return client, engine, data_dir


def test_status_24h_timeline_and_report_endpoints(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    client, engine, _ = isolated_api

    # Записываем активность в таймлайн (включая событие старше 6 ч для проверки 24-часового окна)
    base_ts = time.time()
    engine.timeline.record("in_bed", True, now_ts=base_ts - 36000.0)
    engine.timeline.record("in_bed", True, now_ts=base_ts - 35980.0)
    engine.timeline.record("sitting_up", True, now_ts=base_ts - 600.0)
    engine.timeline.record("sitting_up", True, now_ts=base_ts - 590.0)
    engine.timeline.record("in_bed", True, now_ts=base_ts - 60.0)
    engine.timeline.record("in_bed", True, now_ts=base_ts - 45.0)

    ev = AlertEvent(
        state=PatientState.SITTING_UP,
        severity=AlertSeverity.WARNING,
        title="Бабушка садится в кровати",
        description="Подъём корпуса выше порога.",
        timestamp=100.0,
        patient_track_id=1,
    )
    saved_clip = engine.archive.save_event_snapshot(
        ev,
        b"\xff\xd8\xff\xe0" + b"J" * 256,
        unix_ts=base_ts - 590.0,
    )

    # 1. GET /api/status?window_sec=86400
    resp_status = client.get("/api/status?window_sec=86400")
    assert resp_status.status_code == 200
    st_body = resp_status.json()
    assert st_body["state"] == "idle"
    assert st_body["timeline"]["window_sec"] == 86400
    assert len(st_body["timeline"]["spans"]) >= 2
    assert "prefs" in st_body
    assert "care" in st_body
    assert "diagnostics" in st_body["care"]
    assert "summary_24h" in st_body["care"]
    assert st_body["clips_stats"]["count"] == 1
    assert st_body["recent_clips"][0]["id"] == saved_clip["id"]

    # 2. GET /api/timeline?window_sec=86400 и сравнение с 6-часовым окном (21600 с)
    resp_tl_6h = client.get("/api/timeline?window_sec=21600")
    assert resp_tl_6h.status_code == 200
    spans_6h = resp_tl_6h.json()["timeline"]["spans"]

    resp_tl = client.get("/api/timeline?window_sec=86400")
    assert resp_tl.status_code == 200
    tl_body = resp_tl.json()
    assert tl_body["ok"] is True
    assert tl_body["timeline"]["window_sec"] == 86400
    assert len(tl_body["timeline"]["spans"]) == 3
    assert len(spans_6h) == 2
    assert any(c["id"] == saved_clip["id"] for c in tl_body["clips"])
    assert tl_body["summary_24h"]["sit_up_count"] >= 1

    # 3. GET /api/report
    resp_rep = client.get("/api/report")
    assert resp_rep.status_code == 200
    rep_body = resp_rep.json()
    assert rep_body["ok"] is True
    assert "Сводка ухода FallGuard за 24 часа" in rep_body["text"]
    assert "Посадок в кровати:" in rep_body["text"]
    assert "Записей в видеоархиве: 1" in rep_body["text"]
    assert "summary_24h" in rep_body
    assert "diagnostics" in rep_body


def test_prefs_recalibrate_baseline_and_reset_anchor(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    client, engine, data_dir = isolated_api

    # 1. GET /api/prefs
    r_get = client.get("/api/prefs")
    assert r_get.status_code == 200
    get_data = r_get.json()
    assert get_data["ok"] is True
    assert get_data["prefs"]["sensitivity_preset"] == "standard"

    # 2. POST /api/prefs — переключение пресета и параметров ухода на лету
    r_post = client.post(
        "/api/prefs",
        json={
            "sensitivity_preset": "high",
            "enable_help_gesture": False,
            "prolonged_absence_minutes": 35.0,
            "immobility_reminder_minutes": 120.0,
            "night_strict_mode": False,
        },
    )
    assert r_post.status_code == 200
    post_data = r_post.json()
    assert post_data["ok"] is True
    assert post_data["prefs"]["sensitivity_preset"] == "high"
    assert post_data["prefs"]["sit_confirm_seconds"] == 1.2
    assert post_data["prefs"]["enable_help_gesture"] is False
    assert post_data["prefs"]["prolonged_absence_minutes"] == 35.0
    assert post_data["prefs"]["immobility_reminder_minutes"] == 120.0
    assert post_data["care"]["diagnostics"]["enable_help_gesture"] is False
    assert (data_dir / "runtime_prefs.json").is_file()

    # 3. POST /api/recalibrate-baseline — сброс накопленного эталона позы сна
    engine.fsm._baseline.samples = 12
    engine.fsm._baseline.shoulder_y = 0.44
    assert engine.get_care_diagnostics()["baseline"]["calibrated"] is True

    r_recal = client.post("/api/recalibrate-baseline")
    assert r_recal.status_code == 200
    recal_data = r_recal.json()
    assert recal_data["ok"] is True
    assert recal_data["diagnostics"]["baseline"]["calibrated"] is False
    assert recal_data["diagnostics"]["baseline"]["samples"] == 0

    # 4. POST /api/reset-anchor — удаление суточного якоря одежды с диска и из памяти
    anchor_file = data_dir / "patient_anchor.json"
    assert anchor_file.exists()
    assert engine.identifier.has_appearance_anchor is True

    r_anchor = client.post("/api/reset-anchor")
    assert r_anchor.status_code == 200
    anchor_data = r_anchor.json()
    assert anchor_data == {"ok": True, "has_appearance_anchor": False}
    assert not anchor_file.exists()


def test_telegram_qr_svg_endpoint_decodes_via_cv2(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    client, _, _ = isolated_api

    resp = client.get("/api/telegram-qr.svg")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert "no-cache" in resp.headers.get("cache-control", "")

    svg_text = resp.text
    assert svg_text.startswith("<svg ")
    bgr = _rasterize_qr_svg(svg_text, scale=12)
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
    assert decoded == "https://t.me/NoFallenAi?start=watch"


def test_clips_endpoints_http_range_206_and_deletion(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    client, engine, _ = isolated_api

    jpg_payload = b"\xff\xd8\xff\xe0" + bytes(range(128))
    mp4_payload = b"\x00\x00\x00\x18ftypmp42" + bytes(i % 256 for i in range(200))

    ev1 = AlertEvent(
        state=PatientState.FALLEN,
        severity=AlertSeverity.CRITICAL,
        title="Падение вне кровати!",
        description="Горизонтальное положение на полу.",
        timestamp=200.0,
        patient_track_id=1,
    )
    ev2 = AlertEvent(
        state=PatientState.LEFT_BED,
        severity=AlertSeverity.WARNING,
        title="Вышла из кровати",
        description="Самостоятельный подъём.",
        timestamp=250.0,
        patient_track_id=1,
    )
    rec1 = engine.archive.save_event_snapshot(ev1, jpg_payload, unix_ts=1765000100.0)
    engine.archive.attach_mp4(rec1["id"], mp4_payload)
    rec2 = engine.archive.save_event_snapshot(ev2, jpg_payload, unix_ts=1765000200.0)

    clip_id = rec1["id"]

    # GET /api/clips (список и фильтр по state)
    r_list = client.get("/api/clips")
    assert r_list.status_code == 200
    list_body = r_list.json()
    assert list_body["ok"] is True
    assert len(list_body["clips"]) == 2
    assert list_body["stats"]["count"] == 2
    assert list_body["stats"]["mp4_count"] == 1

    r_filtered = client.get("/api/clips?state=fallen")
    assert r_filtered.status_code == 200
    assert [c["id"] for c in r_filtered.json()["clips"]] == [clip_id]

    # GET /api/clips/{id}
    r_meta = client.get(f"/api/clips/{clip_id}")
    assert r_meta.status_code == 200
    assert r_meta.json()["clip"]["id"] == clip_id
    assert r_meta.json()["clip"]["has_mp4"] is True

    # GET /api/clips/{id}.jpg
    r_jpg = client.get(f"/api/clips/{clip_id}.jpg")
    assert r_jpg.status_code == 200
    assert r_jpg.headers["content-type"].startswith("image/jpeg")
    assert r_jpg.content == jpg_payload

    # GET /api/clips/{id}.mp4 с заголовком Range: bytes=0-15 -> 206 Partial Content
    r_range = client.get(f"/api/clips/{clip_id}.mp4", headers={"Range": "bytes=0-15"})
    assert r_range.status_code == 206
    assert r_range.headers["content-type"].startswith("video/mp4")
    assert r_range.headers["content-range"] == f"bytes 0-15/{len(mp4_payload)}"
    assert len(r_range.content) == 16
    assert r_range.content == mp4_payload[:16]

    # Проверка скачивания ?download=1 -> Content-Disposition: attachment
    r_dl = client.get(f"/api/clips/{clip_id}.mp4?download=1")
    assert r_dl.status_code == 200
    assert "attachment" in r_dl.headers.get("content-disposition", "")

    # Защита от path traversal в clip_id
    assert client.get("/api/clips/..%5Czones").status_code == 404
    assert client.get("/api/clips/nonexistent_20990101_000000_fallen_abcd").status_code == 404

    # DELETE /api/clips/{id}
    r_del_one = client.delete(f"/api/clips/{clip_id}")
    assert r_del_one.status_code == 200
    assert r_del_one.json()["stats"]["count"] == 1
    assert client.get(f"/api/clips/{clip_id}").status_code == 404
    assert client.delete(f"/api/clips/{clip_id}").status_code == 404

    # DELETE /api/clips (очистка всего архива)
    assert rec2["id"] in [c["id"] for c in r_del_one.json()["clips"]]
    r_clear = client.delete("/api/clips")
    assert r_clear.status_code == 200
    clear_body = r_clear.json()
    assert clear_body["ok"] is True
    assert clear_body["removed"] == 1
    assert clear_body["clips"] == []
    assert clear_body["stats"]["count"] == 0


def test_face_photo_endpoint_and_path_traversal_protection(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    client, _, data_dir = isolated_api

    patient_dir = data_dir / "faces" / "patient"
    patient_dir.mkdir(parents=True, exist_ok=True)

    valid_jpg = b"\xff\xd8\xff\xe0" + b"FACE_THUMB" * 16
    (patient_dir / "cam_patient_1.jpg").write_bytes(valid_jpg)
    # Файл с запрещённым расширением внутри самого каталога ролей
    (patient_dir / "notes.json").write_text('{"secret": true}', encoding="utf-8")
    # Приманки в родительских директориях для проверки блокировки выхода из песочницы
    (data_dir / "zones.json").write_text('{"bed_polygon": []}', encoding="utf-8")
    (data_dir / "faces" / "zones.json").write_text('{"leak": true}', encoding="utf-8")
    (data_dir / "faces" / "evil.jpg").write_bytes(valid_jpg)

    # Легитимная отдача миниатюры эталона
    r_ok = client.get("/api/faces/patient/photo/cam_patient_1.jpg")
    assert r_ok.status_code == 200
    assert r_ok.content == valid_jpg

    # Неверная роль -> 400
    r_bad_role = client.get("/api/faces/intruder/photo/cam_patient_1.jpg")
    assert r_bad_role.status_code == 400

    # Попытки path traversal (URL-encoded /, \, двойные точки и запрещённые расширения) -> 404
    traversal_urls = [
        "/api/faces/patient/photo/..%2Fzones.json",
        "/api/faces/patient/photo/..%2F..%2Fzones.json",
        "/api/faces/patient/photo/..%2Fevil.jpg",
        "/api/faces/patient/photo/..%5Czones.json",
        "/api/faces/patient/photo/..%5Cevil.jpg",
        "/api/faces/patient/photo/%2e%2e",
        "/api/faces/patient/photo/notes.json",
        "/api/faces/patient/photo/missing.jpg",
    ]
    for target in traversal_urls:
        resp = client.get(target)
        assert resp.status_code == 404, f"Ожидался 404 для {target}, получен {resp.status_code}"


def test_pwa_routes_manifest_sw_and_icon(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    client, _, _ = isolated_api

    # 1. GET /manifest.webmanifest
    r_manifest = client.get("/manifest.webmanifest")
    assert r_manifest.status_code == 200
    assert r_manifest.headers["content-type"].startswith("application/manifest+json")
    manifest_json = r_manifest.json()
    assert "FallGuard" in manifest_json.get("name", "")
    assert manifest_json.get("display") in ("standalone", "minimal-ui", "fullscreen")

    # 2. GET /sw.js (проверка критического заголовка Service-Worker-Allowed: /)
    r_sw = client.get("/sw.js")
    assert r_sw.status_code == 200
    assert r_sw.headers["content-type"].startswith("application/javascript")
    assert r_sw.headers.get("service-worker-allowed") == "/"
    assert "no-store" in r_sw.headers.get("cache-control", "")
    assert "self.addEventListener" in r_sw.text

    # 3. GET /icon.svg
    r_icon = client.get("/icon.svg")
    assert r_icon.status_code == 200
    assert r_icon.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in r_icon.text


def _make_pose_obs(
    track_id: int,
    bbox_norm: tuple[float, float, float, float],
    horizontal: bool = True,
) -> object:
    from app.geometry import PoseObservation

    x1, y1, x2, y2 = bbox_norm
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    kps: list[tuple[float, float, float]] = [(cx, cy, 0.90) for _ in range(17)]
    if horizontal:
        kps[0] = (x1 + 0.03, cy, 0.92)
        kps[1] = (x1 + 0.04, cy - 0.01, 0.90)
        kps[2] = (x1 + 0.04, cy + 0.01, 0.90)
        kps[5] = (x1 + 0.08, cy - 0.04, 0.92)
        kps[6] = (x1 + 0.08, cy + 0.04, 0.92)
        kps[11] = (x2 - 0.08, cy - 0.04, 0.90)
        kps[12] = (x2 - 0.08, cy + 0.04, 0.90)
        kps[15] = (x2 - 0.02, cy - 0.03, 0.88)
        kps[16] = (x2 - 0.02, cy + 0.03, 0.88)
    else:
        kps[0] = (cx, y1 + 0.04, 0.92)
        kps[1] = (cx - 0.01, y1 + 0.03, 0.90)
        kps[2] = (cx + 0.01, y1 + 0.03, 0.90)
        kps[5] = (cx - 0.04, y1 + 0.12, 0.92)
        kps[6] = (cx + 0.04, y1 + 0.12, 0.92)
        kps[11] = (cx - 0.04, y1 + 0.34, 0.90)
        kps[12] = (cx + 0.04, y1 + 0.34, 0.90)
        kps[15] = (cx - 0.04, y2 - 0.04, 0.88)
        kps[16] = (cx + 0.04, y2 - 0.04, 0.88)
    return PoseObservation(
        track_id=track_id,
        bbox_norm=bbox_norm,
        conf=0.86,
        keypoints_norm=tuple(kps),
    )


def test_assign_patient_and_assign_visitor_endpoints_with_auto_face_and_alarm_reset(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, engine, _ = isolated_api

    # 1. Несуществующий трек -> 404 на обоих эндпоинтах
    assert client.post("/api/assign-patient", json={"track_id": 9999}).status_code == 404
    assert client.post("/api/assign-visitor", json={"track_id": 9999}).status_code == 404

    # 2. Заводим зону кровати и два живых силуэта в кадре (без открытого лица — спиной к камере)
    zones = engine.zones.update(
        bed_polygon=[[0.30, 0.42], [0.68, 0.42], [0.68, 0.78], [0.30, 0.78]],
        door_polygon=[[0.05, 0.20], [0.22, 0.20], [0.24, 0.88], [0.05, 0.88]],
        ignore_polygons=[],
    )
    blank_frame = np.full((360, 640, 3), 80, dtype=np.uint8)
    obs_sleeper = _make_pose_obs(101, (0.36, 0.50, 0.62, 0.70), horizontal=True)
    obs_guest_t0 = _make_pose_obs(202, (0.14, 0.28, 0.28, 0.84), horizontal=False)
    obs_guest = _make_pose_obs(202, (0.22, 0.28, 0.36, 0.84), horizontal=False)
    engine.identifier.update(
        observations=[obs_sleeper, obs_guest_t0],  # type: ignore[list-item]
        zones=zones,
        now=99.5,
        frame_bgr=blank_frame,
    )
    obs_list = [obs_sleeper, obs_guest]  # type: ignore[list-item]

    identities, _ = engine.identifier.update(
        observations=obs_list,
        zones=zones,
        now=100.0,
        frame_bgr=blank_frame,
    )
    with engine._lock:
        engine._latest_raw_frame = blank_frame
        engine._latest_observations = list(obs_list)
        engine._latest_identities = dict(identities)
        engine._sync_tracks_summary_locked()

    # Заглушаем детектор лиц: сначала человек стоит спиной (лицо не найдено)
    monkeypatch.setattr(
        engine.faces,
        "enroll_from_frame",
        lambda role, frame, obs=None: (False, "На кадре не видно открытого лица"),
    )

    # 3. POST /api/assign-patient для трека 101 (без лица -> одежда запомнена, ok=True)
    r_pat = client.post("/api/assign-patient", json={"track_id": 101})
    assert r_pat.status_code == 200
    pat_json = r_pat.json()
    assert pat_json["ok"] is True
    assert pat_json["role"] == "patient"
    assert pat_json["auto_face_saved"] is False
    assert pat_json["message"] == "Закреплено: Бабушка (одежда запомнена)"

    st_after_pat = client.get("/api/status").json()
    tr_101 = next(t for t in st_after_pat["tracks"] if t["track_id"] == 101)
    assert tr_101["role"] == "patient"
    assert tr_101["reason_ru"] == "указана вручную"

    # 4. Имитируем ложную тревогу FSM от ошибочно помеченного гостя и снимаем её через POST /api/assign-visitor
    r_false_pat = client.post("/api/assign-patient", json={"track_id": 202})
    assert r_false_pat.status_code == 200
    engine.fsm._state = PatientState.SITTING_UP
    engine.fsm._fall_unacknowledged = True

    # Теперь лицо гостя видно в камеру -> проверяем одновременное автосохранение эталона лица
    captured_crops: list[tuple[int, int]] = []

    def _mock_face_found(role: object, frame: np.ndarray, obs: object = None) -> tuple[bool, str]:
        captured_crops.append((int(frame.shape[0]), int(frame.shape[1])))
        return True, f"cam_{getattr(role, 'value', role)}_202.jpg"

    monkeypatch.setattr(engine.faces, "enroll_from_frame", _mock_face_found)

    r_vis = client.post("/api/assign-visitor", json={"track_id": 202})
    assert r_vis.status_code == 200
    vis_json = r_vis.json()
    assert vis_json["ok"] is True
    assert vis_json["role"] == "visitor"
    assert vis_json["auto_face_saved"] is True
    assert vis_json["face_filename"] == "cam_visitor_202.jpg"
    assert vis_json["state"] == "idle"
    assert vis_json["message"] == "Закреплено: Свой (одежда и лицо сохранены)"
    assert engine.fsm.state == PatientState.IDLE
    assert engine.fsm.fall_unacknowledged is False
    # При двух людях в кадре в enroll_from_frame передан изолированный кроп вокруг трека 202, а не весь кадр 360x640
    assert len(captured_crops) == 1
    assert captured_crops[0] != (360, 640)

    st_after_vis = client.get("/api/status").json()
    tr_202 = next(t for t in st_after_vis["tracks"] if t["track_id"] == 202)
    assert tr_202["role"] == "visitor"
    assert tr_202["reason_ru"] == "указан вручную"


def test_settings_rtsp_hwaccel_and_door_visit_integration(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Проверка чтения RTSP_HWACCEL из .env
    for key in ("RTSP_HWACCEL", "RTSP_URL", "TG_BOT_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    env_nvdec = tmp_path / "nvdec.env"
    env_nvdec.write_text("RTSP_HWACCEL=NVDEC\n", encoding="utf-8")
    assert Settings.from_env(env_nvdec).rtsp_hwaccel == "nvdec"

    env_invalid = tmp_path / "invalid.env"
    env_invalid.write_text("RTSP_HWACCEL=weird_mode\n", encoding="utf-8")
    assert Settings.from_env(env_invalid).rtsp_hwaccel == "auto"

    # 2. Новые поля в GET /api/status и связка door_visit_active / clear_door_visit
    client, engine, _ = isolated_api
    st0 = client.get("/api/status").json()
    assert "decoder_backend" in st0
    assert isinstance(st0["decoder_backend"], str)
    assert st0["door_visit_active"] is False
    assert isinstance(st0["thumbs"], list)
    for sub in st0["telegram_subscribers"]:
        assert isinstance(sub.get("enabled"), bool)

    clear_calls: list[str] = []
    monkeypatch.setattr(engine.identifier, "door_visit_active", True, raising=False)
    monkeypatch.setattr(
        engine.identifier,
        "clear_door_visit",
        lambda: clear_calls.append("cleared"),
        raising=False,
    )

    st_visit = client.get("/api/status").json()
    assert st_visit["door_visit_active"] is True
    assert st_visit["caregiver_nearby"] is True

    # POST /api/resume должен сбрасывать активный визит через clear_door_visit()
    r_res = client.post("/api/resume")
    assert r_res.status_code == 200
    assert len(clear_calls) == 1

    # POST /api/assign-patient на реальном треке тоже вызывает clear_door_visit(), а 404 — не трогает
    assert client.post("/api/assign-patient", json={"track_id": 8888}).status_code == 404
    assert len(clear_calls) == 1

    zones = engine.zones.update(
        bed_polygon=[[0.25, 0.40], [0.70, 0.40], [0.70, 0.80], [0.25, 0.80]],
    )
    frame = np.full((360, 640, 3), 64, dtype=np.uint8)
    obs = _make_pose_obs(301, (0.32, 0.48, 0.62, 0.72), horizontal=True)
    identities, _ = engine.identifier.update(
        observations=[obs],  # type: ignore[list-item]
        zones=zones,
        now=200.0,
        frame_bgr=frame,
    )
    with engine._lock:
        engine._latest_raw_frame = frame
        engine._latest_observations = [obs]  # type: ignore[list-item]
        engine._latest_identities = dict(identities)
        engine._sync_tracks_summary_locked()
    monkeypatch.setattr(
        engine.faces,
        "enroll_from_frame",
        lambda role, frm, obs=None: (False, "Спиной к камере"),
    )
    assert client.post("/api/assign-patient", json={"track_id": 301}).status_code == 200
    assert len(clear_calls) == 2


def test_timeline_thumbnails_endpoints_and_path_traversal(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    client, engine, data_dir = isolated_api

    now_ts = time.time()
    # Пока кадров нет -> 404
    assert client.get(f"/api/timeline/thumb?ts={now_ts}").status_code == 404
    assert client.get("/api/timeline/thumb/nonexistent_thumb.jpg").status_code == 404

    # Записываем два чистых кадра хроники (первый при старте и второй при смене состояния)
    frame_a = np.zeros((360, 640, 3), dtype=np.uint8)
    frame_a[:, :] = (40, 90, 30)
    frame_b = np.zeros((360, 640, 3), dtype=np.uint8)
    frame_b[:, :] = (20, 40, 140)

    rec_a = engine.thumbs.maybe_record(frame_a, "in_bed", now_ts=now_ts - 120.0)
    rec_b = engine.thumbs.maybe_record(frame_b, "sitting_up", now_ts=now_ts - 50.0)
    assert rec_a is not None
    assert rec_b is not None

    # 1. GET /api/timeline возвращает индекс мини-снимков thumbs
    r_tl = client.get("/api/timeline?window_sec=21600")
    assert r_tl.status_code == 200
    thumbs = r_tl.json()["thumbs"]
    thumb_ids = [t["id"] for t in thumbs]
    assert rec_a["id"] in thumb_ids
    assert rec_b["id"] in thumb_ids

    # 2. GET /api/timeline/thumb/{id}.jpg -> JPEG 320x180 + Cache-Control: public, max-age=86400
    r_by_id = client.get(f"/api/timeline/thumb/{rec_a['id']}.jpg")
    assert r_by_id.status_code == 200
    assert r_by_id.headers["content-type"].startswith("image/jpeg")
    assert "max-age=86400" in r_by_id.headers.get("cache-control", "")
    decoded_img = cv2.imdecode(np.frombuffer(r_by_id.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded_img is not None
    assert decoded_img.shape[:2] == (360, 640)

    # 3. GET /api/timeline/thumb?ts=... -> ближайший кадр с заголовками X-Thumb-Ts и X-Thumb-State
    r_near = client.get(f"/api/timeline/thumb?ts={now_ts - 48.0}")
    assert r_near.status_code == 200
    assert r_near.headers["content-type"].startswith("image/jpeg")
    assert "no-cache" in r_near.headers.get("cache-control", "")
    assert r_near.headers.get("x-thumb-state") == "sitting_up"
    assert abs(float(r_near.headers["x-thumb-ts"]) - float(rec_b["ts"])) < 1.0

    # Слишком далёкий timestamp за пределами окна -> 404
    assert client.get(f"/api/timeline/thumb?ts={now_ts - 50000.0}").status_code == 404

    # 4. Защита от path traversal
    (data_dir / "secret.jpg").write_bytes(b"\xff\xd8\xff\xe0SECRET")
    for bad_url in (
        "/api/timeline/thumb/..%2Fsecret.jpg",
        "/api/timeline/thumb/..%5Csecret.jpg",
        "/api/timeline/thumb/%2e%2e.jpg",
    ):
        assert client.get(bad_url).status_code == 404


def test_telegram_subscriber_toggle_endpoint(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    client, engine, _ = isolated_api

    # Добавляем второго родственника через публичный метод register_subscriber
    engine.notifier.register_subscriber("200002", name="Анна", username="anna_care")

    # 1. Неизвестный chat_id -> 404
    assert client.post("/api/telegram/subscribers/999999/toggle").status_code == 404

    # 2. POST без тела инвертирует enabled (True -> False) только для 200002
    r_off = client.post("/api/telegram/subscribers/200002/toggle")
    assert r_off.status_code == 200
    off_body = r_off.json()
    assert off_body["ok"] is True
    assert off_body["enabled"] is False
    sub_202 = next(s for s in off_body["telegram_subscribers"] if s["chat_id"] == "200002")
    sub_101 = next(s for s in off_body["telegram_subscribers"] if s["chat_id"] == "100001")
    assert sub_202["enabled"] is False
    assert sub_101["enabled"] is True
    assert off_body["telegram_recipients_count"] == 1

    # 3. Повторный POST без тела включает обратно (False -> True)
    r_on = client.post("/api/telegram/subscribers/200002/toggle")
    assert r_on.status_code == 200
    assert r_on.json()["enabled"] is True
    assert r_on.json()["telegram_recipients_count"] == 2

    # 4. Явная передача {"enabled": false}
    r_explicit = client.post("/api/telegram/subscribers/200002/toggle", json={"enabled": False})
    assert r_explicit.status_code == 200
    assert r_explicit.json()["enabled"] is False
    assert r_explicit.json()["telegram_recipients_count"] == 1


def test_real_door_visit_e2e_status_and_resume_clear(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    """Сквозная проверка правила двери 1+3 без моков: вход через door_polygon -> door_visit_active -> /api/resume."""
    client, engine, _ = isolated_api

    zones = engine.zones.update(
        bed_polygon=[[0.32, 0.42], [0.72, 0.42], [0.72, 0.80], [0.32, 0.80]],
        door_polygon=[[0.04, 0.18], [0.22, 0.18], [0.22, 0.90], [0.04, 0.90]],
        ignore_polygons=[],
    )
    frame = np.full((360, 640, 3), 72, dtype=np.uint8)
    obs_sleeper = _make_pose_obs(11, (0.38, 0.50, 0.66, 0.70), horizontal=True)
    obs_door_guest_t0 = _make_pose_obs(22, (0.06, 0.26, 0.18, 0.86), horizontal=False)
    obs_door_guest = _make_pose_obs(22, (0.16, 0.26, 0.28, 0.86), horizontal=False)

    engine.identifier.update(
        observations=[obs_sleeper, obs_door_guest_t0],  # type: ignore[list-item]
        zones=zones,
        now=499.5,
        frame_bgr=frame,
    )
    identities, cg = engine.identifier.update(
        observations=[obs_sleeper, obs_door_guest],  # type: ignore[list-item]
        zones=zones,
        now=500.0,
        frame_bgr=frame,
    )
    assert engine.identifier.door_visit_active is True
    assert cg is True
    assert identities[22].role.value == "visitor"
    assert identities[22].reason == "door_entry"

    with engine._lock:
        engine._latest_raw_frame = frame
        engine._latest_observations = [obs_sleeper, obs_door_guest]  # type: ignore[list-item]
        engine._latest_identities = dict(identities)
        engine._caregiver_nearby = bool(cg or engine.identifier.door_visit_active)
        engine._sync_tracks_summary_locked()

    st = client.get("/api/status").json()
    assert st["door_visit_active"] is True
    assert st["caregiver_nearby"] is True
    tr_guest = next(t for t in st["tracks"] if t["track_id"] == 22)
    assert tr_guest["role"] == "visitor"
    assert tr_guest["reason_ru"] == "вошёл через дверь"

    # Ручное возобновление охраны мгновенно сбрасывает door_visit_active
    assert client.post("/api/resume").status_code == 200
    assert engine.identifier.door_visit_active is False
    assert client.get("/api/status").json()["door_visit_active"] is False


def test_filter_ignored_observations_preserves_door_crossing_on_real_zones() -> None:
    """
    Регрессия боевого кейса #74: соседняя стеновая ignore_polygons[1] вплотную граничит с door_polygon;
    входящий/выходящий в дверях человек (центр/плечи/таз в двери или перекрытие >= 20%) НЕ должен отбрасываться,
    тогда как предмет внутри ignore_polygons вне двери по-прежнему отсекается.
    """
    from app.engine import _filter_ignored_observations
    from app.zones import ZonesConfig

    zones = ZonesConfig(
        bed_polygon=[
            [0.31, 0.41],
            [0.67, 0.42],
            [0.69, 0.58],
            [0.60, 0.65],
            [0.60, 0.79],
            [0.28, 0.79],
            [0.26, 0.58],
        ],
        door_polygon=[
            [0.07, 0.27],
            [0.23, 0.17],
            [0.28, 0.81],
            [0.18, 0.94],
        ],
        ignore_polygons=[
            [
                [0.70052, 0.41627],
                [0.30959, 0.4067],
                [0.24567, 0.35885],
                [0.22818, 0.16627],
                [0.06871, 0.26555],
                [0.1831, 0.9988],
                [0.00076, 0.9988],
                [0.00076, 0.00239],
                [0.71398, 0.0],
            ]
        ],
    )

    # 1. Человек на левой кромке дверного проёма: его левая половина и центр тела попадают в ignore_polygons,
    #    но bbox существенно перекрывает door_polygon (>= 20%) -> обязан сохраниться!
    obs_entering_door = _make_pose_obs(701, (0.04, 0.28, 0.20, 0.86), horizontal=False)
    # 2. Человек с плечами/центром в door_polygon -> обязан сохраниться!
    obs_in_door = _make_pose_obs(702, (0.10, 0.26, 0.24, 0.86), horizontal=False)
    # 3. Предмет на стене сверху внутри ignore_polygons вне двери (overlap с дверью == 0) -> обязан отсечься!
    obs_wall_artifact = _make_pose_obs(703, (0.36, 0.06, 0.50, 0.34), horizontal=False)

    kept = _filter_ignored_observations(
        [obs_entering_door, obs_in_door, obs_wall_artifact],  # type: ignore[list-item]
        zones,
    )
    kept_ids = [o.track_id for o in kept]
    assert 701 in kept_ids
    assert 702 in kept_ids
    assert 703 not in kept_ids


def _build_fmp4_sample(width: int = 320, height: int = 180) -> bytes:
    import shutil
    import subprocess
    import sys
    from app.capture import CREATE_NO_WINDOW

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        venv_ffmpeg = Path(sys.executable).resolve().parent / (
            "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        )
        if venv_ffmpeg.is_file():
            ffmpeg_bin = str(venv_ffmpeg)
    assert ffmpeg_bin is not None, "Для генерации тестового fMP4 требуется ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate=10",
        "-frames:v",
        "12",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "6",
        "-keyint_min",
        "6",
        "-sc_threshold",
        "0",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
        "-frag_duration",
        "200000",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        timeout=15.0,
        check=True,
        creationflags=CREATE_NO_WINDOW,
    )
    assert len(proc.stdout) > 512
    return proc.stdout


def test_wave6_zones_homography_status_and_clean_mjpeg(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
    tmp_path: Path,
) -> None:
    from app.detector import render_annotated_frame
    from app.zones import ZoneStore

    client, engine, _ = isolated_api

    # 1. Безопасная миграция старого zones.json без потери существующих полигонов
    legacy_path = tmp_path / "legacy_zones.json"
    legacy_bed = [[0.31, 0.41], [0.67, 0.42], [0.60, 0.79], [0.28, 0.79]]
    legacy_door = [[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]]
    legacy_ignore = [[[0.01, 0.01], [0.10, 0.01], [0.10, 0.10], [0.01, 0.10]]]
    legacy_path.write_text(
        json.dumps(
            {
                "bed_polygon": legacy_bed,
                "door_polygon": legacy_door,
                "ignore_polygons": legacy_ignore,
            }
        ),
        encoding="utf-8",
    )
    migrated = ZoneStore(legacy_path).get()
    assert migrated.bed_polygon == legacy_bed
    assert migrated.door_polygon == legacy_door
    assert migrated.ignore_polygons == legacy_ignore
    assert len(migrated.floor_polygon) == 4
    assert migrated.mattress_height_cm == 55.0

    # 2. Валидация POST /api/zones: 3 точки пола -> 422, 4 точки -> 200 и homography_calibrated=True
    r_bad = client.post(
        "/api/zones",
        json={"floor_polygon": [[0.2, 0.8], [0.6, 0.8], [0.6, 0.98]]},
    )
    assert r_bad.status_code == 422

    floor_quad = [[0.28, 0.79], [0.60, 0.79], [0.64, 0.98], [0.22, 0.98]]
    r_ok = client.post(
        "/api/zones",
        json={
            "bed_polygon": legacy_bed,
            "door_polygon": legacy_door,
            "floor_polygon": floor_quad,
            "floor_width_cm": 210.0,
            "floor_depth_cm": 130.0,
            "mattress_height_cm": 58.0,
        },
    )
    assert r_ok.status_code == 200
    z_resp = r_ok.json()
    assert z_resp["ok"] is True
    assert z_resp["homography_calibrated"] is True
    assert z_resp["zones"]["floor_polygon"] == floor_quad
    assert z_resp["zones"]["floor_width_cm"] == 210.0
    assert z_resp["zones"]["floor_depth_cm"] == 130.0
    assert z_resp["zones"]["mattress_height_cm"] == 58.0

    # 3. Проверка расчёта 2.5D метрик (height_cm, floor_xy_cm) и ИК-режима в /api/status
    zones_cur = engine.zones.get()
    obs_bed = _make_pose_obs(10, (0.35, 0.48, 0.60, 0.68), horizontal=True)
    obs_standing = _make_pose_obs(20, (0.38, 0.32, 0.52, 0.94), horizontal=False)
    obs_fallen_floor = _make_pose_obs(30, (0.30, 0.83, 0.58, 0.95), horizontal=True)

    # Дневной цветной кадр (высокая насыщенность S) -> is_night_ir == False
    rgb_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    rgb_frame[:, :] = (30, 140, 210)
    identities, _ = engine.identifier.update(
        observations=[obs_bed, obs_standing, obs_fallen_floor],  # type: ignore[list-item]
        zones=zones_cur,
        now=300.0,
        frame_bgr=rgb_frame,
    )
    annotated = render_annotated_frame(
        frame_bgr=rgb_frame,
        observations=[obs_bed, obs_standing, obs_fallen_floor],  # type: ignore[list-item]
        identities=identities,
        zones=zones_cur,
        state=engine.fsm.state,
    )
    ok_ann, enc_ann = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    assert ok_ann

    with engine._lock:
        engine._latest_raw_frame = rgb_frame
        engine._latest_jpeg = enc_ann.tobytes()
        engine._latest_observations = [obs_bed, obs_standing, obs_fallen_floor]  # type: ignore[list-item]
        engine._latest_identities = dict(identities)
        engine._sync_tracks_summary_locked()

    st_day = client.get("/api/status").json()
    assert st_day["homography_calibrated"] is True
    assert st_day["floor_zone_ready"] is True
    assert st_day["is_night_ir"] is False
    assert isinstance(st_day["anchor_day_ready"], bool)
    assert isinstance(st_day["anchor_night_ready"], bool)
    assert isinstance(st_day["encoder_backend"], str)
    assert st_day["stream_mode"] == "mjpeg"
    assert st_day["stream_clients"] == 0

    by_id = {t["track_id"]: t for t in st_day["tracks"]}
    # Спящий в кровати трек НЕ должен иметь ложного роста/проекции на пол
    assert by_id[10]["height_cm"] is None
    assert by_id[10]["floor_xy_cm"] is None
    # Вертикально стоящий на полу человек имеет реальный рост в см и координаты стоп на полу
    assert by_id[20]["height_cm"] is not None and by_id[20]["height_cm"] >= 95.0
    assert isinstance(by_id[20]["floor_xy_cm"], list) and len(by_id[20]["floor_xy_cm"]) == 2
    # Лежащий на полу человек имеет координаты точки на полу, но без стоячего роста
    assert by_id[30]["height_cm"] is None
    assert isinstance(by_id[30]["floor_xy_cm"], list) and len(by_id[30]["floor_xy_cm"]) == 2

    # Переключаем кадр на монохромный Ч/Б ИК (S == 0) -> is_night_ir == True
    ir_frame = np.full((360, 640, 3), 96, dtype=np.uint8)
    with engine._lock:
        engine._latest_raw_frame = ir_frame
    assert client.get("/api/status").json()["is_night_ir"] is True

    # 4. Проверка чистого кадра без серверного оверлея (?overlay=0 vs ?overlay=1)
    r_snap_clean = client.get("/api/snapshot.jpg?overlay=0")
    r_snap_hud = client.get("/api/snapshot.jpg?overlay=1")
    assert r_snap_clean.status_code == 200
    assert r_snap_hud.status_code == 200
    img_clean = cv2.imdecode(np.frombuffer(r_snap_clean.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_hud = cv2.imdecode(np.frombuffer(r_snap_hud.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img_clean is not None and img_hud is not None
    # Чистый кадр однородно серый (std < 2.0), а кадр с серверным HUD содержит цветные рамки и бейджи
    assert float(np.std(img_clean)) < 2.0
    assert float(np.std(img_hud)) > 5.0

    r_mjpeg_clean = client.get("/api/mjpeg?overlay=0&frames=1")
    assert r_mjpeg_clean.status_code == 200
    assert "multipart/x-mixed-replace" in r_mjpeg_clean.headers["content-type"]
    assert b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8" in r_mjpeg_clean.content


def test_wave6_ws_telemetry_12hz_and_collapsed_keypoints_sanitization(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    from app.geometry import PoseObservation

    client, engine, _ = isolated_api

    zones = engine.zones.update(
        bed_polygon=[[0.30, 0.40], [0.70, 0.40], [0.70, 0.78], [0.30, 0.78]],
        floor_polygon=[[0.28, 0.79], [0.60, 0.79], [0.64, 0.98], [0.22, 0.98]],
    )
    # Спящий под одеялом человек, у которого YOLO свалял нижние точки 11..16 в комок на подушке
    collapsed_kps: list[tuple[float, float, float]] = [(0.48, 0.48, 0.91) for _ in range(17)]
    collapsed_kps[5] = (0.45, 0.50, 0.88)
    collapsed_kps[6] = (0.51, 0.50, 0.88)
    collapsed_obs = PoseObservation(
        track_id=501,
        bbox_norm=(0.43, 0.44, 0.54, 0.55),
        conf=0.74,
        keypoints_norm=tuple(collapsed_kps),
    )

    frame = np.full((360, 640, 3), 85, dtype=np.uint8)
    identities, _ = engine.identifier.update(
        observations=[collapsed_obs],
        zones=zones,
        now=600.0,
        frame_bgr=frame,
    )
    with engine._lock:
        engine._latest_raw_frame = frame
        engine._latest_observations = [collapsed_obs]
        engine._latest_identities = dict(identities)
        engine._sync_tracks_summary_locked()

    with client.websocket_connect("/ws/telemetry") as ws:
        init_pkt = ws.receive_json()
        assert init_pkt["type"] == "telemetry"
        for req_key in (
            "ts",
            "state",
            "status_reason",
            "fall_unacknowledged",
            "pause_remaining_sec",
            "caregiver_nearby",
            "door_visit_active",
            "is_night_ir",
            "anchor_day_ready",
            "anchor_night_ready",
            "homography_calibrated",
            "infer_fps",
            "runtime",
            "decoder_backend",
            "encoder_backend",
            "stream_mode",
            "stream_clients",
            "resolution",
            "tracks",
        ):
            assert req_key in init_pkt, f"Отсутствует поле {req_key} в пакете /ws/telemetry"

        assert len(init_pkt["tracks"]) == 1
        tr = init_pkt["tracks"][0]
        assert tr["track_id"] == 501
        assert len(tr["keypoints"]) == 17
        # Смятые на подушке нижние точки 11..16 обязаны быть обнулены в [0.0, 0.0, 0.0]
        for low_idx in range(11, 17):
            assert tr["keypoints"][low_idx] == [0.0, 0.0, 0.0]
        assert tr["hide_skeleton"] is True
        assert tr["height_cm"] is None
        assert tr["floor_xy_cm"] is None

        # Действие в API мгновенно пушит свежий пакет телеметрии в открытый WebSocket
        r_pause = client.post("/api/pause", json={"seconds": 600})
        assert r_pause.status_code == 200
        pushed_pkt = ws.receive_json()
        assert pushed_pkt["type"] == "telemetry"
        assert pushed_pkt["state"] == "paused"
        assert pushed_pkt["pause_remaining_sec"] > 500.0

        # Проверяем живой пейсинг _run_loop на 12 Гц (монотонный дедлайн + честный межкадровый FPS)
        object.__setattr__(engine.settings, "rtsp_url", "rtsp://127.0.0.1:8554/mock")
        object.__setattr__(engine.settings, "infer_fps", 12.0)
        engine.pose.load = lambda: None  # type: ignore[method-assign]
        engine.pose.infer = lambda _frm, zones=None: [collapsed_obs]  # type: ignore[method-assign]
        engine.capture.get_latest = lambda: (frame, time.monotonic())  # type: ignore[method-assign]

        engine._stop_event.clear()
        import threading as _th

        worker = _th.Thread(target=engine._run_loop, daemon=True)
        worker.start()
        try:
            recv_times: list[float] = []
            last_pkt: dict = {}
            for _ in range(7):
                last_pkt = ws.receive_json()
                recv_times.append(time.monotonic())
            span = recv_times[-1] - recv_times[0]
            measured_hz = (len(recv_times) - 1) / max(1e-3, span)
            assert 10.5 <= measured_hz <= 14.0, f"Неверная частота /ws/telemetry: {measured_hz:.2f} Гц"
            assert 10.0 <= float(last_pkt["infer_fps"]) <= 14.0
        finally:
            engine._stop_event.set()
            worker.join(timeout=2.0)
            object.__setattr__(engine.settings, "rtsp_url", "")


def test_wave6_ws_stream_and_http_fmp4_passthrough_plus_fallback(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
    tmp_path: Path,
) -> None:
    client, engine, _ = isolated_api

    # 1. До появления RTSP-потока: /api/stream.mp4 -> 503, а /ws/stream мгновенно шлёт JSON fallback
    assert client.get("/api/stream.mp4").status_code == 503
    with client.websocket_connect("/ws/stream") as ws_err:
        err_msg = ws_err.receive_json()
        assert err_msg["type"] == "error"
        assert err_msg["fallback"] == "/api/mjpeg?overlay=0"

    # 2. Прогреваем H264StreamHub реальным fragmented MP4 потоком (High Profile 320x180)
    raw_fmp4 = _build_fmp4_sample(width=320, height=180)
    engine.stream_hub.feed_bytes(raw_fmp4)
    assert engine.stream_hub.is_ready is True
    assert client.get("/api/status").json()["stream_mode"] == "ws_fmp4"

    # 3. WS /ws/stream: JSON init -> бинарный ftyp+moov -> бинарный GOP moof+mdat -> декодирование кадра
    with client.websocket_connect("/ws/stream") as ws_ok:
        init_meta = ws_ok.receive_json()
        assert init_meta["type"] == "init"
        assert init_meta["codecs"].startswith("avc1.64")
        assert init_meta["mime"] == f'video/mp4; codecs="{init_meta["codecs"]}"'
        assert init_meta["width"] == 320
        assert init_meta["height"] == 180

        init_bytes = ws_ok.receive_bytes()
        assert init_bytes[:8].endswith(b"ftyp")
        assert b"moov" in init_bytes

        first_frag = ws_ok.receive_bytes()
        assert b"moof" in first_frag and b"mdat" in first_frag
        assert engine.stream_hub.stream_clients >= 1

        joined_mp4 = tmp_path / "ws_stream_probe.mp4"
        joined_mp4.write_bytes(init_bytes + first_frag)
        cap = cv2.VideoCapture(str(joined_mp4))
        try:
            assert cap.isOpened()
            ok, bgr = cap.read()
            assert ok and bgr is not None and bgr.shape == (180, 320, 3)
        finally:
            cap.release()

    # После закрытия WebSocket подписка гарантированно очищена
    for _ in range(20):
        if engine.stream_hub.stream_clients == 0:
            break
        time.sleep(0.02)
    assert engine.stream_hub.stream_clients == 0

    # 4. Прямой HTTP fMP4 поток GET /api/stream.mp4?max_fragments=1 (для iOS Safari native video)
    r_mp4 = client.get("/api/stream.mp4?max_fragments=1")
    assert r_mp4.status_code == 200
    assert r_mp4.headers["content-type"].startswith("video/mp4")
    assert r_mp4.headers.get("x-stream-codecs", "").startswith("avc1.")
    assert r_mp4.content[:8].endswith(b"ftyp")
    assert b"moov" in r_mp4.content and b"moof" in r_mp4.content and b"mdat" in r_mp4.content


def test_wave6_timeline_glitch_absorption_and_sanitize_endpoint(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
    tmp_path: Path,
) -> None:
    from app.timeline import TimelineStore

    client, engine, _ = isolated_api

    # 1. Онлайн-поглощение мимолётного всплеска caregiver (< 3.5 с) между спокойными состояниями
    t0 = time.time() - 300.0
    engine.timeline.record("in_bed", True, now_ts=t0)
    engine.timeline.record("in_bed", True, now_ts=t0 + 60.0)
    # Короткий 1.5-секундный дребезг caregiver -> должен слиться обратно в in_bed при возврате!
    engine.timeline.record("caregiver", True, now_ts=t0 + 61.0)
    engine.timeline.record("caregiver", True, now_ts=t0 + 62.2)
    engine.timeline.record("in_bed", True, now_ts=t0 + 62.5)
    engine.timeline.record("in_bed", True, now_ts=t0 + 120.0)
    # Настоящий 15-секундный визит родственника -> обязан сохраниться!
    engine.timeline.record("caregiver", True, now_ts=t0 + 121.0)
    engine.timeline.record("caregiver", True, now_ts=t0 + 136.0)
    engine.timeline.record("in_bed", True, now_ts=t0 + 137.0)
    engine.timeline.record("in_bed", True, now_ts=t0 + 160.0)

    summary = engine.timeline.summary_stats(window_sec=86400.0)
    assert summary["caregiver_visits_count"] == 1
    assert summary["caregiver_sec"] >= 14.0

    # 2. Автоматическая миграция старого timeline.json (schema_ver < 6) и ручной POST /api/timeline/sanitize
    old_tl_file = tmp_path / "old_timeline.json"
    old_tl_file.write_text(
        json.dumps(
            {
                "updated_ts": t0 + 200.0,
                "spans": [
                    {"start": t0, "end": t0 + 50.0, "state": "in_bed"},
                    {"start": t0 + 50.0, "end": t0 + 51.8, "state": "caregiver"},
                    {"start": t0 + 51.8, "end": t0 + 100.0, "state": "in_bed"},
                ],
            }
        ),
        encoding="utf-8",
    )
    migrated_tl = TimelineStore(old_tl_file)
    migrated_spans = migrated_tl.to_dict()["spans"]
    assert len(migrated_spans) == 1
    assert migrated_spans[0]["state"] == "in_bed"

    r_san = client.post("/api/timeline/sanitize", json={"min_caregiver_sec": 3.5})
    assert r_san.status_code == 200
    san_body = r_san.json()
    assert san_body["ok"] is True
    assert san_body["summary_24h"]["caregiver_visits_count"] == 1


def test_wave6_1_sleeping_skeleton_zero_jitter_and_instant_wake(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    from app.engine import _compute_roi_optical_motion, _prepare_optical_gray
    from app.geometry import PoseObservation
    from app.patient_id import Role, TrackIdentity

    client, engine, _ = isolated_api
    zones = engine.zones.update(
        bed_polygon=[[0.28, 0.38], [0.72, 0.38], [0.72, 0.78], [0.28, 0.78]],
        floor_polygon=[[0.26, 0.79], [0.64, 0.79], [0.68, 0.97], [0.21, 0.97]],
    )
    engine.fsm._state = PatientState.IN_BED

    rng = np.random.default_rng(20261112)
    base_ir = np.full((360, 640, 3), 72, dtype=np.uint8)
    # Рисуем неподвижный силуэт одеяла/подушки на матрасе
    cv2.rectangle(base_ir, (210, 165), (430, 260), (128, 128, 128), -1)

    base_kps: list[tuple[float, float, float]] = [
        (0.36, 0.50, 0.88),  # 0 nose
        (0.35, 0.49, 0.85),  # 1 l_eye
        (0.37, 0.49, 0.85),  # 2 r_eye
        (0.34, 0.50, 0.41),  # 3 l_ear (< 0.48 -> должно стать [0,0,0])
        (0.38, 0.50, 0.39),  # 4 r_ear (< 0.48 -> должно стать [0,0,0])
        (0.42, 0.48, 0.86),  # 5 l_shoulder
        (0.42, 0.56, 0.84),  # 6 r_shoulder
        (0.48, 0.47, 0.76),  # 7 l_elbow
        (0.48, 0.57, 0.33),  # 8 r_elbow (под одеялом < 0.48)
        (0.52, 0.49, 0.29),  # 9 l_wrist (под одеялом < 0.48)
        (0.52, 0.58, 0.27),  # 10 r_wrist (под одеялом < 0.48)
        (0.55, 0.50, 0.79),  # 11 l_hip
        (0.55, 0.56, 0.77),  # 12 r_hip
        (0.62, 0.51, 0.35),  # 13 l_knee (галлюцинация одеяла)
        (0.62, 0.56, 0.31),  # 14 r_knee (галлюцинация одеяла)
        (0.67, 0.52, 0.26),  # 15 l_ankle
        (0.67, 0.56, 0.24),  # 16 r_ankle
    ]
    ident = TrackIdentity(
        track_id=77,
        role=Role.PATIENT,
        reason="bed_sleep_lock",
        bed_accum_sec=45.0,
    )

    prev_opt: np.ndarray | None = None
    packets: list[dict] = []

    with client.websocket_connect("/ws/telemetry") as ws:
        _ = ws.receive_json()  # стартовый снапшот

        # 30 кадров неподвижного сна с ИК-зерном матрицы (sigma=2.5) и джиттером ключевых точек ±0.018
        for step_idx in range(30):
            noise = rng.normal(0.0, 2.5, size=base_ir.shape)
            noisy_ir = np.clip(base_ir.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            cur_opt = _prepare_optical_gray(noisy_ir)
            opt_ratio = _compute_roi_optical_motion(cur_opt, prev_opt, (0.32, 0.45, 0.69, 0.62))
            prev_opt = cur_opt

            jitter_box = tuple(
                float(v + rng.uniform(-0.014, 0.014))
                for v in (0.32, 0.45, 0.69, 0.62)
            )
            jitter_kps: list[tuple[float, float, float]] = []
            for k_idx, (bx, by, bc) in enumerate(base_kps):
                jx = float(bx + rng.uniform(-0.018, 0.018))
                jy = float(by + rng.uniform(-0.018, 0.018))
                # Точки одеяла мигают вокруг 0.22..0.52, но при оптическом покое не должны зажигаться
                if bc < 0.48:
                    jc = float(rng.uniform(0.22, 0.52)) if step_idx > 0 else bc
                else:
                    jc = float(np.clip(bc + rng.uniform(-0.04, 0.04), 0.50, 0.96))
                jitter_kps.append((jx, jy, jc))

            obs = PoseObservation(
                track_id=77,
                bbox_norm=jitter_box,  # type: ignore[arg-type]
                conf=0.82,
                keypoints_norm=tuple(jitter_kps),
            )
            item = engine._build_track_ui_item(
                obs,
                ident,
                zones,
                PatientState.IN_BED,
                optical_motion_ratio=opt_ratio,
                now_mono=1000.0 + step_idx * 0.0833,
            )
            with engine._lock:
                engine._latest_raw_frame = noisy_ir
                engine._active_tracks_summary = [item]
            engine.publish_telemetry()
            packets.append(ws.receive_json())

        ref_track = packets[0]["tracks"][0]
        assert ref_track["is_still"] is True
        # Все шумовые точки с исходным conf < 0.48 обязаны быть строго [0.0, 0.0, 0.0]
        for blanket_idx in (3, 4, 8, 9, 10, 13, 14, 15, 16):
            assert ref_track["keypoints"][blanket_idx] == [0.0, 0.0, 0.0]

        # Все 30 пакетов при сне несут ПОБИТОВО ИДЕНТИЧНЫЕ bbox и keypoints (0.00 px RMS jitter, 0 миганий)
        for pkt in packets[1:]:
            tr_cur = pkt["tracks"][0]
            assert tr_cur["is_still"] is True
            assert tr_cur["bbox"] == ref_track["bbox"]
            assert tr_cur["keypoints"] == ref_track["keypoints"]
            assert tr_cur["hide_skeleton"] == ref_track["hide_skeleton"]

        # Кадр 31: реальное пробуждение / взмах руки вверх (меняется силуэт в кадре + запястье вверх с conf=0.88)
        wake_ir = base_ir.copy()
        cv2.rectangle(wake_ir, (240, 80), (350, 210), (215, 215, 215), -1)
        cur_opt_wake = _prepare_optical_gray(wake_ir)
        wake_opt_ratio = _compute_roi_optical_motion(cur_opt_wake, prev_opt, (0.32, 0.24, 0.62, 0.62))
        assert wake_opt_ratio is not None and wake_opt_ratio >= 0.012

        wake_kps = list(base_kps)
        wake_kps[0] = (0.41, 0.30, 0.91)
        wake_kps[5] = (0.42, 0.36, 0.89)
        wake_kps[7] = (0.42, 0.25, 0.87)
        wake_kps[9] = (0.42, 0.16, 0.88)  # поднятая вверх рука
        wake_obs = PoseObservation(
            track_id=77,
            bbox_norm=(0.32, 0.15, 0.62, 0.62),
            conf=0.89,
            keypoints_norm=tuple(wake_kps),
        )
        wake_item = engine._build_track_ui_item(
            wake_obs,
            ident,
            zones,
            PatientState.HELP_CALL,
            optical_motion_ratio=wake_opt_ratio,
            now_mono=1003.0,
        )
        with engine._lock:
            engine.fsm._state = PatientState.HELP_CALL
            engine._latest_raw_frame = wake_ir
            engine._active_tracks_summary = [wake_item]
        engine.publish_telemetry()
        wake_pkt = ws.receive_json()
        wake_tr = wake_pkt["tracks"][0]
        assert wake_tr["is_still"] is False
        assert wake_tr["keypoints"][9][1] == pytest.approx(0.16, abs=1e-4)
        assert wake_tr["keypoints"][9][2] >= 0.85
        assert wake_tr["bbox"] != ref_track["bbox"]


def test_wave6_1_coasting_suppresses_blanket_dropout_strobe(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    """
    Регрессия строб-дефекта #108: когда бабушка спит под одеялом (IN_BED) и YOLO
    промахивается на сериях кадров ("1x5", "0x28", "1x1"), _build_tracks_ui_for_frame
    удерживает замороженный трек при optical_motion < 0.012 с 0 пустых пакетов,
    и мгновенно снимает удержание при выходе из IN_BED или реальном оптическом движении.
    """
    from app.engine import _prepare_optical_gray
    from app.geometry import PoseObservation
    from app.patient_id import Role, TrackIdentity

    client, engine, _ = isolated_api
    zones = engine.zones.update(
        bed_polygon=[[0.28, 0.38], [0.72, 0.38], [0.72, 0.78], [0.28, 0.78]],
        floor_polygon=[[0.26, 0.79], [0.64, 0.79], [0.68, 0.97], [0.21, 0.97]],
    )
    engine.fsm._state = PatientState.IN_BED

    rng = np.random.default_rng(108)
    base_ir = np.full((360, 640, 3), 70, dtype=np.uint8)
    cv2.rectangle(base_ir, (205, 165), (435, 260), (132, 132, 132), -1)

    kps = [
        (0.36, 0.50, 0.88),
        (0.35, 0.49, 0.84),
        (0.37, 0.49, 0.84),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.42, 0.48, 0.86),
        (0.42, 0.56, 0.84),
        (0.48, 0.47, 0.76),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.55, 0.50, 0.79),
        (0.55, 0.56, 0.77),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]

    prev_opt: np.ndarray | None = None
    # Имитируем реальный RLE провалов под одеялом: 5 детекций, 28 пропусков, 1 детекция со сменой ID (9001), 17 пропусков
    schedule_present = [True] * 5 + [False] * 28 + [True] + [False] * 17
    packets: list[dict] = []

    with client.websocket_connect("/ws/telemetry") as ws:
        _ = ws.receive_json()

        for i, detected in enumerate(schedule_present):
            noisy_ir = np.clip(
                base_ir.astype(np.float32) + rng.normal(0.0, 2.5, size=base_ir.shape),
                0,
                255,
            ).astype(np.uint8)
            cur_opt = _prepare_optical_gray(noisy_ir)
            tid = 9001 if i == 33 else 1
            live_obs = (
                [
                    PoseObservation(
                        track_id=tid,
                        bbox_norm=(
                            0.32 + float(rng.uniform(-0.01, 0.01)),
                            0.45 + float(rng.uniform(-0.01, 0.01)),
                            0.69 + float(rng.uniform(-0.01, 0.01)),
                            0.62 + float(rng.uniform(-0.01, 0.01)),
                        ),
                        conf=0.79,
                        keypoints_norm=tuple(kps),
                        is_still=True,
                        _kp_stabilized=True,
                    )
                ]
                if detected
                else []
            )
            idents = (
                {
                    tid: TrackIdentity(
                        track_id=tid,
                        role=Role.PATIENT,
                        reason="bed_sleep_lock",
                        bed_accum_sec=30.0 + i * 0.083,
                    )
                }
                if detected
                else {}
            )

            tracks_ui = engine._build_tracks_ui_for_frame(
                live_observations=live_obs,
                identities=idents,
                zones=zones,
                state_now=PatientState.IN_BED,
                cur_opt_gray=cur_opt,
                prev_opt_gray=prev_opt,
                now_mono=2000.0 + i * 0.0833,
            )
            prev_opt = cur_opt

            with engine._lock:
                engine._latest_raw_frame = noisy_ir
                engine._active_tracks_summary = tracks_ui
            engine.publish_telemetry()
            packets.append(ws.receive_json())

        # Ни одного пустого пакета и ни одного сдвига координат за все 51 тик!
        assert all(len(p["tracks"]) == 1 for p in packets)
        ref = packets[0]["tracks"][0]
        for p in packets[1:]:
            tr = p["tracks"][0]
            assert tr["is_still"] is True
            assert tr["bbox"] == ref["bbox"]
            assert tr["keypoints"] == ref["keypoints"]

        # Как только в зоне кровати возникает реальное оптическое движение при пустом live_obs — coast мгновенно снимается
        moved_ir = np.full((360, 640, 3), 70, dtype=np.uint8)  # силуэт исчез с матраса
        cur_opt_moved = _prepare_optical_gray(moved_ir)
        tracks_after_move = engine._build_tracks_ui_for_frame(
            live_observations=[],
            identities={},
            zones=zones,
            state_now=PatientState.IN_BED,
            cur_opt_gray=cur_opt_moved,
            prev_opt_gray=prev_opt,
            now_mono=2010.0,
        )
        assert tracks_after_move == []


def test_wave7_bed_persistence_and_clear_endpoint(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Волна 7: закон сохранения Бабушки в кровати в телеметрии и /api/status:
    - дефолт door_visit_idle_timeout_sec == 14400.0 (4 часа);
    - удержание скелета и при PAUSED / CAREGIVER_ASSIST, если одеяло неподвижно;
    - переход на карточку bed_covered_persistence при сдвиге одеяла (anchor_drift >= 0.050) и пустом YOLO;
    - ручной сброс через POST /api/bed_presence/clear.
    """
    from app.engine import _prepare_optical_gray, translate_role_reason
    from app.geometry import PoseObservation
    from app.patient_id import Role, TrackIdentity

    for key in ("DOOR_VISIT_IDLE_TIMEOUT_SECONDS", "DOOR_VISIT_IDLE_TIMEOUT_SEC"):
        monkeypatch.delenv(key, raising=False)
    empty_env = tmp_path / "empty_wave7.env"
    empty_env.write_text("", encoding="utf-8")
    assert Settings.from_env(empty_env).door_visit_idle_timeout_sec == 14400.0

    assert translate_role_reason("patient", "bed_covered_persistence") == "лежит в кровати · под одеялом"
    assert translate_role_reason("visitor", "door_visit_seated") == "в комнате · вошёл через дверь"
    assert translate_role_reason("visitor", "door_visit_present") == "в комнате · вошёл через дверь"

    client, engine, _ = isolated_api
    assert engine.identifier._door_visit_idle_timeout_sec == 14400.0  # type: ignore[attr-defined]

    zones = engine.zones.update(
        bed_polygon=[[0.28, 0.38], [0.72, 0.38], [0.72, 0.78], [0.28, 0.78]],
        door_polygon=[[0.07, 0.25], [0.25, 0.18], [0.26, 0.85], [0.08, 0.92]],
    )
    engine.fsm._state = PatientState.IN_BED

    base_ir = np.full((360, 640, 3), 68, dtype=np.uint8)
    cv2.rectangle(base_ir, (210, 170), (430, 258), (138, 138, 138), -1)
    opt_a = _prepare_optical_gray(base_ir)

    kps = [
        (0.36, 0.50, 0.88),
        (0.35, 0.49, 0.84),
        (0.37, 0.49, 0.84),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.42, 0.48, 0.86),
        (0.42, 0.56, 0.84),
        (0.48, 0.47, 0.76),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.55, 0.50, 0.79),
        (0.55, 0.56, 0.77),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    obs_sleep = PoseObservation(
        track_id=7,
        bbox_norm=(0.32, 0.45, 0.68, 0.63),
        conf=0.81,
        keypoints_norm=tuple(kps),
        is_still=True,
        _kp_stabilized=True,
    )
    ident_sleep = {
        7: TrackIdentity(
            track_id=7,
            role=Role.PATIENT,
            reason="bed_sleep_lock",
            bed_accum_sec=45.0,
        )
    }

    monkeypatch.setattr(engine.identifier, "patient_in_bed_latched", True, raising=False)
    monkeypatch.setattr(
        engine.identifier,
        "latched_patient_bed_bbox",
        (0.32, 0.45, 0.68, 0.63),
        raising=False,
    )
    monkeypatch.setattr(engine.identifier, "latched_patient_track_id", 7, raising=False)

    # 1. Фиксируем живой спальный кадр
    t0 = engine._build_tracks_ui_for_frame(
        live_observations=[obs_sleep],
        identities=ident_sleep,
        zones=zones,
        state_now=PatientState.IN_BED,
        cur_opt_gray=opt_a,
        prev_opt_gray=opt_a,
        now_mono=3000.0,
    )
    assert len(t0) == 1
    assert t0[0]["track_id"] == 7

    # 2. Даже в режиме PAUSED при неподвижном одеяле и пустом YOLO удерживается защёлкнутый скелет
    t_paused = engine._build_tracks_ui_for_frame(
        live_observations=[],
        identities={},
        zones=zones,
        state_now=PatientState.PAUSED,
        cur_opt_gray=opt_a,
        prev_opt_gray=opt_a,
        now_mono=3005.0,
    )
    assert len(t_paused) == 1
    assert t_paused[0]["role"] == "patient"
    assert t_paused[0]["is_still"] is True
    assert t_paused[0]["keypoints"] == t0[0]["keypoints"]

    # 3. Бабушка перевернулась и укрылась с головой (anchor_drift >= 0.050, YOLO = []):
    #    старый скелет сбрасывается, но вместо пустого списка выдаётся bed_covered_persistence!
    shifted_blanket = np.full((360, 640, 3), 68, dtype=np.uint8)
    cv2.rectangle(shifted_blanket, (235, 182), (445, 272), (162, 162, 162), -1)
    opt_b = _prepare_optical_gray(shifted_blanket)
    t_covered = engine._build_tracks_ui_for_frame(
        live_observations=[],
        identities={},
        zones=zones,
        state_now=PatientState.IN_BED,
        cur_opt_gray=opt_b,
        prev_opt_gray=opt_a,
        now_mono=3010.0,
    )
    assert len(t_covered) == 1
    assert t_covered[0]["track_id"] == 7
    assert t_covered[0]["role"] == "patient"
    assert t_covered[0]["reason"] == "bed_covered_persistence"
    assert t_covered[0]["reason_ru"] == "лежит в кровати · под одеялом"
    assert t_covered[0]["bbox"] == [0.32, 0.45, 0.68, 0.63]
    assert t_covered[0]["keypoints"] == []
    assert t_covered[0]["hide_skeleton"] is True
    assert t_covered[0]["is_still"] is True

    with engine._lock:
        engine._active_tracks_summary = t_covered

    with client.websocket_connect("/ws/telemetry") as ws:
        init_pkt = ws.receive_json()
        assert init_pkt["patient_in_bed_latched"] is True
        assert len(init_pkt["tracks"]) == 1
        assert init_pkt["tracks"][0]["reason"] == "bed_covered_persistence"

        st_cov = client.get("/api/status").json()
        assert st_cov["patient_in_bed_latched"] is True
        assert st_cov["tracks"][0]["reason"] == "bed_covered_persistence"

        # 4. Ручной сброс присутствия в кровати POST /api/bed_presence/clear
        clear_bed_calls: list[str] = []
        monkeypatch.setattr(
            engine.identifier,
            "clear_bed_presence",
            lambda: (
                clear_bed_calls.append("cleared"),
                setattr(engine.identifier, "patient_in_bed_latched", False),
            ),
            raising=False,
        )
        r_clear = client.post("/api/bed_presence/clear")
        assert r_clear.status_code == 200
        assert r_clear.json() == {"ok": True, "patient_in_bed_latched": False}
        assert len(clear_bed_calls) == 1

        pushed = ws.receive_json()
        assert pushed["patient_in_bed_latched"] is False
        assert pushed["tracks"] == []
        assert client.get("/api/status").json()["tracks"] == []


def test_wave7_visitor_seated_and_blind_zone_persistence_and_clear_endpoint(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Волна 7: постоянство Своих в «В кадре» и защита стула у двери:
    - стул у двери [0.1589, 0.6671, 0.3364, 0.997] не поглощается левой ignore_polygon;
    - при visitor_spot_occupied == True выдаётся door_visit_seated с bbox стула и virtual_presence == False;
    - при уходе в слепую зону (visitor_spot_occupied == False) выдаётся door_visit_present (track_id: -2, bbox: None, virtual_presence: True);
    - POST /api/door_visit/clear мгновенно снимает карточку гостя, сохраняя спящую Бабушку.
    """
    from app.engine import _filter_ignored_observations
    from app.zones import ZonesConfig

    # 1. Проверка защиты стула у двери в _filter_ignored_observations
    real_zones = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.60, 0.79], [0.28, 0.79]],
        door_polygon=[[0.077, 0.25], [0.256, 0.17], [0.256, 0.86], [0.077, 0.94]],
        ignore_polygons=[
            [
                [0.0, 0.0],
                [0.1831, 0.0],
                [0.1831, 0.9988],
                [0.0, 0.9988],
            ]
        ],
    )
    obs_chair = _make_pose_obs(808, (0.1589, 0.6671, 0.3364, 0.9970), horizontal=False)
    kept_chair = _filter_ignored_observations([obs_chair], real_zones)  # type: ignore[list-item]
    assert [o.track_id for o in kept_chair] == [808]

    # 2. Проверка door_visit_seated -> door_visit_present -> POST /api/door_visit/clear
    client, engine, _ = isolated_api
    engine.zones.update(
        bed_polygon=real_zones.bed_polygon,
        door_polygon=real_zones.door_polygon,
    )
    engine.fsm._state = PatientState.CAREGIVER_ASSIST

    monkeypatch.setattr(engine.identifier, "patient_in_bed_latched", True, raising=False)
    monkeypatch.setattr(
        engine.identifier,
        "latched_patient_bed_bbox",
        (0.32, 0.45, 0.66, 0.74),
        raising=False,
    )
    monkeypatch.setattr(engine.identifier, "latched_patient_track_id", 1, raising=False)
    monkeypatch.setattr(engine.identifier, "door_visit_active", True, raising=False)
    monkeypatch.setattr(engine.identifier, "visitor_spot_occupied", True, raising=False)
    monkeypatch.setattr(
        engine.identifier,
        "latched_visitor_bbox",
        (0.1589, 0.6671, 0.3364, 0.9970),
        raising=False,
    )

    st_seated = client.get("/api/status").json()
    assert st_seated["door_visit_active"] is True
    assert st_seated["patient_in_bed_latched"] is True
    assert len(st_seated["tracks"]) == 2
    pat_tr = next(t for t in st_seated["tracks"] if t["role"] == "patient")
    vis_tr = next(t for t in st_seated["tracks"] if t["role"] == "visitor")
    assert pat_tr["reason"] == "bed_covered_persistence"
    assert vis_tr["reason"] == "door_visit_seated"
    assert vis_tr["reason_ru"] == "в комнате · вошёл через дверь"
    assert vis_tr["bbox"] == [0.1589, 0.6671, 0.3364, 0.997]
    assert vis_tr["keypoints"] == []
    assert vis_tr["hide_skeleton"] is True
    assert vis_tr["is_still"] is True
    assert vis_tr["virtual_presence"] is False

    # Гость отошёл со стула в слепую зону комнаты
    monkeypatch.setattr(engine.identifier, "visitor_spot_occupied", False, raising=False)
    st_blind = client.get("/api/status").json()
    vis_blind = next(t for t in st_blind["tracks"] if t["role"] == "visitor")
    assert vis_blind["track_id"] == -2
    assert vis_blind["reason"] == "door_visit_present"
    assert vis_blind["reason_ru"] == "в комнате · вошёл через дверь"
    assert vis_blind["bbox"] is None
    assert vis_blind["virtual_presence"] is True

    # Ручной сброс визита через POST /api/door_visit/clear
    with client.websocket_connect("/ws/telemetry") as ws:
        init_pkt = ws.receive_json()
        assert any(t["reason"] == "door_visit_present" for t in init_pkt["tracks"])

        r_clear = client.post("/api/door_visit/clear")
        assert r_clear.status_code == 200
        assert r_clear.json() == {"ok": True, "door_visit_active": False}

        after_pkt = ws.receive_json()
        assert after_pkt["door_visit_active"] is False
        assert len(after_pkt["tracks"]) == 1
        assert after_pkt["tracks"][0]["role"] == "patient"
        assert after_pkt["tracks"][0]["reason"] == "bed_covered_persistence"


def test_wave8_false_alert_feedback_store_delayed_mp4_archive_survival_and_timeline(
    tmp_path: Path,
) -> None:
    """
    Волна 8: постоянное хранилище FalseAlertFeedbackStore (raw.jpg, annotated.jpg, context.json,
    доклейка clip.mp4 при раннем нажатии «Ты ошибся»), независимость от ротации/очистки ClipArchiveStore
    и санация ложного пика в TimelineStore.dismiss_false_alarm_around.
    """
    from app.archive import ClipArchiveStore
    from app.feedback import FalseAlertFeedbackStore
    from app.timeline import TimelineStore

    clips_dir = tmp_path / "clips"
    fb_dir = tmp_path / "feedback"
    archive = ClipArchiveStore(clips_dir, max_clips=2, max_total_mb=50.0)
    fb_store = FalseAlertFeedbackStore(fb_dir, max_recent=16, max_saved=20)

    ev = AlertEvent(
        state=PatientState.SITTING_UP,
        severity=AlertSeverity.WARNING,
        title="Бабушка садится в кровати",
        description="Ложный подъём на складку одеяла.",
        timestamp=420.0,
        patient_track_id=3,
    )
    raw_jpg = b"\xff\xd8\xff\xe0RAW_CLEAN_FRAME"
    ann_jpg = b"\xff\xd8\xff\xe0ANNOTATED_ALERT_FRAME"
    mp4_bytes = b"\x00\x00\x00\x18ftypmp42DELAYED_4SEC_CLIP"

    clip_rec = archive.save_event_snapshot(ev, ann_jpg, unix_ts=1_765_500_000.0)
    assert clip_rec["false_alarm"] is False
    assert clip_rec["false_alarm_by"] is None
    alert_id = str(clip_rec["id"])

    fb_store.register_recent_alert(
        alert_id=alert_id,
        event=ev,
        annotated_jpeg=ann_jpg,
        raw_jpeg=raw_jpg,
        context={
            "event_wall_ts": 1_765_500_000.0,
            "telemetry_snapshot": {"state": "sitting_up", "is_night_ir": True},
            "baseline_in_bed": {"calibrated": True, "samples": 24},
            "zones": {"bed_polygon": [[0.3, 0.4], [0.7, 0.4], [0.7, 0.8], [0.3, 0.8]]},
            "was_bed_latched": True,
        },
    )

    # 1. Пользователь жмёт «❌ Ты ошибся» на 2-й секунде (ДО готовности MP4!)
    marked = fb_store.mark_false_alert(
        alert_id=alert_id,
        reporter_chat_id="100500",
        reporter_name="Дмитрий",
    )
    assert marked["alert_id"] == alert_id
    assert marked["has_raw_jpg"] is True
    assert marked["has_annotated_jpg"] is True
    assert marked["has_mp4"] is False
    assert fb_store.stats() == {"count": 1, "with_mp4": 0}

    # Повторный клик идемпотентен и не плодит дубль в false_alerts.json
    marked_again = fb_store.mark_false_alert(
        alert_id=alert_id,
        reporter_chat_id="100500",
        reporter_name="Дмитрий",
    )
    assert marked_again["alert_id"] == alert_id
    assert fb_store.stats() == {"count": 1, "with_mp4": 0}

    # 2. Через 2 с готов 4-секундный хвост MP4 -> attach_mp4_if_flagged доклеивает clip.mp4 на диск!
    assert fb_store.attach_mp4_if_flagged(alert_id, mp4_bytes) is True
    assert fb_store.stats() == {"count": 1, "with_mp4": 1}

    pkg_dir = fb_dir / alert_id
    assert (pkg_dir / "raw.jpg").read_bytes() == raw_jpg
    assert (pkg_dir / "annotated.jpg").read_bytes() == ann_jpg
    assert (pkg_dir / "clip.mp4").read_bytes() == mp4_bytes
    ctx_json = json.loads((pkg_dir / "context.json").read_text(encoding="utf-8"))
    assert ctx_json["alert_id"] == alert_id
    assert ctx_json["reporter_chat_id"] == "100500"
    assert ctx_json["reporter_name"] == "Дмитрий"
    assert ctx_json["state"] == "sitting_up"
    assert ctx_json["patient_track_id"] == 3
    assert ctx_json["telemetry_snapshot"]["is_night_ir"] is True
    assert ctx_json["baseline_in_bed"]["calibrated"] is True
    assert ctx_json["has_raw_jpg"] is True
    assert ctx_json["has_annotated_jpg"] is True
    assert ctx_json["has_mp4"] is True

    # 3. Полная очистка / ротация ClipArchiveStore НЕ удаляет постоянный пакет разбора в data/feedback/!
    archive.mark_false_alarm(alert_id, "Дмитрий")
    reloaded_arch = ClipArchiveStore(clips_dir)
    assert reloaded_arch.get_clip(alert_id)["false_alarm"] is True  # type: ignore[index]
    assert reloaded_arch.get_clip(alert_id)["false_alarm_by"] == "Дмитрий"  # type: ignore[index]
    reloaded_arch.clear_all()
    assert reloaded_arch.list_clips() == []
    assert (pkg_dir / "raw.jpg").is_file()
    assert (pkg_dir / "clip.mp4").is_file()
    assert FalseAlertFeedbackStore(fb_dir).stats() == {"count": 1, "with_mp4": 1}

    # 4. Санация ложного пика в TimelineStore со склейкой соседних интервалов сна
    tl = TimelineStore(tmp_path / "timeline.json", window_sec=86400.0)
    t_base = time.time() - 120.0
    tl.record("in_bed", True, now_ts=t_base)
    tl.record("in_bed", True, now_ts=t_base + 40.0)
    tl.record("sitting_up", True, now_ts=t_base + 42.0)
    tl.record("sitting_up", True, now_ts=t_base + 50.0)
    tl.record("in_bed", True, now_ts=t_base + 52.0)
    tl.record("in_bed", True, now_ts=t_base + 90.0)
    assert tl.summary_stats(now_ts=t_base + 95.0)["sit_up_count"] == 1

    assert tl.dismiss_false_alarm_around(t_base + 45.0, fallback_state="in_bed") is True
    spans_after = tl.to_dict(now_ts=t_base + 95.0)["spans"]
    assert len(spans_after) == 1
    assert spans_after[0]["state"] == "in_bed"
    assert tl.summary_stats(now_ts=t_base + 95.0)["sit_up_count"] == 0


def test_wave8_engine_self_correction_and_feedback_rest_endpoints(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    """
    Волна 8: сквозная проверка GET /api/feedback, POST /api/feedback/{alert_id}/false_alarm
    и самокоррекции FallGuardEngine (перевод стоящего у кровати родственника в VISITOR,
    восстановление защёлки кровати после ложного left_bed, снятие тревоги без 15-минутной паузы).
    """
    client, engine, data_dir = isolated_api

    zones = engine.zones.update(
        bed_polygon=[[0.32, 0.42], [0.72, 0.42], [0.72, 0.80], [0.32, 0.80]],
        door_polygon=[[0.05, 0.20], [0.22, 0.20], [0.22, 0.90], [0.05, 0.90]],
        ignore_polygons=[],
    )
    frame = np.full((360, 640, 3), 84, dtype=np.uint8)
    # Человек входит/стоит вне кровати слева у двери (типичный кейс 23:42:47 — вошедшего родственника приняли за PATIENT)
    obs_visitor_t0 = _make_pose_obs(44, (0.12, 0.26, 0.26, 0.88), horizontal=False)
    obs_visitor_mistaken = _make_pose_obs(44, (0.20, 0.26, 0.31, 0.88), horizontal=False)
    engine.identifier.update(
        observations=[obs_visitor_t0],  # type: ignore[list-item]
        zones=zones,
        now=899.5,
        frame_bgr=frame,
    )
    identities_44, _ = engine.identifier.update(
        observations=[obs_visitor_mistaken],  # type: ignore[list-item]
        zones=zones,
        now=900.0,
        frame_bgr=frame,
    )
    assert 44 in identities_44
    assert engine.identifier.force_assign_patient(44) is True
    # Имитируем сброшенную защёлку кровати и активную тревогу LEFT_BED в FSM и хронике
    if hasattr(engine.identifier, "_patient_in_bed_latched"):
        engine.identifier._patient_in_bed_latched = False  # type: ignore[attr-defined]
    engine.fsm._state = PatientState.LEFT_BED
    engine.fsm._fall_unacknowledged = True

    with engine._lock:
        engine._latest_raw_frame = frame
        engine._latest_observations = [obs_visitor_mistaken]  # type: ignore[list-item]
        engine._latest_identities = dict(engine.identifier._tracks)
        engine._sync_tracks_summary_locked()

    now_wall = time.time()
    engine.timeline.record("in_bed", True, now_ts=now_wall - 40.0)
    engine.timeline.record("left_bed", True, now_ts=now_wall - 4.0)
    engine.timeline.record("left_bed", True, now_ts=now_wall - 1.0)

    ev = AlertEvent(
        state=PatientState.LEFT_BED,
        severity=AlertSeverity.WARNING,
        title="Встала с кровати одна",
        description="Ложное срабатывание на родственника у кровати.",
        timestamp=905.0,
        patient_track_id=44,
    )
    saved_clip = engine.archive.save_event_snapshot(
        ev,
        b"\xff\xd8\xff\xe0ALERT_JPG",
        unix_ts=now_wall - 2.0,
    )
    alert_id = str(saved_clip["id"])
    engine.archive.attach_mp4(alert_id, b"\x00\x00\x00\x18ftypmp42CLIP_MP4")
    engine.feedback.register_recent_alert(
        alert_id=alert_id,
        event=ev,
        annotated_jpeg=b"\xff\xd8\xff\xe0ALERT_JPG",
        raw_jpeg=b"\xff\xd8\xff\xe0RAW_JPG",
        context={
            "event_wall_ts": now_wall - 2.0,
            "telemetry_snapshot": engine.get_telemetry_snapshot(),
            "baseline_in_bed": dict(getattr(engine.fsm, "baseline_in_bed", {})),
            "zones": zones.to_dict(),
            "was_bed_latched": True,
        },
    )

    # Изначально список ошибок пуст
    r_fb0 = client.get("/api/feedback")
    assert r_fb0.status_code == 200
    assert r_fb0.json() == {"ok": True, "items": [], "stats": {"count": 0, "with_mp4": 0}}
    assert client.get("/api/status").json()["feedback_stats"] == {"count": 0, "with_mp4": 0}

    # Защита от невалидного alert_id
    assert client.post("/api/feedback/bad$id/false_alarm").status_code == 400

    # Отправляем отметку «Ты ошибся» через REST API
    r_mark = client.post(
        f"/api/feedback/{alert_id}/false_alarm",
        json={"reporter": "Веб-пульт"},
    )
    assert r_mark.status_code == 200
    mark_body = r_mark.json()
    assert mark_body["ok"] is True
    assert mark_body["alert_id"] == alert_id
    assert mark_body["reassigned_visitor_track_id"] == 44
    assert mark_body["caregiver_nearby"] is True
    assert mark_body["patient_in_bed_latched"] is True
    assert mark_body["timeline_cleaned"] is True
    assert mark_body["clip"]["false_alarm"] is True
    assert mark_body["clip"]["false_alarm_by"] == "Веб-пульт"
    # Тревога снята БЕЗ 15-минутной паузы!
    assert mark_body["state"] in ("in_bed", "idle")
    assert engine.fsm.state != PatientState.PAUSED
    assert engine.fsm.pause_remaining_sec(time.monotonic()) == 0.0
    assert engine.fsm.fall_unacknowledged is False
    assert engine.identifier._tracks[44].role.value == "visitor"

    # Проверяем пакет на диске и обновлённые /api/feedback, /api/clips, /api/status
    pkg_dir = data_dir / "feedback" / alert_id
    assert (pkg_dir / "raw.jpg").is_file()
    assert (pkg_dir / "annotated.jpg").is_file()
    assert (pkg_dir / "clip.mp4").is_file()
    assert (pkg_dir / "context.json").is_file()

    r_fb1 = client.get("/api/feedback?limit=10")
    assert r_fb1.status_code == 200
    fb1_body = r_fb1.json()
    assert fb1_body["stats"] == {"count": 1, "with_mp4": 1}
    assert fb1_body["items"][0]["alert_id"] == alert_id
    assert fb1_body["items"][0]["reporter_name"] == "Веб-пульт"
    assert fb1_body["items"][0]["has_mp4"] is True

    st_after = client.get("/api/status").json()
    assert st_after["feedback_stats"] == {"count": 1, "with_mp4": 1}
    assert st_after["recent_clips"][0]["false_alarm"] is True
    assert st_after["recent_clips"][0]["false_alarm_by"] == "Веб-пульт"


def _make_signed_init_data(
    bot_token: str,
    user_id: int | str,
    first_name: str = "Ирина",
    username: str = "irina_care",
    auth_date: int | None = None,
    start_param: str = "",
) -> str:
    import hashlib
    import hmac
    from urllib.parse import urlencode

    eff_ts = int(time.time()) if auth_date is None else int(auth_date)
    fields: dict[str, str] = {
        "auth_date": str(eff_ts),
        "query_id": "AAHdF6IQAAAAAN0XohDhrOrc",
        "user": json.dumps(
            {"id": int(user_id), "first_name": first_name, "username": username},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    if start_param:
        fields["start_param"] = start_param
    check_str = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    digest = hmac.new(secret_key, check_str.encode("utf-8"), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": digest})


def test_wave9_telegram_miniapp_auth_guard_http_and_websockets(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
    tmp_path: Path,
) -> None:
    """
    Волна 9: сквозная проверка локального бесшовного режима (LAN) и криптографической
    защиты внешнего домена https://nofall.example.com (HTTP REST, медиа и оба WebSocket).
    """
    from starlette.websockets import WebSocketDisconnect

    client, engine, _ = isolated_api
    bot_token = engine.settings.tg_bot_token

    # Проверяем чтение PUBLIC_WEBAPP_URL в Settings.from_env
    env_file = tmp_path / "custom_webapp.env"
    env_file.write_text(
        "RTSP_URL=rtsp://127.0.0.1/live\n"
        "TG_BOT_TOKEN=123456:ENV_TEST_TOKEN\n"
        "PUBLIC_WEBAPP_URL=https://nofall.example.com/\n",
        encoding="utf-8",
    )
    parsed_settings = Settings.from_env(dotenv_path=env_file)
    assert parsed_settings.public_webapp_url == "https://nofall.example.com"

    # Подкладываем тестовый JPEG-кадр и клип в архив для проверки медиа-маршрутов
    with engine._lock:
        engine._latest_jpeg = b"\xff\xd8\xff\xe0WAVE9_ANNOTATED_JPG\xff\xd9"
        engine._latest_clean_jpeg = b"\xff\xd8\xff\xe0WAVE9_CLEAN_JPG\xff\xd9"
    ev = AlertEvent(
        state=PatientState.SITTING_UP,
        severity=AlertSeverity.WARNING,
        title="Садится в кровати",
        description="Тест защищённой выдачи клипа.",
        timestamp=100.0,
        patient_track_id=1,
    )
    saved = engine.archive.save_event_snapshot(ev, b"\xff\xd8\xff\xe0CLIP_JPG")
    clip_id = str(saved["id"])
    engine.archive.attach_mp4(clip_id, b"\x00\x00\x00\x18ftypmp42WAVE9_CLIP")

    # Регистрируем второго родственника (200202) с выключенными пуш-уведомлениями (enabled=False)
    engine.notifier.register_subscriber("200202", name="Сергей", username="sergey_son")
    engine.notifier.set_subscriber_enabled("200202", False)

    # 1. Прямой LAN-доступ (testserver и 192.168.1.100:8090) — работает БЕЗ токена!
    for lan_headers in ({}, {"Host": "192.168.1.100:8090"}):
        r_sess_lan = client.get("/api/tg/session", headers=lan_headers)
        assert r_sess_lan.status_code == 200
        assert r_sess_lan.json()["authenticated"] is True
        assert r_sess_lan.json()["mode"] == "lan"
        assert client.get("/api/status", headers=lan_headers).status_code == 200
        assert client.get("/api/snapshot.jpg", headers=lan_headers).status_code == 200
        assert client.get("/api/clips", headers=lan_headers).status_code == 200

    with client.websocket_connect("/ws/telemetry") as ws_lan:
        snap_lan = ws_lan.receive_json()
        assert snap_lan["type"] == "telemetry"

    # 2. Локальная раздача Telegram WebApp SDK
    r_sdk = client.get("/telegram-web-app.js")
    assert r_sdk.status_code == 200
    assert "application/javascript" in r_sdk.headers["content-type"]
    assert len(r_sdk.content) > 10_000
    assert client.head("/").status_code == 200
    assert client.head("/telegram-web-app.js").status_code == 200

    # 3. Внешний доступ (nofall.example.com) БЕЗ токена: оболочка 200, данные 401, WS 1008!
    ext_headers = {"Host": "nofall.example.com", "X-Forwarded-Proto": "https"}
    for pub_path in (
        "/",
        "/manifest.webmanifest",
        "/icon.svg",
        "/sw.js",
        "/telegram-web-app.js",
        "/api/health",
    ):
        assert client.get(pub_path, headers=ext_headers).status_code == 200

    r_sess_ext_anon = client.get("/api/tg/session", headers=ext_headers)
    assert r_sess_ext_anon.status_code == 200
    sess_anon = r_sess_ext_anon.json()
    assert sess_anon["ok"] is True
    assert sess_anon["authenticated"] is False
    assert sess_anon["mode"] == "none"
    assert sess_anon["bot_username"] == "NoFallenAi"

    for protected_path in (
        "/api/status",
        "/api/snapshot.jpg",
        "/api/clips",
        f"/api/clips/{clip_id}.mp4",
        "/api/timeline",
    ):
        r_denied = client.get(protected_path, headers=ext_headers)
        assert r_denied.status_code == 401
        assert r_denied.json() == {
            "ok": False,
            "detail": "Требуется вход через Telegram Mini App",
        }

    assert (
        client.get(
            "/api/status",
            headers={"X-Forwarded-Host": "nofall.example.com"},
        ).status_code
        == 401
    )

    for ws_path in ("/ws/telemetry", "/ws/stream"):
        with pytest.raises(WebSocketDisconnect) as ws_exc:
            with client.websocket_connect(ws_path, headers={"Host": "nofall.example.com"}):
                pass
        assert ws_exc.value.code == 1008

    # 4. Проверка отказов POST /api/tg/auth (битый hash -> 401, чужой chat_id -> 403)
    bad_init = _make_signed_init_data("999999:OTHER_BOT_TOKEN", user_id=100001)
    r_bad = client.post("/api/tg/auth", headers=ext_headers, json={"init_data": bad_init})
    assert r_bad.status_code == 401

    expired_init = _make_signed_init_data(
        bot_token,
        user_id=100001,
        auth_date=int(time.time()) - 100_000,
    )
    assert (
        client.post("/api/tg/auth", headers=ext_headers, json={"init_data": expired_init}).status_code
        == 401
    )

    stranger_init = _make_signed_init_data(bot_token, user_id=777888, first_name="Чужой")
    r_stranger = client.post("/api/tg/auth", headers=ext_headers, json={"init_data": stranger_init})
    assert r_stranger.status_code == 403
    assert r_stranger.json()["detail"] == (
        "Аккаунт не подключён к боту @NoFallenAi. Отправьте /start боту."
    )

    # 5. Успешный вход подписчика 200202 (даже при enabled=False) и все 4 канала токена!
    valid_init = _make_signed_init_data(
        bot_token,
        user_id=200202,
        first_name="Сергей",
        username="sergey_son",
        start_param=f"clip_{clip_id}",
    )
    r_auth = client.post("/api/tg/auth", headers=ext_headers, json={"init_data": valid_init})
    assert r_auth.status_code == 200
    set_cookie_hdr = r_auth.headers.get("set-cookie", "")
    assert "fg_tg_session=" in set_cookie_hdr
    assert "HttpOnly" in set_cookie_hdr
    assert "Secure" in set_cookie_hdr
    assert "SameSite=none" in set_cookie_hdr

    auth_data = r_auth.json()
    assert auth_data["ok"] is True
    assert auth_data["authenticated"] is True
    assert auth_data["mode"] == "telegram"
    assert auth_data["start_param"] == f"clip_{clip_id}"
    assert auth_data["user"]["chat_id"] == "200202"
    session_token = auth_data["session_token"]
    assert len(session_token) > 40

    # (a) Authorization: Bearer <token>
    bearer_headers = {"Host": "nofall.example.com", "Authorization": f"Bearer {session_token}"}
    assert client.get("/api/status", headers=bearer_headers).status_code == 200
    assert client.get("/api/clips", headers=bearer_headers).status_code == 200

    # (b) Query ?token=<token> для медиа и обоих WebSocket
    r_snap_tok = client.get(
        f"/api/snapshot.jpg?token={session_token}",
        headers={"Host": "nofall.example.com"},
    )
    assert r_snap_tok.status_code == 200
    assert r_snap_tok.content.startswith(b"\xff\xd8")
    assert (
        client.get(
            f"/api/clips/{clip_id}.mp4?token={session_token}",
            headers={"Host": "nofall.example.com"},
        ).status_code
        == 200
    )

    with client.websocket_connect(
        f"/ws/telemetry?token={session_token}",
        headers={"Host": "nofall.example.com"},
    ) as ws_ext_tel:
        ext_tel_msg = ws_ext_tel.receive_json()
        assert ext_tel_msg["type"] == "telemetry"

    with client.websocket_connect(
        f"/ws/stream?token={session_token}",
        headers={"Host": "nofall.example.com"},
    ) as ws_ext_stream:
        # В тестовом окружении без живой RTSP-камеры хаб отдаёт штатный JSON фоллбэка ПОСЛЕ accept()
        stream_msg = ws_ext_stream.receive_json()
        assert stream_msg["type"] in ("init", "error")

    # (c) Cookie fg_tg_session
    cookie_headers = {
        "Host": "nofall.example.com",
        "Cookie": f"fg_tg_session={session_token}",
    }
    assert client.get("/api/status", headers=cookie_headers).status_code == 200
    r_sess_cookie = client.get("/api/tg/session", headers=cookie_headers)
    assert r_sess_cookie.status_code == 200
    assert r_sess_cookie.json()["authenticated"] is True
    assert r_sess_cookie.json()["mode"] == "telegram"
    assert r_sess_cookie.json()["session_token"]

    # (d) X-Telegram-Init-Data напрямую в заголовке
    init_headers = {
        "Host": "nofall.example.com",
        "X-Telegram-Init-Data": valid_init,
    }
    assert client.get("/api/status", headers=init_headers).status_code == 200

    # 6. Мгновенный отзыв доступа после удаления подписчика 200202!
    removed_ok, _ = engine.notifier.remove_subscriber("200202")
    assert removed_ok is True
    assert client.get("/api/status", headers=bearer_headers).status_code == 401
    assert client.get("/api/status", headers=cookie_headers).status_code == 401
    with pytest.raises(WebSocketDisconnect) as ws_revoked:
        with client.websocket_connect(
            f"/ws/telemetry?token={session_token}",
            headers={"Host": "nofall.example.com"},
        ):
            pass
    assert ws_revoked.value.code == 1008

    # 7. POST /api/tg/logout очищает cookie сессии
    r_logout = client.post("/api/tg/logout", headers=ext_headers)
    assert r_logout.status_code == 200
    assert r_logout.json() == {"ok": True, "authenticated": False}
    assert "fg_tg_session=" in r_logout.headers.get("set-cookie", "")


def test_wave12_ignore_track_and_zones_protect_bed_and_door_from_self_sabotage(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    """
    Волна 12 (§2.5): защита кровати и двери от самострела ignore_polygons:
    - POST /api/ignore-track возвращает 409 при клике на PATIENT, на трек поверх кровати (>= 12%)
      или поверх двери (>= 25%), не трогая зоны, защёлку кровати и якорь внешности;
    - POST /api/ignore-track возвращает 200 на статический фантом вне кровати и двери, сохраняя защёлку кровати;
    - POST /api/zones автоматически вычищает полигоны, накрывающие >= 50% кровати;
    - _filter_ignored_observations защищает человека на кровати даже при прямом конструировании ZonesConfig.
    """
    from app.engine import _filter_ignored_observations
    from app.geometry import PoseObservation
    from app.patient_id import Role, TrackIdentity
    from app.zones import ZonesConfig

    client, engine, _ = isolated_api

    bed_poly = [[0.296, 0.422], [0.642, 0.433], [0.600, 0.802], [0.273, 0.785]]
    door_poly = [[0.077, 0.250], [0.256, 0.170], [0.256, 0.860], [0.077, 0.940]]
    zones = engine.zones.update(
        bed_polygon=bed_poly,
        door_polygon=door_poly,
        ignore_polygons=[],
    )

    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    obs_patient_bed = _make_pose_obs(101, (0.33, 0.46, 0.61, 0.72), horizontal=True)
    # Точная геометрия фатального клика 13:55:20: сиделка у кровати (Role.VISITOR, покрывает ~79% кровати)
    obs_caregiver_over_bed = _make_pose_obs(
        102,
        (0.29177, 0.46397, 0.66743, 0.83524),
        horizontal=False,
    )
    obs_visitor_door = _make_pose_obs(103, (0.09, 0.26, 0.24, 0.84), horizontal=False)
    obs_corner_ghost = _make_pose_obs(104, (0.84, 0.10, 0.95, 0.34), horizontal=False)

    engine.identifier._patient_in_bed_latched = True  # type: ignore[attr-defined]
    engine.identifier._latched_patient_bed_bbox = (0.33, 0.46, 0.61, 0.72)  # type: ignore[attr-defined]
    engine.identifier._latched_patient_track_id = 101  # type: ignore[attr-defined]
    assert engine.identifier.anchor_day_ready is True

    with engine._lock:
        engine._latest_raw_frame = frame
        engine._latest_observations = [
            obs_patient_bed,
            obs_caregiver_over_bed,
            obs_visitor_door,
            obs_corner_ghost,
        ]  # type: ignore[list-item]
        engine._latest_identities = {
            101: TrackIdentity(track_id=101, role=Role.PATIENT, reason="bed_sleep_lock", bed_accum_sec=40.0),
            102: TrackIdentity(track_id=102, role=Role.VISITOR, reason="standing_near_bed"),
            103: TrackIdentity(track_id=103, role=Role.VISITOR, reason="door_entry"),
            104: TrackIdentity(track_id=104, role=Role.UNKNOWN, reason="static_artifact"),
        }

    # 1. Запрет игнор-зоны по PATIENT, поверх кровати и поверх двери -> HTTP 409
    for forbidden_tid in (101, 102, 103):
        resp_409 = client.post("/api/ignore-track", json={"track_id": forbidden_tid})
        assert resp_409.status_code == 409
        assert "Нельзя добавить игнор-зону поверх кровати или двери" in resp_409.json()["detail"]

    assert engine.zones.get().ignore_polygons == []
    assert engine.identifier.patient_in_bed_latched is True
    assert engine.identifier.anchor_day_ready is True

    # 2. Фантом в углу вне кровати и двери игнорируется штатно (HTTP 200), НЕ сбрасывая защёлку бабушки
    resp_ok = client.post("/api/ignore-track", json={"track_id": 104})
    assert resp_ok.status_code == 200
    assert len(engine.zones.get().ignore_polygons) == 1
    assert engine.identifier.patient_in_bed_latched is True

    # 3. POST /api/zones отфильтровывает полигон, накрывающий >= 50% кровати, сохраняя валидную штору
    curtain_poly = [[0.82, 0.05], [0.98, 0.05], [0.98, 0.45], [0.82, 0.45]]
    killer_bed_poly = [
        [0.29177, 0.46397],
        [0.66743, 0.46397],
        [0.66743, 0.83524],
        [0.29177, 0.83524],
    ]
    r_zones = client.post(
        "/api/zones",
        json={
            "bed_polygon": bed_poly,
            "door_polygon": door_poly,
            "ignore_polygons": [curtain_poly, killer_bed_poly],
        },
    )
    assert r_zones.status_code == 200
    saved_ignores = r_zones.json()["zones"]["ignore_polygons"]
    assert len(saved_ignores) == 1
    assert saved_ignores[0] == curtain_poly

    # 4. Самозащита _filter_ignored_observations, даже если ZonesConfig создан в обход ZoneStore.update
    corrupted_zones = ZonesConfig(
        bed_polygon=bed_poly,
        door_polygon=door_poly,
        ignore_polygons=[killer_bed_poly],
    )
    kps_sit = [(0.0, 0.0, 0.0)] * 17
    kps_sit[0] = (0.48, 0.49, 0.82)
    kps_sit[5] = (0.45, 0.54, 0.85)
    kps_sit[6] = (0.52, 0.54, 0.85)
    kps_sit[11] = (0.46, 0.66, 0.78)
    kps_sit[12] = (0.52, 0.66, 0.78)
    kps_sit[13] = (0.45, 0.71, 0.98)
    kps_sit[14] = (0.53, 0.71, 0.95)
    obs_grandma_1407 = PoseObservation(
        track_id=205,
        bbox_norm=(0.39, 0.45, 0.58, 0.76),
        conf=0.6197,
        keypoints_norm=tuple(kps_sit),
    )
    kept = _filter_ignored_observations([obs_grandma_1407], corrupted_zones)
    assert [o.track_id for o in kept] == [205]


def test_wave12_assign_visitor_restores_bed_latch_caregiver_idle_reset_and_skeleton_unfreeze(
    isolated_api: tuple[TestClient, FallGuardEngine, Path],
) -> None:
    """
    Волна 12 (§2.6): исправление каскада 14:02 -> 14:07 -> 14:09:
    - при ручном переводе ложного PATIENT вне кровати в VISITOR (POST /api/assign-visitor), когда
      кровать временно перекрыта (best_bed_tid is None), восстанавливается patient_in_bed_latched == True
      и карточка bed_covered_persistence;
    - сидящая поза с conf=0.6197 и коленями >= 0.48 не отбрасывается старым порогом 0.65;
    - в CAREGIVER_ASSIST у спящего без уверенного таза hide_skeleton == True, а при уходе перекрывающего
      VISITOR замороженный контур обновляется;
    - при пустом кадре без защёлки кровати и без сиделки FSM выходит из CAREGIVER_ASSIST в IDLE.
    """
    from app.geometry import PoseObservation
    from app.patient_id import Role, TrackIdentity

    client, engine, _ = isolated_api
    zones = engine.zones.update(
        bed_polygon=[[0.30, 0.42], [0.65, 0.42], [0.62, 0.80], [0.28, 0.80]],
        door_polygon=[[0.07, 0.24], [0.25, 0.18], [0.25, 0.86], [0.07, 0.92]],
    )
    frame = np.full((360, 640, 3), 80, dtype=np.uint8)

    # 1. Бабушка спала и надёжно зафиксирована в кровати
    obs_bed = _make_pose_obs(11, (0.34, 0.47, 0.60, 0.73), horizontal=True)
    for step_idx in range(12):
        engine.identifier.update(
            observations=[obs_bed],  # type: ignore[list-item]
            zones=zones,
            now=100.0 + step_idx * 1.0,
            frame_bgr=frame,
        )
    assert engine.identifier.patient_in_bed_latched is True

    # 2. Ложный трек вне кровати ошибочно получил PATIENT и сбросил patient_in_bed_latched, пока кровать перекрыта
    obs_false_t0 = _make_pose_obs(22, (0.12, 0.26, 0.26, 0.88), horizontal=False)
    obs_false_outside = _make_pose_obs(22, (0.20, 0.26, 0.31, 0.88), horizontal=False)
    engine.identifier.update(
        observations=[obs_false_t0],  # type: ignore[list-item]
        zones=zones,
        now=119.5,
        frame_bgr=frame,
    )
    engine.identifier.update(
        observations=[obs_false_outside],  # type: ignore[list-item]
        zones=zones,
        now=120.0,
        frame_bgr=frame,
    )
    assert engine.identifier.force_assign_patient(22, now=120.1) is True
    engine.identifier._patient_in_bed_latched = False  # type: ignore[attr-defined]
    engine.identifier._latched_patient_bed_bbox = None  # type: ignore[attr-defined]

    with engine._lock:
        engine._latest_raw_frame = frame
        engine._latest_observations = [obs_false_outside]  # type: ignore[list-item]
        engine._latest_identities = dict(engine.identifier._tracks)
        engine._sync_tracks_summary_locked()

    # 3. Оператор жмёт «Свой» (POST /api/assign-visitor) -> ложный трек становится VISITOR, а защёлка кровати возвращается!
    r_vis = client.post("/api/assign-visitor", json={"track_id": 22})
    assert r_vis.status_code == 200
    assert r_vis.json()["state"] == "idle"
    assert engine.identifier.patient_in_bed_latched is True
    assert engine.identifier.latched_patient_bed_bbox is not None

    # Когда гость ушёл в слепую зону (0 живых YOLO-треков), бабушка остаётся в статусе как bed_covered_persistence
    with engine._lock:
        engine._latest_observations = []
        engine._latest_identities = {}
        engine._sync_tracks_summary_locked()
    st_after = client.get("/api/status").json()
    assert st_after["patient_in_bed_latched"] is True
    assert any(t["reason"] == "bed_covered_persistence" for t in st_after["tracks"])

    # 4. Порог 0.48 для сидячей позы с коленями (14:07: conf=0.6197, колени 0.9887 / 0.9463):
    #    даже неподвижный вертикальный торс НЕ классифицируется как стоячий статический фантом!
    sit_kps = [(0.0, 0.0, 0.0)] * 17
    sit_kps[0] = (0.48, 0.46, 0.84)
    sit_kps[5] = (0.45, 0.52, 0.86)
    sit_kps[6] = (0.52, 0.52, 0.86)
    sit_kps[11] = (0.46, 0.65, 0.74)
    sit_kps[12] = (0.52, 0.65, 0.74)
    sit_kps[13] = (0.45, 0.72, 0.9887)
    sit_kps[14] = (0.53, 0.72, 0.9463)
    sit_obs = PoseObservation(
        track_id=33,
        bbox_norm=(0.40, 0.40, 0.57, 0.78),
        conf=0.6197,
        keypoints_norm=tuple(sit_kps),
    )
    sit_tr = TrackIdentity(track_id=33, role=Role.UNKNOWN, reason="candidate", has_moved=False)
    assert engine.identifier._is_standing_static_posture(sit_obs, sit_tr) is False  # type: ignore[attr-defined]

    # 5. hide_skeleton == True в CAREGIVER_ASSIST без уверенного таза + разморозка контура после ухода VISITOR
    no_hip_kps = [(0.0, 0.0, 0.0)] * 17
    no_hip_kps[0] = (0.46, 0.49, 0.85)
    no_hip_kps[5] = (0.43, 0.53, 0.83)
    no_hip_kps[6] = (0.50, 0.53, 0.83)
    obs_covered_occluded = PoseObservation(
        track_id=44,
        bbox_norm=(0.35, 0.46, 0.53, 0.66),
        conf=0.71,
        keypoints_norm=tuple(no_hip_kps),
        is_still=True,
    )
    obs_visitor_overlap = _make_pose_obs(45, (0.48, 0.30, 0.68, 0.88), horizontal=False)
    idents_pair = {
        44: TrackIdentity(track_id=44, role=Role.PATIENT, reason="bed_sleep_lock", bed_accum_sec=35.0),
        45: TrackIdentity(track_id=45, role=Role.VISITOR, reason="standing_near_bed"),
    }
    ui_occluded = engine._build_tracks_ui_for_frame(
        live_observations=[obs_covered_occluded, obs_visitor_overlap],  # type: ignore[list-item]
        identities=idents_pair,
        zones=zones,
        state_now=PatientState.CAREGIVER_ASSIST,
        cur_opt_gray=None,
        prev_opt_gray=None,
        now_mono=500.0,
    )
    pat_item_1 = next(t for t in ui_occluded if t["track_id"] == 44)
    assert pat_item_1["hide_skeleton"] is True
    assert pat_item_1["bbox"] == [0.35, 0.46, 0.53, 0.66]

    # Сиделка отошла, и реальная рамка спящей бабушки раскрылась шире -> frozen_bbox обязан обновиться!
    obs_covered_full = PoseObservation(
        track_id=44,
        bbox_norm=(0.35, 0.46, 0.61, 0.72),
        conf=0.76,
        keypoints_norm=tuple(no_hip_kps),
        is_still=True,
    )
    ui_unoccluded = engine._build_tracks_ui_for_frame(
        live_observations=[obs_covered_full],
        identities={44: idents_pair[44]},
        zones=zones,
        state_now=PatientState.CAREGIVER_ASSIST,
        cur_opt_gray=None,
        prev_opt_gray=None,
        now_mono=500.5,
    )
    assert ui_unoccluded[0]["bbox"] == [0.35, 0.46, 0.61, 0.72]
    assert ui_unoccluded[0]["hide_skeleton"] is True

    # 6. Выход FSM из зависшего CAREGIVER_ASSIST в IDLE при отсутствии пациента, сиделки и защёлки
    engine.fsm._state = PatientState.CAREGIVER_ASSIST
    engine.fsm._was_monitored_recently = False
    ev_idle = engine.fsm.step(
        observations=[],
        identities={},
        caregiver_nearby=False,
        zones=zones,
        now=600.0,
        patient_in_bed_latched=False,
    )
    assert ev_idle is None
    assert engine.fsm.state == PatientState.IDLE
    assert engine.fsm.status_reason == "ожидание в кадре"


def test_wave12_fsm_metric_floor_doll_hip_baseline_shield_and_bent_standing_130059(
    tmp_path: Path,
) -> None:
    """
    Волна 12 (§2.4): метрические инварианты FallStateMachine:
    - защита BedBaseline.hip_y и _hip_y_history от сжатого кукольного таза (L_torso_cm < 30.5) и mid_hip is None;
    - метрический фильтр _is_lying_on_floor отсекает кошку / диванную подушку / ходунки без анатомии,
      но пропускает реального взрослого человека на полу (включая позу калачиком вдоль оси камеры);
    - наклонённое стояние у кровати (кейс 130059: angle ~ 56°, длинные вертикальные ноги, hip_y <= 0.72,
      без стремительного падения) переходит в LEFT_BED, а НЕ в ложный FALLEN.
    """
    from app.geometry import PoseObservation
    from app.patient_id import Role, TrackIdentity
    from app.state_machine import FallStateMachine
    from app.zones import ZonesConfig

    fsm = FallStateMachine()
    zones = ZonesConfig(
        bed_polygon=[[0.30, 0.42], [0.66, 0.42], [0.62, 0.78], [0.28, 0.78]],
        door_polygon=[],
        floor_polygon=[[0.26, 0.79], [0.66, 0.79], [0.72, 0.98], [0.20, 0.98]],
    )
    ident_1 = {1: TrackIdentity(track_id=1, role=Role.PATIENT, reason="bed_sleep_lock", bed_accum_sec=30.0)}

    # 1. Сжатый «кукольный» таз на подушке (L_torso_cm ~ 8 см < 30.5 см) НЕ калибрует hip_y
    doll_kps = [(0.0, 0.0, 0.0)] * 17
    doll_kps[0] = (0.44, 0.50, 0.85)
    doll_kps[5] = (0.46, 0.51, 0.86)
    doll_kps[6] = (0.49, 0.53, 0.86)
    doll_kps[11] = (0.48, 0.52, 0.75)
    doll_kps[12] = (0.50, 0.54, 0.75)
    doll_obs = PoseObservation(
        track_id=1,
        bbox_norm=(0.40, 0.47, 0.62, 0.66),
        conf=0.78,
        keypoints_norm=tuple(doll_kps),
    )
    for idx in range(10):
        fsm.step(
            observations=[doll_obs],
            identities=ident_1,
            caregiver_nearby=False,
            zones=zones,
            now=10.0 + idx * 0.5,
        )
    assert fsm.baseline_in_bed["calibrated"] is True
    assert fsm._baseline.hip_calibrated is False
    assert fsm._baseline.torso_len_2d == 0.0

    # Сон под одеялом без таза (mid_hip is None) НЕ загрязняет _hip_y_history суррогатом body_center
    fsm._hip_y_history.clear()
    blanket_kps = [(0.0, 0.0, 0.0)] * 17
    blanket_kps[0] = (0.45, 0.50, 0.85)
    blanket_kps[5] = (0.46, 0.53, 0.82)
    blanket_kps[6] = (0.51, 0.53, 0.82)
    blanket_obs = PoseObservation(
        track_id=1,
        bbox_norm=(0.38, 0.46, 0.62, 0.68),
        conf=0.74,
        keypoints_norm=tuple(blanket_kps),
    )
    fsm.step(
        observations=[blanket_obs],
        identities=ident_1,
        caregiver_nearby=False,
        zones=zones,
        now=20.0,
    )
    assert len(fsm._hip_y_history) == 0

    # 2. Метрический и анатомический фильтр _is_lying_on_floor
    # (а) Кошка / мелкая подушка на ковре (примерно 34x22 см -> Area < 1650 см²) -> False
    cat_kps = [(0.0, 0.0, 0.0)] * 17
    cat_kps[0] = (0.44, 0.88, 0.65)
    cat_obs = PoseObservation(
        track_id=2,
        bbox_norm=(0.42, 0.85, 0.49, 0.91),
        conf=0.55,
        keypoints_norm=tuple(cat_kps),
    )
    assert fsm._is_lying_on_floor(cat_obs, zones) is False

    # (б) Крупный предмет / ходунки без уверенной головы и без связки торс+конечность -> False
    walker_obs = PoseObservation(
        track_id=3,
        bbox_norm=(0.32, 0.78, 0.58, 0.95),
        conf=0.58,
        keypoints_norm=tuple([(0.0, 0.0, 0.0)] * 17),
    )
    assert fsm._is_lying_on_floor(walker_obs, zones) is False

    # (в) Человек упал на ковёр калачиком вдоль оси камеры (AR ~ 0.76, но метрически >= 52x26 см + есть голова) -> True
    curled_kps = [(0.0, 0.0, 0.0)] * 17
    curled_kps[0] = (0.45, 0.83, 0.82)
    curled_kps[1] = (0.44, 0.82, 0.79)
    curled_kps[5] = (0.42, 0.85, 0.84)
    curled_kps[6] = (0.49, 0.85, 0.84)
    curled_kps[11] = (0.43, 0.92, 0.78)
    curled_kps[12] = (0.49, 0.92, 0.78)
    curled_obs = PoseObservation(
        track_id=4,
        bbox_norm=(0.36, 0.78, 0.54, 0.96),
        conf=0.81,
        keypoints_norm=tuple(curled_kps),
    )
    assert fsm._is_lying_on_floor(curled_obs, zones) is True

    # 3. Кейс 130059: бабушка стоит у кровати, наклонившись вперёд к матрасу (angle ~ 56°, длинные вертикальные ноги)
    fsm_stand = FallStateMachine()
    ident_10 = {10: TrackIdentity(track_id=10, role=Role.PATIENT, reason="bed_sleep_lock", bed_accum_sec=35.0)}
    normal_bed_obs = _make_pose_obs(10, (0.34, 0.46, 0.62, 0.72), horizontal=True)
    for idx in range(8):
        fsm_stand.step(
            observations=[normal_bed_obs],  # type: ignore[list-item]
            identities=ident_10,
            caregiver_nearby=False,
            zones=zones,
            now=100.0 + idx * 0.5,
        )
    assert fsm_stand.state == PatientState.IN_BED

    # Стоит на ногах перед кроватью, корпус наклонён (dx=0.105, dy=0.07 -> angle ~ 56.3°), лодыжки на полу y=0.88, таз y=0.61
    bent_kps = [(0.0, 0.0, 0.0)] * 17
    bent_kps[0] = (0.35, 0.50, 0.88)
    bent_kps[5] = (0.375, 0.54, 0.90)
    bent_kps[6] = (0.395, 0.54, 0.90)
    bent_kps[11] = (0.48, 0.61, 0.87)
    bent_kps[12] = (0.50, 0.61, 0.87)
    bent_kps[13] = (0.48, 0.75, 0.86)
    bent_kps[14] = (0.50, 0.75, 0.86)
    bent_kps[15] = (0.48, 0.89, 0.85)
    bent_kps[16] = (0.50, 0.89, 0.85)
    bent_stand_obs = PoseObservation(
        track_id=10,
        bbox_norm=(0.33, 0.45, 0.54, 0.92),
        conf=0.86,
        keypoints_norm=tuple(bent_kps),
    )

    emitted_states: list[PatientState] = []
    for step_i in range(10):
        ev = fsm_stand.step(
            observations=[bent_stand_obs],
            identities=ident_10,
            caregiver_nearby=False,
            zones=zones,
            now=110.0 + step_i * 0.4,
        )
        if ev is not None:
            emitted_states.append(ev.state)

    assert PatientState.FALLEN not in emitted_states
    assert fsm_stand.state == PatientState.LEFT_BED







