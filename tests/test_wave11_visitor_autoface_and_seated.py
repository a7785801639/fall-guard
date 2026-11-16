from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

from app.config import Settings
from app.engine import FallGuardEngine
from app.geometry import PoseObservation
from app.notifier import TelegramNotifier
from app.patient_id import FaceRegistry, PatientIdentifier, Role, TrackIdentity
from app.state_machine import AlertEvent, AlertSeverity, FallStateMachine, PatientState
from app.zones import ZonesConfig

BED_AND_DOOR_ZONES = ZonesConfig(
    bed_polygon=[
        [0.32, 0.38],
        [0.68, 0.38],
        [0.68, 0.74],
        [0.32, 0.74],
    ],
    door_polygon=[
        [0.06, 0.20],
        [0.20, 0.20],
        [0.20, 0.90],
        [0.06, 0.90],
    ],
    floor_polygon=[
        [0.22, 0.78],
        [0.74, 0.78],
        [0.80, 0.97],
        [0.18, 0.97],
    ],
    floor_width_cm=200.0,
    floor_depth_cm=120.0,
    mattress_height_cm=55.0,
)


def _unit_vec(seed: int, mix_with: np.ndarray | None = None, mix_alpha: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=128).astype(np.float32)
    v /= float(np.linalg.norm(v))
    if mix_with is not None and mix_alpha > 0.0:
        orth = v - float(np.dot(v, mix_with)) * mix_with
        orth /= float(np.linalg.norm(orth))
        c = float(mix_alpha)
        s = float(np.sqrt(max(0.0, 1.0 - c * c)))
        v = (c * mix_with + s * orth).astype(np.float32)
        v /= float(np.linalg.norm(v))
    return v


def _make_pose_full(
    track_id: int,
    bbox: tuple[float, float, float, float],
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    *,
    nose: tuple[float, float] | None = None,
    l_eye: tuple[float, float] | None = None,
    r_eye: tuple[float, float] | None = None,
    l_shoulder: tuple[float, float] | None = None,
    r_shoulder: tuple[float, float] | None = None,
    knee: tuple[float, float] | None = None,
    ankle: tuple[float, float] | None = None,
    face_conf: float = 0.92,
) -> PoseObservation:
    kps: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 17
    if nose is not None:
        kps[0] = (nose[0], nose[1], face_conf)
    if l_eye is not None:
        kps[1] = (l_eye[0], l_eye[1], face_conf)
    if r_eye is not None:
        kps[2] = (r_eye[0], r_eye[1], face_conf)
    ls = l_shoulder or (shoulder[0] + 0.03, shoulder[1])
    rs = r_shoulder or (shoulder[0] - 0.03, shoulder[1])
    kps[5] = (ls[0], ls[1], 0.95)
    kps[6] = (rs[0], rs[1], 0.95)
    kps[11] = (hip[0] + 0.025, hip[1], 0.95)
    kps[12] = (hip[0] - 0.025, hip[1], 0.95)
    if knee is not None:
        kps[13] = (knee[0] + 0.02, knee[1], 0.90)
        kps[14] = (knee[0] - 0.02, knee[1], 0.90)
    if ankle is not None:
        kps[15] = (ankle[0] + 0.02, ankle[1], 0.90)
        kps[16] = (ankle[0] - 0.02, ankle[1], 0.90)
    return PoseObservation(
        track_id=track_id,
        bbox_norm=bbox,
        conf=0.93,
        keypoints_norm=tuple(kps),
    )


def test_wave11_auto_enroll_visitor_from_false_alert_saves_portrait_and_telegram_caption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Сквозной тест Части 1:
    1) Нажатие «❌ Ты ошибся» (report_false_alert + Telegram wrong:) декодирует сырой raw.jpg пакета
       тревоги и реконструирует скелет виновного трека из telemetry_snapshot['tracks'].
    2) Портрет нового гостя сохраняется в data/faces/visitor/auto_visitor_*.jpg, эмбеддинг сразу активен.
    3) В подписи Telegram-поста появляется «· лицо «Своего» запомнено».
    4) Повторное нажатие на того же гостя возвращает status == 'already_known' без дублирования файлов.
    """
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    settings = Settings(
        rtsp_url="",
        tg_bot_token="123456:TEST_TOKEN_WAVE11",
        tg_chat_ids=("1001",),
        data_dir=data_dir,
        models_dir=models_dir,
        web_host="127.0.0.1",
        web_port=8090,
        infer_fps=6.0,
        capture_fps=10.0,
        pre_event_seconds=4.0,
        post_event_seconds=3.0,
        pose_model="yolo11n-pose.pt",
        device="cpu",
        pose_conf=0.35,
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=2.5,
        sit_confirm_seconds=1.5,
        leave_confirm_seconds=1.8,
        fall_confirm_seconds=1.5,
        missing_confirm_seconds=12.0,
        alert_cooldown_seconds=45.0,
        repeat_fall_alert_seconds=15.0,
        max_fall_repeats=10,
        suppress_sitting_when_caregiver=True,
        face_match_threshold=0.40,
    )
    engine = FallGuardEngine(settings)
    engine.zones.update(bed_polygon=BED_AND_DOOR_ZONES.bed_polygon, door_polygon=BED_AND_DOOR_ZONES.door_polygon)

    grandma_emb = _unit_vec(101)
    visitor_emb = _unit_vec(202)
    assert float(np.dot(grandma_emb, visitor_emb)) < 0.25

    monkeypatch.setattr(engine.faces, "ensure_models", lambda: True)
    engine.faces._models_ready = True
    engine.faces._embeddings[Role.PATIENT.value]["grandma_ref.jpg"] = grandma_emb

    def _fake_detect(img_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        h_i, w_i = img_bgr.shape[:2]
        if h_i < 48 or w_i < 48:
            return None, None
        return visitor_emb.copy(), (w_i // 2 - 24, h_i // 2 - 26, 48, 52)

    monkeypatch.setattr(engine.faces, "_detect_best_face_and_embedding", _fake_detect)

    # Формируем сырой кадр тревоги 540x960 и стоящего вне кровати родственника (ошибочно принятого за PATIENT)
    raw_frame = np.full((540, 960, 3), 120, dtype=np.uint8)
    ok_jpg, enc_jpg = cv2.imencode(".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok_jpg
    raw_jpeg_bytes = enc_jpg.tobytes()

    visitor_pose = _make_pose_full(
        track_id=77,
        bbox=(0.12, 0.22, 0.27, 0.88),
        shoulder=(0.195, 0.34),
        hip=(0.195, 0.58),
        nose=(0.195, 0.27),
        l_eye=(0.205, 0.26),
        r_eye=(0.185, 0.26),
        knee=(0.195, 0.73),
        ankle=(0.195, 0.86),
    )

    zones_now = engine.zones.get()
    track_dict = engine._build_track_ui_item(
        obs=visitor_pose,
        ident=TrackIdentity(track_id=77, role=Role.PATIENT, reason="reid_appearance"),
        zones=zones_now,
        state=PatientState.LEFT_BED,
    )

    alert_id = "20261113_123000_left_bed_w11a"
    ev = AlertEvent(
        state=PatientState.LEFT_BED,
        severity=AlertSeverity.WARNING,
        title="Встала с кровати одна",
        description="Ложное срабатывание на вошедшего сына.",
        timestamp=time.monotonic(),
        patient_track_id=77,
    )
    engine.feedback.register_recent_alert(
        alert_id=alert_id,
        event=ev,
        annotated_jpeg=raw_jpeg_bytes,
        raw_jpeg=raw_jpeg_bytes,
        context={
            "event_wall_ts": round(time.time(), 3),
            "telemetry_snapshot": {"tracks": [track_dict]},
            "zones": zones_now.to_dict(),
            "was_bed_latched": True,
        },
    )

    # Проверяем интеграцию через кнопку Telegram «❌ Ты ошибся» (wrong:<alert_id>)
    tg_calls: list[tuple[str, str]] = []

    def _mock_tg(request: httpx.Request) -> httpx.Response:
        ep = request.url.path.rsplit("/", 1)[-1]
        body_str = request.read().decode("utf-8", errors="replace")
        tg_calls.append((ep, body_str))
        if ep == "sendPhoto":
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 701}})
        return httpx.Response(200, json={"ok": True, "result": True})

    notifier = TelegramNotifier(
        bot_token="123456:TEST_TOKEN_WAVE11",
        chat_ids=("1001",),
        on_pause_cmd=lambda s: None,
        on_resume_cmd=lambda: None,
        get_snapshot_jpeg=lambda: raw_jpeg_bytes,
        is_fall_unacknowledged=lambda: False,
        on_false_alert_cmd=engine.report_false_alert,
        transport=httpx.MockTransport(_mock_tg),
    )
    msg_ids = notifier.send_alert_photo(ev, raw_jpeg_bytes, alert_id=alert_id)
    assert msg_ids == {"1001": 701}

    tg_calls.clear()
    with httpx.Client(transport=httpx.MockTransport(_mock_tg)) as client:
        notifier._handle_update(
            client,
            {
                "update_id": 501,
                "callback_query": {
                    "id": "cb_w11_1",
                    "data": f"wrong:{alert_id}",
                    "from": {"id": 1001, "first_name": "Алексей"},
                    "message": {"message_id": 701, "chat": {"id": 1001}},
                },
            },
        )

    edit_calls = [body for ep, body in tg_calls if ep == "editMessageCaption"]
    assert len(edit_calls) == 1
    parsed_edit = urllib.parse.parse_qs(edit_calls[0])
    edited_caption = parsed_edit["caption"][0]
    assert "Ошибка записана (Алексей)" in edited_caption
    assert "лицо «Своего» запомнено" in edited_caption

    visitor_photos = engine.faces.list_photos()[Role.VISITOR.value]
    assert len(visitor_photos) == 1
    saved_file = visitor_photos[0]
    assert saved_file.startswith(f"auto_visitor_{alert_id[:28]}_")
    assert (data_dir / "faces" / "visitor" / saved_file).is_file()

    # Теперь этот же человек на входе сразу опознаётся как Role.VISITOR по лицу!
    matched_role, matched_score = engine.faces.match_person_crop(
        raw_frame,
        visitor_pose,
        zones=zones_now,
    )
    assert matched_role == Role.VISITOR
    assert matched_score >= 0.99

    # Повторный вызов report_false_alert для того же лица НЕ плодит дубликат (already_known)
    dup_res = engine.report_false_alert(alert_id, reporter_chat_id="1001", reporter_name="Алексей")
    assert dup_res["visitor_face_saved"] is False
    assert dup_res["visitor_face_file"] is None
    assert dup_res["visitor_face_status"] == "already_known"
    assert len(engine.faces.list_photos()[Role.VISITOR.value]) == 1


def test_wave11_auto_enroll_safety_guards_reject_bed_back_turned_patient_similarity_and_rotate_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Проверяет ВСЕ железные предохранители безопасности FaceRegistry.auto_enroll_visitor_from_false_alert:
    - трек на кровати -> 'on_bed';
    - поза спиной (затылок) -> 'back_turned';
    - горизонтальная поза -> 'horizontal_posture';
    - отсутствие лицевых точек -> 'no_face_keypoints';
    - строгий запрет поворотов ±90° (allow_rotations=False);
    - лицо внутри bed_polygon или в 1.5% буфере -> 'inside_bed_polygon';
    - лицо внутри latched_patient_bed_bbox -> 'inside_latched_bed_bbox';
    - сходство с эталоном бабушки >= 0.48 -> 'too_close_to_patient';
    - ротация <= 24 файлов auto_visitor_* без удаления ручных снимков!
    """
    reg = FaceRegistry(faces_dir=tmp_path / "faces", models_dir=tmp_path / "models", threshold=0.40)
    monkeypatch.setattr(reg, "ensure_models", lambda: True)
    reg._models_ready = True

    grandma_emb = _unit_vec(303)
    reg._embeddings[Role.PATIENT.value]["grandma_1.jpg"] = grandma_emb

    # Создаём ручной портрет гостя, который никогда не должен удаляться авторотацией
    manual_path = tmp_path / "faces" / "visitor" / "manual_daughter.jpg"
    manual_path.write_bytes(b"\xff\xd8manual")
    reg._embeddings[Role.VISITOR.value]["manual_daughter.jpg"] = _unit_vec(999)

    frame = np.full((540, 960, 3), 100, dtype=np.uint8)

    # 1. Трек на кровати -> жёсткий отказ 'on_bed'
    on_bed_pose = _make_pose_full(
        1,
        (0.40, 0.42, 0.58, 0.70),
        (0.49, 0.48),
        (0.49, 0.62),
        nose=(0.49, 0.45),
    )
    ok, fname, reason = reg.auto_enroll_visitor_from_false_alert(frame, on_bed_pose, BED_AND_DOOR_ZONES)
    assert (ok, fname, reason) == (False, "", "on_bed")

    # 2. Затылок / спиной к камере вне кровати -> 'back_turned'
    back_pose = _make_pose_full(
        2,
        (0.12, 0.24, 0.26, 0.86),
        (0.19, 0.35),
        (0.19, 0.58),
        nose=None,
        l_eye=None,
        r_eye=None,
    )
    ok, fname, reason = reg.auto_enroll_visitor_from_false_alert(frame, back_pose, BED_AND_DOOR_ZONES)
    assert (ok, fname, reason) == (False, "", "back_turned")

    # 3. Горизонтальный наклон корпуса >= 42° -> 'horizontal_posture'
    horiz_pose = _make_pose_full(
        3,
        (0.10, 0.72, 0.30, 0.90),
        (0.14, 0.80),
        (0.27, 0.82),
        nose=(0.12, 0.79),
    )
    ok, fname, reason = reg.auto_enroll_visitor_from_false_alert(frame, horiz_pose, BED_AND_DOOR_ZONES)
    assert (ok, fname, reason) == (False, "", "horizontal_posture")

    # 4. Стоит у края кровати, но найденное на кропе лицо проецируется в буфер/внутрь bed_polygon -> 'inside_bed_polygon'
    near_bed_pose = _make_pose_full(
        4,
        (0.20, 0.28, 0.35, 0.90),
        (0.25, 0.42),
        (0.25, 0.66),
        nose=(0.28, 0.40),
        l_eye=(0.29, 0.39),
        r_eye=(0.27, 0.39),
    )
    distinct_emb = _unit_vec(404)

    # Смещаем детектируемый прямоугольник к правому краю головного кропа (в сторону x >= 0.31 -> буфер bed_polygon x=0.32!)
    def _detect_shifted_into_bed(img_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        h_i, w_i = img_bgr.shape[:2]
        return distinct_emb.copy(), (max(0, w_i - 44), h_i // 2 - 20, 40, 40)

    monkeypatch.setattr(reg, "_detect_best_face_and_embedding", _detect_shifted_into_bed)
    ok, fname, reason = reg.auto_enroll_visitor_from_false_alert(frame, near_bed_pose, BED_AND_DOOR_ZONES)
    assert (ok, fname, reason) == (False, "", "inside_bed_polygon")

    # 5. Запрет поворотов ±90°: если лицо находится только при повороте кропа, allow_rotations=False обязано отказать!
    clean_outside_pose = _make_pose_full(
        5,
        (0.10, 0.22, 0.25, 0.88),
        (0.175, 0.34),
        (0.175, 0.58),
        nose=(0.175, 0.27),
        l_eye=(0.185, 0.26),
        r_eye=(0.165, 0.26),
    )
    rot_calls: list[bool] = []
    orig_rot_fn = reg._detect_face_with_rot_code

    def _spy_rot(img_bgr: np.ndarray, allow_rotations: bool = True) -> Any:
        rot_calls.append(allow_rotations)
        return orig_rot_fn(img_bgr, allow_rotations=allow_rotations)

    monkeypatch.setattr(reg, "_detect_face_with_rot_code", _spy_rot)
    monkeypatch.setattr(reg, "_detect_best_face_and_embedding", lambda img: (None, None))
    ok, fname, reason = reg.auto_enroll_visitor_from_false_alert(frame, clean_outside_pose, BED_AND_DOOR_ZONES)
    assert ok is False
    assert reason == "no_face_detected"
    assert rot_calls == [False]

    # 6. Пороги анти-коллизии (v6.6.1):
    # 6а. Фоновый косинус RTSP-домена ~0.50 у двери (кейс 20261113_103839_left_bed_e7ad, 0.4912 < 0.58)
    # обязан УСПЕШНО проходить авто-энролл, а повторный вызов обязан вернуть 'already_known' ДО проверки бабушки!
    rtsp_domain_visitor = _unit_vec(504, mix_with=grandma_emb, mix_alpha=0.50)
    assert 0.49 <= float(np.dot(rtsp_domain_visitor, grandma_emb)) <= 0.51
    monkeypatch.setattr(
        reg,
        "_detect_best_face_and_embedding",
        lambda img: (rtsp_domain_visitor.copy(), (img.shape[1] // 2 - 22, img.shape[0] // 2 - 22, 44, 44)),
    )
    ok_rtsp, fname_rtsp, reason_rtsp = reg.auto_enroll_visitor_from_false_alert(
        frame,
        clean_outside_pose,
        BED_AND_DOOR_ZONES,
        alert_id="103839_e7ad",
    )
    assert ok_rtsp is True
    assert reason_rtsp == "enrolled"
    # Даже если у уже известного гостя (>= 0.82) сходство с эталоном бабушки высокое, возвращается 'already_known'
    known_plus_pat = _unit_vec(506, mix_with=rtsp_domain_visitor, mix_alpha=0.92)
    reg._embeddings[Role.PATIENT.value]["cam_domain_extra.jpg"] = _unit_vec(
        507,
        mix_with=known_plus_pat,
        mix_alpha=0.62,
    )
    monkeypatch.setattr(
        reg,
        "_detect_best_face_and_embedding",
        lambda img: (known_plus_pat.copy(), (img.shape[1] // 2 - 22, img.shape[0] // 2 - 22, 44, 44)),
    )
    ok_kn, fname_kn, reason_kn = reg.auto_enroll_visitor_from_false_alert(
        frame,
        clean_outside_pose,
        BED_AND_DOOR_ZONES,
    )
    assert (ok_kn, fname_kn, reason_kn) == (False, "", "already_known")
    reg._embeddings[Role.PATIENT.value].pop("cam_domain_extra.jpg", None)
    (tmp_path / "faces" / "visitor" / fname_rtsp).unlink(missing_ok=True)
    reg._embeddings[Role.VISITOR.value].pop(fname_rtsp, None)

    # 6б. Реальное сходство с бабушкой >= 0.58 у двери (и >= 0.54 в комнате вне двери) -> 'too_close_to_patient'
    similar_to_grandma = _unit_vec(505, mix_with=grandma_emb, mix_alpha=0.62)
    assert float(np.dot(similar_to_grandma, grandma_emb)) >= 0.60
    monkeypatch.setattr(
        reg,
        "_detect_best_face_and_embedding",
        lambda img: (similar_to_grandma.copy(), (img.shape[1] // 2 - 22, img.shape[0] // 2 - 22, 44, 44)),
    )
    ok, fname, reason = reg.auto_enroll_visitor_from_false_alert(frame, clean_outside_pose, BED_AND_DOOR_ZONES)
    assert (ok, fname, reason) == (False, "", "too_close_to_patient")

    mid_room_pose = _make_pose_full(
        55,
        (0.74, 0.24, 0.88, 0.88),
        (0.81, 0.35),
        (0.81, 0.58),
        nose=(0.81, 0.28),
        l_eye=(0.82, 0.27),
        r_eye=(0.80, 0.27),
    )
    similar_mid_room = _unit_vec(508, mix_with=grandma_emb, mix_alpha=0.55)
    assert 0.54 <= float(np.dot(similar_mid_room, grandma_emb)) < 0.58
    monkeypatch.setattr(
        reg,
        "_detect_best_face_and_embedding",
        lambda img: (similar_mid_room.copy(), (img.shape[1] // 2 - 22, img.shape[0] // 2 - 22, 44, 44)),
    )
    ok_mid, fname_mid, reason_mid = reg.auto_enroll_visitor_from_false_alert(
        frame,
        mid_room_pose,
        BED_AND_DOOR_ZONES,
    )
    assert (ok_mid, fname_mid, reason_mid) == (False, "", "too_close_to_patient")

    # 7. Квота <= 24 авто-портретов auto_visitor_*: при записи 26 разных гостей остаётся ровно 24 auto_visitor_* + ручной снимок!
    enrolled_names: list[str] = []
    for idx in range(26):
        unique_vis_emb = _unit_vec(10_000 + idx)
        # Гарантируем отсутствие случайной близости >= 0.48 к бабушке
        if float(np.dot(unique_vis_emb, grandma_emb)) >= 0.35:
            unique_vis_emb = -unique_vis_emb
        monkeypatch.setattr(
            reg,
            "_detect_best_face_and_embedding",
            lambda img, e=unique_vis_emb: (e.copy(), (img.shape[1] // 2 - 22, img.shape[0] // 2 - 22, 44, 44)),
        )
        ok_i, name_i, st_i = reg.auto_enroll_visitor_from_false_alert(
            frame,
            clean_outside_pose,
            BED_AND_DOOR_ZONES,
            alert_id=f"q_{idx:02d}",
        )
        assert ok_i is True
        assert st_i == "enrolled"
        fpath = tmp_path / "faces" / "visitor" / name_i
        ts_val = 1_700_000_000 + idx * 10
        os.utime(fpath, (ts_val, ts_val))
        enrolled_names.append(name_i)

    photos_after = reg.list_photos()[Role.VISITOR.value]
    auto_after = [n for n in photos_after if n.startswith("auto_visitor_")]
    assert len(auto_after) == 24
    assert "manual_daughter.jpg" in photos_after
    assert enrolled_names[0] not in auto_after
    assert enrolled_names[1] not in auto_after
    assert enrolled_names[0] not in reg._embeddings[Role.VISITOR.value]
    assert enrolled_names[-1] in auto_after


def test_wave11_seated_after_caregiver_leaves_stays_calm_in_bed_and_catches_real_fall() -> None:
    """
    Часть 2 (Сценарии 1 и 2):
    1) Родственник был рядом (caregiver_nearby=True), бабушка сидит на кровати со свешенными ногами.
       Родственник уходит (caregiver_nearby=False), бабушка продолжает сидеть ->
       state == IN_BED ('сидит после ухода родственника'), 0 тревог SITTING_UP / LEFT_BED даже через 35 секунд!
    2) Из этого сидячего положения бабушка заваливается/сползает на пол ->
       мгновенная критическая тревога FALLEN!
    """
    fsm = FallStateMachine(
        sit_confirm_seconds=1.5,
        leave_confirm_seconds=1.8,
        fall_confirm_seconds=1.2,
    )
    roles = {1: TrackIdentity(track_id=1, role=Role.PATIENT, reason="bed_sleep_lock")}
    t = 100.0

    # Калибруем сон бабушки в кровати
    sleep_pose = _make_pose_full(
        1,
        (0.36, 0.44, 0.64, 0.64),
        (0.42, 0.52),
        (0.57, 0.54),
        nose=(0.40, 0.50),
    )
    for _ in range(6):
        t += 0.3
        assert fsm.step([sleep_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t) is None
    assert fsm.state == PatientState.IN_BED

    # Пришёл родственник и помог бабушке сесть на край кровати (свесила ноги)
    sit_on_bed = _make_pose_full(
        1,
        (0.39, 0.36, 0.57, 0.77),
        (0.48, 0.42),
        (0.48, 0.59),
        nose=(0.48, 0.38),
        knee=(0.48, 0.65),
        ankle=(0.48, 0.76),
    )
    for _ in range(8):
        t += 0.3
        ev_cg = fsm.step([sit_on_bed], roles, caregiver_nearby=True, zones=BED_AND_DOOR_ZONES, now=t)
        assert ev_cg is None
        assert fsm.state == PatientState.CAREGIVER_ASSIST

    # Родственник ушёл (caregiver_nearby=False), а бабушка осталась ровно сидеть на кровати 35 секунд!
    for _ in range(70):
        t += 0.5
        ev_solo = fsm.step(
            [sit_on_bed],
            roles,
            caregiver_nearby=False,
            zones=BED_AND_DOOR_ZONES,
            now=t,
            local_hour=3,  # Даже в строгом ночном режиме 0 ложных тревог!
        )
        assert ev_solo is None
        assert fsm.state == PatientState.IN_BED
        assert fsm.status_reason == "сидит после ухода родственника"

    # Но как только бабушка из сидячего положения сползла/упала на пол — штатно бьёт критическая тревога FALLEN!
    fallen_on_floor = _make_pose_full(
        1,
        (0.30, 0.76, 0.66, 0.95),
        (0.36, 0.85),
        (0.58, 0.87),
        nose=(0.33, 0.84),
    )
    fall_alert: AlertEvent | None = None
    for _ in range(8):
        t += 0.25
        ev_fall = fsm.step([fallen_on_floor], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t)
        if ev_fall is not None:
            fall_alert = ev_fall
            break

    assert fall_alert is not None
    assert fall_alert.state == PatientState.FALLEN
    assert fall_alert.severity == AlertSeverity.CRITICAL
    assert fsm.state == PatientState.FALLEN


def test_wave11_seated_clears_after_lying_back_to_sleep_and_rearms_future_solo_sit_up() -> None:
    """
    Часть 2 (Сценарий 3):
    Бабушка осталась сидеть после ухода родственника (0 тревог), затем легла обратно спать на >= 4.0 с (5 с),
    а через минуту САМА села без родственника -> штатно срабатывает упреждающая тревога SITTING_UP!
    """
    fsm = FallStateMachine(sit_confirm_seconds=1.5, leave_confirm_seconds=1.8, fall_confirm_seconds=1.5)
    roles = {1: TrackIdentity(track_id=1, role=Role.PATIENT, reason="bed_sleep_lock")}
    t = 200.0

    sleep_pose = _make_pose_full(
        1,
        (0.36, 0.44, 0.64, 0.64),
        (0.42, 0.52),
        (0.57, 0.54),
        nose=(0.40, 0.50),
    )
    sit_pose = _make_pose_full(
        1,
        (0.39, 0.36, 0.57, 0.77),
        (0.48, 0.42),
        (0.48, 0.59),
        nose=(0.48, 0.38),
        knee=(0.48, 0.65),
        ankle=(0.48, 0.76),
    )

    for _ in range(6):
        t += 0.3
        fsm.step([sleep_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t)

    # Посадили при родственнике и ушли
    for _ in range(6):
        t += 0.3
        fsm.step([sit_pose], roles, caregiver_nearby=True, zones=BED_AND_DOOR_ZONES, now=t)
    for _ in range(10):
        t += 0.4
        assert fsm.step([sit_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t) is None
        assert fsm.state == PatientState.IN_BED
        assert fsm.status_reason == "сидит после ухода родственника"

    # Бабушка легла обратно горизонтально в кровать на 5 секунд (>= 4.0 с) -> защёлка сидения сбрасывается!
    for _ in range(12):
        t += 0.5
        assert fsm.step([sleep_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t) is None
    assert fsm.state == PatientState.IN_BED
    assert "сидит" not in fsm.status_reason

    # Через минуту бабушка САМА поднимается и садится без родственника -> штатно срабатывает SITTING_UP!
    t += 60.0
    fsm.step([sleep_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t)
    sit_alert: AlertEvent | None = None
    for _ in range(10):
        t += 0.25
        ev = fsm.step([sit_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t)
        if ev is not None:
            sit_alert = ev
            break

    assert sit_alert is not None
    assert sit_alert.state == PatientState.SITTING_UP


def test_wave11_seated_on_bedside_chair_stays_calm_catches_chair_slump_and_alerts_left_bed_on_walk() -> None:
    """
    Часть 2 (Сценарий 4 + тонкость прикроватного стула):
    1) Эталон сна откалиброван высоко на подушке (shoulder_y ~ 0.42). Родственник пересадил бабушку на
       прикроватный стул перед кроватью (upper_y = 0.55 — ниже подушки на 0.13, но на стуле!) и ушёл:
       НЕ срабатывает ложный FALLEN от сравнения с подушкой и НЕ срабатывает LEFT_BED -> спокойный IN_BED!
    2) Если бабушка сползает со стула вниз на >= 0.095 -> штатно срабатывает критический FALLEN!
    3) Если бабушка встаёт со стула/кровати и уходит пешком по комнате -> штатно срабатывает LEFT_BED!
    """
    fsm = FallStateMachine(sit_confirm_seconds=1.5, leave_confirm_seconds=1.5, fall_confirm_seconds=1.2)
    roles = {1: TrackIdentity(track_id=1, role=Role.PATIENT, reason="bed_sleep_lock")}
    t = 300.0

    high_pillow_sleep = _make_pose_full(
        1,
        (0.36, 0.38, 0.64, 0.56),
        (0.42, 0.42),
        (0.57, 0.45),
        nose=(0.40, 0.41),
    )
    for _ in range(6):
        t += 0.3
        fsm.step([high_pillow_sleep], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t)
    assert fsm.baseline_in_bed["calibrated"] is True

    # Прикроватный стул вне матраса (x=0.25 < bed_x1=0.32): upper_y=0.55 (на +0.13 ниже подушки 0.42!)
    chair_pose = _make_pose_full(
        1,
        (0.18, 0.46, 0.31, 0.86),
        (0.245, 0.55),
        (0.245, 0.70),
        nose=(0.245, 0.49),
        knee=(0.25, 0.77),
        ankle=(0.25, 0.85),
    )
    for _ in range(6):
        t += 0.3
        assert fsm.step([chair_pose], roles, caregiver_nearby=True, zones=BED_AND_DOOR_ZONES, now=t) is None

    # Родственник ушёл — бабушка спокойно сидит на стуле 30 с (0 ложных FALLEN и 0 ложных LEFT_BED!)
    for _ in range(60):
        t += 0.5
        ev_chair = fsm.step([chair_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t)
        assert ev_chair is None
        assert fsm.state == PatientState.IN_BED
        assert fsm.status_reason == "сидит после ухода родственника"

    # Ветка А: Бабушка встала со стула в полный рост и пошла пешком по комнате -> штатно срабатывает LEFT_BED!
    walk_away_pose = _make_pose_full(
        1,
        (0.68, 0.24, 0.84, 0.88),
        (0.76, 0.35),
        (0.76, 0.58),
        nose=(0.76, 0.28),
        knee=(0.76, 0.73),
        ankle=(0.76, 0.86),
    )
    left_alert: AlertEvent | None = None
    for _ in range(10):
        t += 0.25
        ev_w = fsm.step([walk_away_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t)
        if ev_w is not None:
            left_alert = ev_w
            break
    assert left_alert is not None
    assert left_alert.state == PatientState.LEFT_BED

    # Ветка Б: Повторно усаживаем на стул с родственником, уходим, и бабушка медленно сползает со стула на пол (+0.11 по Y)
    for _ in range(6):
        t += 0.3
        fsm.step([chair_pose], roles, caregiver_nearby=True, zones=BED_AND_DOOR_ZONES, now=t)
    for _ in range(4):
        t += 0.3
        assert fsm.step([chair_pose], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t) is None

    slumped_from_chair = _make_pose_full(
        1,
        (0.17, 0.58, 0.33, 0.94),
        (0.245, 0.665),
        (0.255, 0.82),
        nose=(0.245, 0.61),
    )
    slump_alert: AlertEvent | None = None
    for _ in range(10):
        t += 0.25
        ev_s = fsm.step([slumped_from_chair], roles, caregiver_nearby=False, zones=BED_AND_DOOR_ZONES, now=t)
        if ev_s is not None:
            slump_alert = ev_s
            break
    assert slump_alert is not None
    assert slump_alert.state == PatientState.FALLEN
    assert slump_alert.severity == AlertSeverity.CRITICAL


def test_wave11_hotfix_v661_reclining_pillow_sleeper_never_inverted_by_bed_edge_visitor_or_rotated_face(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Регрессия боевого Дефекта 1 (#171, снимок 12:58:46):
    - Бабушка (tid=9001) лежит горизонтально глубоко на подушке; на повёрнутом ±90° кропе подушки
      шумный боковой профиль случайно даёт best_visitor = 0.44 > best_patient + 0.02.
    - Родственница (tid=2), вошедшая через door_entry, садится вертикально на передний левый край матраса
      со стопами на полу (y >= 0.80).
    - Спящая tid=9001 НИКОГДА не должна получать face_visitor или проигрывать арбитраж tid=2;
      при уходе tid=2 к двери не срабатывает ложный LEFT_BED!
    """
    real_zones = ZonesConfig(
        bed_polygon=[
            [0.2516, 0.4222],
            [0.7031, 0.4311],
            [0.7188, 0.6133],
            [0.6469, 0.7867],
            [0.2750, 0.7822],
            [0.2391, 0.5911],
        ],
        door_polygon=[[0.0891, 0.0480], [0.2688, 0.0924], [0.2703, 0.9973], [0.1125, 0.9973]],
    )
    reg = FaceRegistry(faces_dir=tmp_path / "faces", models_dir=tmp_path / "models", threshold=0.38)
    monkeypatch.setattr(reg, "ensure_models", lambda: True)
    reg._models_ready = True

    vis_ref = _unit_vec(711)
    pat_ref = _unit_vec(712)
    noisy_pillow_emb = _unit_vec(713, mix_with=vis_ref, mix_alpha=0.44)
    reg._embeddings[Role.VISITOR.value]["daughter.jpg"] = vis_ref
    reg._embeddings[Role.PATIENT.value]["grandma.jpg"] = pat_ref

    # Симулируем ситуацию, где кроп подушки дал детекцию только при повороте ±90° с умеренным сходством к гостю (0.44)
    monkeypatch.setattr(
        reg,
        "_detect_face_with_rot_code",
        lambda crop_img, allow_rotations=True: (
            (noisy_pillow_emb.copy(), (crop_img.shape[1] // 2 - 20, crop_img.shape[0] // 2 - 20, 40, 40), crop_img, cv2.ROTATE_90_CLOCKWISE)
            if allow_rotations
            else (None, None, crop_img, None)
        ),
    )

    ident = PatientIdentifier(
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=2.0,
        face_registry=reg,
    )
    fsm = FallStateMachine(sit_confirm_seconds=1.5, leave_confirm_seconds=1.5)
    frame = np.full((540, 960, 3), 132, dtype=np.uint8)

    # tid=9001: бабушка спит горизонтально на подушке
    grandma_9001 = _make_pose_full(
        9001,
        (0.41, 0.44, 0.66, 0.59),
        (0.58, 0.49),
        (0.45, 0.52),
        nose=(0.60, 0.48),
        l_eye=(0.61, 0.47),
        r_eye=(0.59, 0.47),
    )
    role_9001_crop, _ = reg.match_person_crop(frame, grandma_9001, zones=real_zones)
    assert role_9001_crop != Role.VISITOR

    t = 0.0
    for _ in range(8):
        t += 0.3
        roles, cg = ident.update([grandma_9001], real_zones, now=t, frame_bgr=frame)
        assert fsm.step([grandma_9001], roles, cg, real_zones, now=t) is None
    assert roles[9001].role == Role.PATIENT

    # tid=2: родственница входит через door_polygon и садится вертикально на передний левый край матраса (стопы y=0.86)
    t += 0.3
    vis_door_0 = _make_pose_full(
        2,
        (0.11, 0.20, 0.25, 0.90),
        (0.18, 0.32),
        (0.18, 0.58),
        nose=(0.18, 0.25),
        ankle=(0.18, 0.88),
    )
    ident.update([grandma_9001, vis_door_0], real_zones, now=t, frame_bgr=frame)
    t += 0.3
    vis_door_1 = _make_pose_full(
        2,
        (0.16, 0.22, 0.29, 0.90),
        (0.225, 0.33),
        (0.225, 0.59),
        nose=(0.225, 0.26),
        ankle=(0.225, 0.88),
    )
    ident.update([grandma_9001, vis_door_1], real_zones, now=t, frame_bgr=frame)

    vis_bed_edge = _make_pose_full(
        2,
        (0.27, 0.41, 0.41, 0.89),
        (0.34, 0.49),
        (0.34, 0.67),
        nose=(0.34, 0.44),
        knee=(0.34, 0.77),
        ankle=(0.34, 0.86),
    )
    for _ in range(16):
        t += 1.0
        roles_edge, cg_edge = ident.update([grandma_9001, vis_bed_edge], real_zones, now=t, frame_bgr=frame)
        ev_edge = fsm.step([grandma_9001, vis_bed_edge], roles_edge, cg_edge, real_zones, now=t)
        assert roles_edge[9001].role == Role.PATIENT
        assert roles_edge[9001].reason != "face_visitor"
        assert roles_edge[2].role == Role.VISITOR
        assert cg_edge is True
        assert ev_edge is None

    # Родственница встаёт с края кровати и уходит в дверь -> бабушка спит на подушке, 0 ложных LEFT_BED!
    for step_x in (0.22, 0.16):
        t += 0.4
        vis_exit = _make_pose_full(
            2,
            (step_x - 0.06, 0.18, step_x + 0.07, 0.90),
            (step_x, 0.30),
            (step_x, 0.56),
            nose=(step_x, 0.23),
            ankle=(step_x, 0.88),
        )
        roles_ex, cg_ex = ident.update([grandma_9001, vis_exit], real_zones, now=t, frame_bgr=frame)
        assert fsm.step([grandma_9001, vis_exit], roles_ex, cg_ex, real_zones, now=t) is None

    for _ in range(10):
        t += 0.4
        roles_after, cg_after = ident.update([grandma_9001], real_zones, now=t, frame_bgr=frame)
        ev_after = fsm.step([grandma_9001], roles_after, cg_after, real_zones, now=t)
        assert roles_after[9001].role == Role.PATIENT
        assert ev_after is None
    assert fsm.state == PatientState.IN_BED


def test_wave11_hotfix_v661_foreground_left_chair_caregiver_123527_and_125111_zero_false_sitting_up() -> None:
    """
    Регрессия боевого Дефекта 2 (#171, пакеты 20261113_123527_sitting_up_e24b и 20261113_125111_sitting_up_5b6a):
    - Боевая разметка 192.168.1.100 включает левую настенную ignore_polygon [0.2328..0.3656, 0.0222..0.9956].
    - Даже после прогрева > 15 с без предварительного door_visit_active И после долгого сидения со сменой track_id,
      уверенный сидящий на переднем левом стуле родственник (conf=0.919, плечи + таз, y1 >= 0.62, y2 == 1.0)
      НЕ отсекается StaticObjectGate, получает Role.VISITOR и caregiver_nearby == True, подавляя ложный SITTING_UP!
    """
    real_zones = ZonesConfig(
        bed_polygon=[
            [0.2516, 0.4222],
            [0.7031, 0.4311],
            [0.7188, 0.6133],
            [0.6469, 0.7867],
            [0.2750, 0.7822],
            [0.2391, 0.5911],
        ],
        door_polygon=[[0.0891, 0.0480], [0.2688, 0.0924], [0.2703, 0.9973], [0.1125, 0.9973]],
        ignore_polygons=[
            [[0.7141, 0.0756], [0.8078, 0.0867], [0.7688, 0.6089], [0.7016, 0.6022]],
            [[0.2328, 0.0222], [0.3656, 0.0444], [0.3547, 0.9956], [0.2422, 0.9956]],
        ],
    )
    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(sit_confirm_seconds=1.2, suppress_sitting_when_caregiver=True)

    # 1. Бабушка спит, проходит > 20 с с запуска (включается strict_door_track)
    t = 0.0
    grandma_sleep = _make_pose_full(1, (0.35, 0.46, 0.64, 0.63), (0.55, 0.52), (0.41, 0.55), nose=(0.57, 0.50))
    for _ in range(6):
        t += 0.3
        roles, cg = ident.update([grandma_sleep], real_zones, now=t)
        assert fsm.step([grandma_sleep], roles, cg, real_zones, now=t) is None
    assert roles[1].role == Role.PATIENT

    t = 25.0
    grandma_sitting = _make_pose_full(
        1,
        (0.41, 0.37, 0.57, 0.77),
        (0.49, 0.43),
        (0.49, 0.60),
        nose=(0.49, 0.39),
        knee=(0.49, 0.68),
        ankle=(0.49, 0.77),
    )

    # Пакет 123527: bbox=[0.157, 0.634, 0.324, 1.0], conf=0.919, спиной/вполоборота (плечи 0.99, таз 0.96, без глаз)
    kps_123527: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 17
    kps_123527[5] = (0.205, 0.72, 0.98)
    kps_123527[6] = (0.275, 0.72, 1.00)
    kps_123527[7] = (0.185, 0.82, 0.75)
    kps_123527[8] = (0.295, 0.82, 0.78)
    kps_123527[11] = (0.210, 0.90, 0.94)
    kps_123527[12] = (0.270, 0.90, 0.98)
    chair_123527 = PoseObservation(
        track_id=41,
        bbox_norm=(0.157, 0.634, 0.324, 1.0),
        conf=0.919,
        keypoints_norm=tuple(kps_123527),
    )

    for _ in range(8):
        t += 0.3
        roles_1, cg_1 = ident.update([grandma_sitting, chair_123527], real_zones, now=t)
        ev_1 = fsm.step([grandma_sitting, chair_123527], roles_1, cg_1, real_zones, now=t)
        assert roles_1[1].role == Role.PATIENT
        assert 41 in roles_1 and roles_1[41].role == Role.VISITOR
        assert cg_1 is True
        assert ev_1 is None
        assert fsm.state == PatientState.CAREGIVER_ASSIST

    # Через 30 секунд спокойного сидения ByteTrack меняет track_id 41 -> 88 и стул чуть сдвинут правее
    # (точные координаты пакета 125111: bbox=[0.243, 0.629, 0.402, 1.0], conf=0.919, плечи 0.97/1.00, таз 0.76/0.88)
    t += 30.0
    kps_125111: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 17
    kps_125111[5] = (0.285, 0.72, 0.97)
    kps_125111[6] = (0.358, 0.72, 1.00)
    kps_125111[7] = (0.268, 0.82, 0.72)
    kps_125111[8] = (0.375, 0.82, 0.76)
    kps_125111[11] = (0.292, 0.90, 0.76)
    kps_125111[12] = (0.352, 0.90, 0.88)
    chair_125111 = PoseObservation(
        track_id=88,
        bbox_norm=(0.243, 0.629, 0.402, 1.0),
        conf=0.919,
        keypoints_norm=tuple(kps_125111),
    )

    for _ in range(8):
        t += 0.3
        roles_2, cg_2 = ident.update([grandma_sitting, chair_125111], real_zones, now=t)
        ev_2 = fsm.step([grandma_sitting, chair_125111], roles_2, cg_2, real_zones, now=t)
        assert roles_2[1].role == Role.PATIENT
        assert 88 in roles_2 and roles_2[88].role == Role.VISITOR
        assert cg_2 is True
        assert ev_2 is None
        assert fsm.state == PatientState.CAREGIVER_ASSIST

    # После реального выхода родственницы через дверь бабушка остаётся спокойно сидеть (_assisted_seated_latched == True)!
    t += 0.4
    leave_door = _make_pose_full(88, (0.11, 0.16, 0.25, 0.92), (0.18, 0.28), (0.18, 0.56), nose=(0.18, 0.22), ankle=(0.18, 0.90))
    roles_lv, cg_lv = ident.update([grandma_sitting, leave_door], real_zones, now=t)
    assert fsm.step([grandma_sitting, leave_door], roles_lv, cg_lv, real_zones, now=t) is None

    for _ in range(10):
        t += 0.4
        roles_solo, cg_solo = ident.update([grandma_sitting], real_zones, now=t)
        ev_solo = fsm.step([grandma_sitting], roles_solo, cg_solo, real_zones, now=t)
        assert ev_solo is None
    assert cg_solo is False
    assert fsm.state == PatientState.IN_BED
    assert fsm.status_reason == "сидит после ухода родственника"
