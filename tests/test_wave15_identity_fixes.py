from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.geometry import PoseObservation, is_person_on_bed
from app.homography import FloorHomography
from app.patient_id import FaceRegistry, PatientIdentifier, Role, TrackIdentity
from tests.test_wave10_feedback_fixes import REAL_ZONES, _build_engine, _jitter_obs, _obs


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


def _make_pose(
    track_id: int,
    bbox: tuple[float, float, float, float],
    shoulder: tuple[float, float],
    hip: tuple[float, float] | None = None,
    *,
    nose: tuple[float, float] | None = None,
    l_eye: tuple[float, float] | None = None,
    r_eye: tuple[float, float] | None = None,
    knee: tuple[float, float] | None = None,
    ankle: tuple[float, float] | None = None,
    conf: float = 0.90,
    face_conf: float = 0.92,
) -> PoseObservation:
    kps: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 17
    if nose is not None:
        kps[0] = (nose[0], nose[1], face_conf)
    if l_eye is not None:
        kps[1] = (l_eye[0], l_eye[1], face_conf)
    if r_eye is not None:
        kps[2] = (r_eye[0], r_eye[1], face_conf)
    kps[5] = (shoulder[0] + 0.025, shoulder[1], 0.92)
    kps[6] = (shoulder[0] - 0.025, shoulder[1], 0.92)
    if hip is not None:
        kps[11] = (hip[0] + 0.02, hip[1], 0.92)
        kps[12] = (hip[0] - 0.02, hip[1], 0.92)
    if knee is not None:
        kps[13] = (knee[0] + 0.02, knee[1], 0.90)
        kps[14] = (knee[0] - 0.02, knee[1], 0.90)
    if ankle is not None:
        kps[15] = (ankle[0] + 0.02, ankle[1], 0.90)
        kps[16] = (ankle[0] - 0.02, ankle[1], 0.90)
    return PoseObservation(
        track_id=track_id,
        bbox_norm=bbox,
        conf=conf,
        keypoints_norm=tuple(kps),
    )


def _sleep_grandma_1987(track_id: int = 1987) -> PoseObservation:
    return _make_pose(
        track_id=track_id,
        bbox=(0.43, 0.46, 0.65, 0.62),
        shoulder=(0.58, 0.52),
        hip=(0.48, 0.55),
        nose=(0.60, 0.50),
        l_eye=(0.61, 0.49),
        r_eye=(0.59, 0.49),
        conf=0.88,
    )


def _sitting_grandma_1834(track_id: int = 1834) -> PoseObservation:
    return _make_pose(
        track_id=track_id,
        bbox=(0.256, 0.428, 0.413, 0.917),
        shoulder=(0.335, 0.505),
        hip=(0.345, 0.650),
        nose=(0.338, 0.465),
        l_eye=(0.346, 0.456),
        r_eye=(0.330, 0.456),
        knee=(0.330, 0.760),
        ankle=(0.330, 0.875),
        conf=0.90,
    )


def _blanket_roi_9001(
    track_id: int = 9001,
    bbox: tuple[float, float, float, float] = (0.411, 0.456, 0.617, 0.772),
    conf: float = 0.17,
) -> PoseObservation:
    """Низкоуверенная складка жёлтого одеяла без таза (mh is None), проходящая is_blanket_sleeper_in_bed."""
    cx = (bbox[0] + bbox[2]) * 0.5
    cy = (bbox[1] + bbox[3]) * 0.5
    kps: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 17
    kps[0] = (cx + 0.02, cy - 0.03, 0.48)
    kps[1] = (cx + 0.03, cy - 0.04, 0.52)
    kps[3] = (cx + 0.04, cy - 0.03, 0.46)
    kps[5] = (cx + 0.05, cy, 0.51)
    kps[6] = (cx - 0.02, cy + 0.01, 0.44)
    return PoseObservation(
        track_id=track_id,
        bbox_norm=bbox,
        conf=conf,
        keypoints_norm=tuple(kps),
    )


def test_wave15_enroll_guards_no_fullframe_spatial_backturned_min_size_and_cross_gallery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Z3 п.1: защита энролла лиц в FaceRegistry и FallGuardEngine:
    - при переданном obs убран full-frame фолбэк;
    - проверка на затылок (back-turned) отклоняет энролл;
    - пространственное владение (spatial ownership) отклоняет лицо соседа;
    - минимальный размер лица >= 36 px в enroll_from_frame и add_photo;
    - cross-gallery guard: sim_other >= 0.56 ИЛИ (sim_other >= 0.48 и sim_other >= sim_own + 0.08),
      но НЕ жёсткий порог 0.45 (профиль бабушки 0.458 обязан проходить!).
    """
    reg = FaceRegistry(faces_dir=tmp_path / "faces", models_dir=tmp_path / "models", threshold=0.40)
    monkeypatch.setattr(reg, "ensure_models", lambda: True)
    reg._models_ready = True

    grandma_ref = _unit_vec(11)
    visitor_ref = _unit_vec(22)
    assert float(np.dot(grandma_ref, visitor_ref)) < 0.25
    reg._embeddings[Role.PATIENT.value]["grandma_base.jpg"] = grandma_ref
    reg._embeddings[Role.VISITOR.value]["caregiver_base.jpg"] = visitor_ref

    frame = np.full((540, 960, 3), 110, dtype=np.uint8)

    # 1. Затылок (back-turned) -> мгновенный отказ без обращения к детектору
    back_turned_obs = _make_pose(
        track_id=10,
        bbox=(0.30, 0.30, 0.46, 0.86),
        shoulder=(0.38, 0.42),
        hip=(0.38, 0.64),
        nose=None,
        l_eye=None,
        r_eye=None,
    )
    detect_called = {"n": 0}

    def _spy_detect(img_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        detect_called["n"] += 1
        h_i, w_i = img_bgr.shape[:2]
        return grandma_ref.copy(), (w_i // 2 - 22, h_i // 2 - 22, 44, 44)

    monkeypatch.setattr(reg, "_detect_best_face_and_embedding", _spy_detect)
    ok_bt, _ = reg.enroll_from_frame(Role.PATIENT, frame, obs=back_turned_obs)
    assert ok_bt is False
    assert detect_called["n"] == 0

    # 2. Нет full-frame фолбэка при переданном obs: лицо есть только на полном кадре (960x540), не в кропе obs
    grandma_obs = _sleep_grandma_1987(1987)

    def _only_full_frame_detect(img_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        h_i, w_i = img_bgr.shape[:2]
        if (h_i, w_i) == (540, 960):
            return visitor_ref.copy(), (200, 200, 50, 50)
        return None, None

    monkeypatch.setattr(reg, "_detect_best_face_and_embedding", _only_full_frame_detect)
    ok_ff, _ = reg.enroll_from_frame(Role.PATIENT, frame, obs=grandma_obs)
    assert ok_ff is False

    # 3. Spatial ownership: лицо в кропе сидит на координатах соседа (other_obs), ближе к нему -> отказ!
    neighbor_obs = _make_pose(
        track_id=3826,
        bbox=(0.45, 0.36, 0.57, 0.82),
        shoulder=(0.51, 0.45),
        hip=(0.51, 0.64),
        nose=(0.51, 0.49),
        l_eye=(0.52, 0.48),
        r_eye=(0.50, 0.48),
    )

    def _neighbor_shifted_detect(img_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        # Сдвигаем лицо в левый край кропа бабушки (~x=0.51, ровно голова neighbor_obs, далеко от x=0.60 бабушки)
        h_i, _ = img_bgr.shape[:2]
        return grandma_ref.copy(), (4, max(0, h_i // 2 - 22), 44, 44)

    monkeypatch.setattr(reg, "_detect_best_face_and_embedding", _neighbor_shifted_detect)
    ok_own, _ = reg.enroll_from_frame(
        Role.PATIENT,
        frame,
        obs=grandma_obs,
        other_observations=[neighbor_obs],
        zones=REAL_ZONES,
    )
    assert ok_own is False

    # 4. Минимальный размер лица >= 36 px в enroll_from_frame и add_photo
    clean_grandma_emb = _unit_vec(33, mix_with=grandma_ref, mix_alpha=0.75)

    def _small_face_detect(img_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        h_i, w_i = img_bgr.shape[:2]
        return clean_grandma_emb.copy(), (w_i // 2 - 16, h_i // 2 - 17, 32, 34)

    monkeypatch.setattr(reg, "_detect_best_face_and_embedding", _small_face_detect)
    ok_small, _ = reg.enroll_from_frame(Role.PATIENT, frame, obs=grandma_obs)
    assert ok_small is False

    ok_enc, enc_jpg = cv2.imencode(".jpg", frame)
    assert ok_enc
    ok_add_small, _ = reg.add_photo(Role.PATIENT, "small.jpg", enc_jpg.tobytes())
    assert ok_add_small is False

    # 5. Cross-gallery guard:
    # 5a) Профиль настоящей бабушки (sim_other=0.458, sim_own=0.40) -> РАЗРЕШЁН (не жёсткий 0.45!)
    # Строим точный вектор в плоскости (grandma_ref, visitor_ref) с dot(v, pat)=0.40, dot(v, vis)=0.458
    u1 = grandma_ref.copy()
    u2 = visitor_ref - float(np.dot(visitor_ref, u1)) * u1
    u2 /= float(np.linalg.norm(u2))
    u3 = _unit_vec(77)
    u3 = u3 - float(np.dot(u3, u1)) * u1 - float(np.dot(u3, u2)) * u2
    u3 /= float(np.linalg.norm(u3))

    def _vec_with_dots(dot_pat: float, dot_vis: float) -> np.ndarray:
        c1 = dot_pat
        p_dot_v = float(np.dot(grandma_ref, visitor_ref))
        v_norm_orth = float(np.dot(visitor_ref, u2))
        c2 = (dot_vis - c1 * p_dot_v) / v_norm_orth
        c3 = float(np.sqrt(max(0.0, 1.0 - c1 * c1 - c2 * c2)))
        vec = (c1 * u1 + c2 * u2 + c3 * u3).astype(np.float32)
        vec /= float(np.linalg.norm(vec))
        return vec

    profile_ok_emb = _vec_with_dots(0.40, 0.458)
    assert float(np.dot(profile_ok_emb, visitor_ref)) == pytest.approx(0.458, abs=0.005)
    monkeypatch.setattr(
        reg,
        "_detect_best_face_and_embedding",
        lambda img: (profile_ok_emb.copy(), (img.shape[1] // 2 - 20, img.shape[0] // 2 - 21, 40, 42)),
    )
    ok_prof, saved_prof = reg.enroll_from_frame(Role.PATIENT, frame, obs=grandma_obs)
    assert ok_prof is True
    assert saved_prof.endswith(".jpg")
    # Убираем временный эталон перед проверкой следующих порогов
    reg.delete_photo(Role.PATIENT, saved_prof)

    # 5b) sim_other=0.50, sim_own=0.65 -> РАЗРЕШЁН (0.50 < 0.56 и 0.50 < 0.65 + 0.08)
    own_strong_emb = _vec_with_dots(0.65, 0.50)
    monkeypatch.setattr(
        reg,
        "_detect_best_face_and_embedding",
        lambda img: (own_strong_emb.copy(), (img.shape[1] // 2 - 20, img.shape[0] // 2 - 21, 40, 42)),
    )
    ok_own_strong, saved_os = reg.enroll_from_frame(Role.PATIENT, frame, obs=grandma_obs)
    assert ok_own_strong is True
    reg.delete_photo(Role.PATIENT, saved_os)

    # 5c) sim_other=0.50, sim_own=0.30 (>= 0.48 и >= sim_own + 0.08) -> ОТКЛОНЁН в enroll_from_frame и add_photo
    caregiver_leak_emb = _vec_with_dots(0.30, 0.50)
    monkeypatch.setattr(
        reg,
        "_detect_best_face_and_embedding",
        lambda img: (caregiver_leak_emb.copy(), (img.shape[1] // 2 - 20, img.shape[0] // 2 - 21, 40, 42)),
    )
    ok_leak, _ = reg.enroll_from_frame(Role.PATIENT, frame, obs=grandma_obs)
    assert ok_leak is False
    ok_leak_add, _ = reg.add_photo(Role.PATIENT, "leak.jpg", enc_jpg.tobytes())
    assert ok_leak_add is False

    # 5d) sim_other=0.57, sim_own=0.60 (>= 0.56 даже при высоком sim_own) -> ОТКЛОНЁН
    hard_cross_emb = _vec_with_dots(0.60, 0.57)
    monkeypatch.setattr(
        reg,
        "_detect_best_face_and_embedding",
        lambda img: (hard_cross_emb.copy(), (img.shape[1] // 2 - 20, img.shape[0] // 2 - 21, 40, 42)),
    )
    ok_hard, _ = reg.enroll_from_frame(Role.PATIENT, frame, obs=grandma_obs)
    assert ok_hard is False
    ok_hard_add, _ = reg.add_photo(Role.PATIENT, "hard.jpg", enc_jpg.tobytes())
    assert ok_hard_add is False

    # 6. FallGuardEngine.enroll_face_from_camera с 2 людьми: сохраняет obs и other_observations и блокирует затылок
    engine = _build_engine(tmp_path)
    monkeypatch.setattr(engine.faces, "ensure_models", lambda: True)
    engine.faces._models_ready = True
    engine.faces._embeddings[Role.VISITOR.value]["caregiver.jpg"] = visitor_ref
    with engine._lock:
        engine._latest_raw_frame = frame.copy()
        engine._latest_observations = [back_turned_obs, neighbor_obs]
        engine._latest_identities = {
            back_turned_obs.track_id: TrackIdentity(track_id=back_turned_obs.track_id, role=Role.PATIENT),
            neighbor_obs.track_id: TrackIdentity(track_id=neighbor_obs.track_id, role=Role.VISITOR),
        }
    ok_eng_bt, _ = engine.enroll_face_from_camera(Role.PATIENT, track_id=back_turned_obs.track_id)
    assert ok_eng_bt is False

    # Если переданы 2 человека рядом и детектор находит лицо сиделки — cross-gallery guard в engine тоже отклонит
    monkeypatch.setattr(
        engine.faces,
        "_detect_best_face_and_embedding",
        lambda img: (visitor_ref.copy(), (img.shape[1] // 2 - 20, img.shape[0] // 2 - 20, 42, 44)),
    )
    with engine._lock:
        engine._latest_observations = [grandma_obs, neighbor_obs]
        engine._latest_identities = {
            grandma_obs.track_id: TrackIdentity(track_id=grandma_obs.track_id, role=Role.PATIENT),
            neighbor_obs.track_id: TrackIdentity(track_id=neighbor_obs.track_id, role=Role.VISITOR),
        }
    ok_eng_leak, _ = engine.enroll_face_from_camera(Role.PATIENT, track_id=grandma_obs.track_id)
    assert ok_eng_leak is False
    assert engine.faces.list_photos()[Role.PATIENT.value] == []


def test_wave15_3dde_match_person_crop_gates_and_door_entry_vs_reclining_patient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Z3 п.2 (инцидент 3dde):
    - match_person_crop запрещён для tid>=9000 & conf<0.35 и для любого conf<0.25;
    - matched_face_w >= 32 обязателен при on_bed;
    - сиделка (#3826, door_entry, on_bed) НЕ сбрасывает спящую бабушку (#1987, reclining_on_bed);
    - возвращена 2-hit проверка для VISITOR при on_bed;
    - разрешена перепроверка face_patient при not reclining_on_bed (2 сильных хита VISITOR -> face_visitor).
    """
    reg = FaceRegistry(faces_dir=tmp_path / "faces", models_dir=tmp_path / "models", threshold=0.40)
    monkeypatch.setattr(reg, "ensure_models", lambda: True)
    reg._models_ready = True

    pat_ref = _unit_vec(101)
    vis_ref = _unit_vec(202)
    reg._embeddings[Role.PATIENT.value]["pat.jpg"] = pat_ref
    reg._embeddings[Role.VISITOR.value]["vis.jpg"] = vis_ref

    frame = np.full((540, 960, 3), 120, dtype=np.uint8)
    face_state = {"emb": pat_ref, "w": 42}

    def _fake_rot(crop_bgr: np.ndarray, allow_rotations: bool = True):
        ch, cw = crop_bgr.shape[:2]
        fw = int(face_state["w"])
        rect = (max(0, (cw - fw) // 2), max(0, (ch - fw) // 2), fw, fw)
        return face_state["emb"].copy(), rect, crop_bgr, None

    monkeypatch.setattr(reg, "_detect_face_with_rot_code", _fake_rot)

    # 1. Запрет match_person_crop для tid>=9000 & conf<0.35 и для любого conf<0.25
    roi_low = _blanket_roi_9001(9001, conf=0.30)
    assert reg.match_person_crop(frame, roi_low, zones=REAL_ZONES) == (Role.UNKNOWN, 0.0)

    any_low = _make_pose(42, (0.40, 0.45, 0.62, 0.65), (0.52, 0.52), (0.45, 0.56), nose=(0.54, 0.50), conf=0.22)
    assert reg.match_person_crop(frame, any_low, zones=REAL_ZONES) == (Role.UNKNOWN, 0.0)

    # 2. matched_face_w >= 32 при on_bed
    grandma_1987 = _sleep_grandma_1987(1987)
    face_state["w"] = 28
    role_tiny, _ = reg.match_person_crop(frame, grandma_1987, zones=REAL_ZONES)
    assert role_tiny == Role.UNKNOWN

    face_state["w"] = 34
    role_ok, score_ok = reg.match_person_crop(frame, grandma_1987, zones=REAL_ZONES)
    assert role_ok == Role.PATIENT
    assert score_ok >= 0.90

    # 3. Инцидент 3dde: спящая бабушка #1987 + сиделка #3826 (door_entry), севшая на край кровати (on_bed=True)
    ident = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        face_registry=reg,
        state_path=tmp_path / "anchor_3dde.json",
    )
    t = 0.0
    for _ in range(5):
        t += 0.3
        roles, _ = ident.update([grandma_1987], REAL_ZONES, now=t, frame_bgr=frame)
    assert roles[1987].role == Role.PATIENT

    # Сиделка #3826 входит через дверь
    t += 0.3
    cg_door_0 = _make_pose(3826, (0.10, 0.24, 0.22, 0.86), (0.16, 0.35), (0.16, 0.58), nose=(0.16, 0.29), ankle=(0.16, 0.84))
    ident.update([grandma_1987, cg_door_0], REAL_ZONES, now=t, frame_bgr=frame)
    t += 0.3
    cg_door_1 = _make_pose(3826, (0.16, 0.24, 0.28, 0.86), (0.22, 0.35), (0.22, 0.58), nose=(0.22, 0.29), ankle=(0.22, 0.84))
    roles_d, _ = ident.update([grandma_1987, cg_door_1], REAL_ZONES, now=t, frame_bgr=frame)
    assert roles_d[3826].role == Role.VISITOR
    assert roles_d[3826].reason == "door_entry"

    # Сиделка садится на край кровати (on_bed=True), а отравленный SFace возвращает Role.PATIENT (0.55) на #3826!
    cg_on_bed = _make_pose(
        3826,
        (0.28, 0.42, 0.42, 0.86),
        (0.35, 0.50),
        (0.35, 0.66),
        nose=(0.35, 0.46),
        l_eye=(0.36, 0.45),
        r_eye=(0.34, 0.45),
        knee=(0.35, 0.75),
        ankle=(0.35, 0.85),
    )
    assert is_person_on_bed(cg_on_bed, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    monkeypatch.setattr(
        reg,
        "match_person_crop",
        lambda frm, obs, **kw: (Role.PATIENT, 0.55) if obs.track_id == 3826 else (Role.UNKNOWN, 0.0),
    )
    for _ in range(4):
        t += 2.2
        roles_3dde, cg_3dde = ident.update([grandma_1987, cg_on_bed], REAL_ZONES, now=t, frame_bgr=frame)
        assert roles_3dde[1987].role == Role.PATIENT
        assert roles_3dde[3826].role == Role.VISITOR
        assert cg_3dde is True

    # 4. 2-hit проверка для VISITOR на кровати (когда нет другого спящего пациента)
    ident_2hit = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        face_registry=reg,
        state_path=tmp_path / "anchor_2hit.json",
    )
    sit_vis = _make_pose(
        50,
        (0.29, 0.42, 0.43, 0.86),
        (0.36, 0.50),
        (0.36, 0.66),
        nose=(0.36, 0.46),
        l_eye=(0.37, 0.45),
        r_eye=(0.35, 0.45),
        knee=(0.36, 0.75),
        ankle=(0.36, 0.85),
    )
    t2 = 10.0
    ident_2hit._tracks[50] = TrackIdentity(
        track_id=50,
        role=Role.VISITOR,
        reason="co_present_with_patient",
        first_seen=5.0,
        last_seen=9.5,
        last_face_check=0.0,
        has_moved=True,
        confirmed_second_track=True,
    )
    ident_2hit._door_visit_active = True
    ident_2hit._door_visit_started_at = 5.0
    ident_2hit._last_visitor_seen_at = 9.5

    monkeypatch.setattr(reg, "match_person_crop", lambda frm, obs, **kw: (Role.PATIENT, 0.52))
    # 1-й хит -> остаётся VISITOR (2-hit guard!)
    roles_h1, _ = ident_2hit.update([sit_vis], REAL_ZONES, now=t2, frame_bgr=frame)
    assert roles_h1[50].role == Role.VISITOR
    assert roles_h1[50].face_patient_hits == 1

    # 2-й хит через >= 2.0 с -> переходит в PATIENT (face_patient)
    t2 += 2.2
    roles_h2, _ = ident_2hit.update([sit_vis], REAL_ZONES, now=t2, frame_bgr=frame)
    assert roles_h2[50].role == Role.PATIENT
    assert roles_h2[50].reason == "face_patient"

    # 5. Разрешена перепроверка face_patient при not reclining_on_bed: 2 сильных хита VISITOR снимают ложный face_patient!
    monkeypatch.setattr(reg, "match_person_crop", lambda frm, obs, **kw: (Role.VISITOR, 0.60))
    t2 += 2.2
    roles_r1, _ = ident_2hit.update([sit_vis], REAL_ZONES, now=t2, frame_bgr=frame)
    assert roles_r1[50].role == Role.PATIENT
    assert roles_r1[50].face_visitor_hits == 1

    t2 += 2.2
    roles_r2, _ = ident_2hit.update([sit_vis], REAL_ZONES, now=t2, frame_bgr=frame)
    assert roles_r2[50].role == Role.VISITOR
    assert roles_r2[50].reason == "face_visitor"


def test_wave15_b07d_6020_d8b9_d627_synthetic_roi_never_visitor_and_closed_door_dedup(
    tmp_path: Path,
) -> None:
    """
    Z3 п.3 (инциденты b07d, 6020, d8b9, d627):
    - при закрытой двери (door_visit_active=False) дубль #9001 рядом с сидящей бабушкой #1834 подавляется;
    - #9001 НИКОГДА не становится Role.VISITOR (в т.ч. после временного исчезновения #1834 — 6020/d8b9);
    - два перекрывающихся бокса на одной сидящей бабушке (#1383 + #1582, d627) дедупятся при dv=False;
    - other_on_bed_edge_with_sleeper не обходит strict_door_now=True;
    - реальный высокий ByteTrack ID (18445) при входе через дверь штатно получает Role.VISITOR.
    """
    ident = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        state_path=tmp_path / "anchor_b07d.json",
    )
    grandma_1834 = _sitting_grandma_1834(1834)
    blanket_9001 = _blanket_roi_9001(9001, bbox=(0.411, 0.456, 0.617, 0.780), conf=0.17)

    # Прогрев > 15 с (включается strict_door_now=True, door_visit_active=False)
    t = 0.0
    for i in range(6):
        t += 0.3
        ident.update([_jitter_obs(_sleep_grandma_1987(1834), 0.001 * (i % 2))], REAL_ZONES, now=t)
    assert ident.door_visit_active is False

    # b07d: t > 15 с, закрытая дверь, сидящая бабушка #1834 + фантом одеяла #9001
    t = 22.0
    for i in range(8):
        t += 0.3
        roles, cg = ident.update(
            [_jitter_obs(grandma_1834, 0.001 * (i % 2)), blanket_9001],
            REAL_ZONES,
            now=t,
        )
        assert 1834 in roles and roles[1834].role == Role.PATIENT
        assert all(tr.role != Role.VISITOR for tr in roles.values())
        assert cg is False

    # 6020 / d8b9: #9001 искусственно имеет Role.PATIENT / confirmed_second_track в памяти, возвращается #1834 ->
    # #9001 не обходит 4-кадровый гейт и НИКОГДА не становится Role.VISITOR!
    ident._tracks[9001] = TrackIdentity(
        track_id=9001,
        role=Role.PATIENT,
        reason="sole_bed_sleeper_recovery",
        first_seen=t - 2.0,
        last_seen=t,
        confirmed_second_track=True,
    )
    t += 0.3
    roles_6020, cg_6020 = ident.update([grandma_1834, blanket_9001], REAL_ZONES, now=t)
    assert roles_6020[1834].role == Role.PATIENT
    assert all(tr.role != Role.VISITOR for tr in roles_6020.values())
    assert 9001 not in roles_6020 or roles_6020[9001].role != Role.VISITOR
    assert cg_6020 is False

    # d627: закрытая дверь (dv=False), два перекрывающихся реальных бокса на одной сидящей бабушке (#1383 + #1582)
    ident_d627 = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        state_path=tmp_path / "anchor_d627.json",
    )
    td = 0.0
    for i in range(6):
        td += 0.3
        ident_d627.update([_jitter_obs(_sleep_grandma_1987(1383), 0.001 * (i % 2))], REAL_ZONES, now=td)
    td = 22.0
    box_1383 = _make_pose(
        1383,
        (0.286, 0.480, 0.448, 0.816),
        (0.365, 0.545),
        (0.375, 0.665),
        nose=(0.365, 0.505),
        knee=(0.365, 0.740),
        conf=0.84,
    )
    box_1582 = _make_pose(
        1582,
        (0.349, 0.491, 0.529, 0.901),
        (0.415, 0.550),
        (0.425, 0.670),
        nose=(0.415, 0.510),
        knee=(0.420, 0.755),
        ankle=(0.425, 0.875),
        conf=0.89,
    )
    for _ in range(6):
        td += 0.3
        roles_d627, cg_d627 = ident_d627.update([box_1383, box_1582], REAL_ZONES, now=td)
        assert len(roles_d627) == 1
        only_tr = next(iter(roles_d627.values()))
        assert only_tr.role == Role.PATIENT
        assert cg_d627 is False

    # Проверка _arbitrate_multi_person_roles напрямую: при allow_non_door_visitor=False (strict_door_now=True)
    # other_on_bed_edge_with_sleeper НЕ превращает недверной трек на краю кровати в Role.VISITOR
    ident_arb = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        state_path=tmp_path / "anchor_arb.json",
    )
    sleeper = _sleep_grandma_1987(10)
    edge_dup = _sitting_grandma_1834(11)
    ident_arb._tracks[10] = TrackIdentity(track_id=10, role=Role.PATIENT, reason="bed_sleep_lock", bed_accum_sec=12.0, was_on_bed=True)
    ident_arb._tracks[11] = TrackIdentity(track_id=11, role=Role.UNKNOWN, reason="init")
    ident_arb._arbitrate_multi_person_roles([sleeper, edge_dup], REAL_ZONES, now=30.0, allow_non_door_visitor=False)
    assert ident_arb._tracks[10].role == Role.PATIENT
    assert ident_arb._tracks[11].role != Role.VISITOR

    # Высокий реальный ByteTrack ID (18445) при входе через дверь НЕ путается с синтетическим ROI (9000..9999)
    t += 0.3
    vis_18445_a = _make_pose(18445, (0.11, 0.24, 0.23, 0.86), (0.17, 0.35), (0.17, 0.58), nose=(0.17, 0.29), ankle=(0.17, 0.84))
    ident.update([grandma_1834, vis_18445_a], REAL_ZONES, now=t)
    t += 0.3
    vis_18445_b = _make_pose(18445, (0.17, 0.24, 0.29, 0.86), (0.23, 0.35), (0.23, 0.58), nose=(0.23, 0.29), ankle=(0.23, 0.84))
    roles_hi, cg_hi = ident.update([grandma_1834, vis_18445_b], REAL_ZONES, now=t)
    assert 18445 in roles_hi and roles_hi[18445].role == Role.VISITOR
    assert cg_hi is True


def test_wave15_0071_560e_latch_reset_and_empty_bed_low_conf_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Z3 п.4 (инциденты 0071 и 560e):
    - видимый уход PATIENT с кровати сбрасывает patient_in_bed_latched даже при homography.calibrated=True
      и при активном door_visit_active=True (уход с сиделкой);
    - после ухода (_ever_locked_patient and not _patient_in_bed_latched) неподвижное пустое одеяло
      (conf < 0.35, mh is None) НЕ захватывает bed_sleep_lock / reid_bed_position / face_patient;
    - возврат реальной бабушки (conf >= 0.48, mh != None, движение) снова штатно защёлкивает кровать;
    - Закон сохранения под одеялом при уходе одной сиделки сохранён (bl остаётся True).
    """
    reg = FaceRegistry(faces_dir=tmp_path / "faces", models_dir=tmp_path / "models", threshold=0.40)
    monkeypatch.setattr(reg, "ensure_models", lambda: True)
    reg._models_ready = True
    monkeypatch.setattr(reg, "match_person_crop", lambda frm, obs, **kw: (Role.PATIENT, 0.65))

    ident = PatientIdentifier(
        bed_lock_seconds=6.0,
        initial_bed_lock_seconds=0.8,
        face_registry=reg,
        state_path=tmp_path / "anchor_0071.json",
    )
    frame = np.full((540, 960, 3), 100, dtype=np.uint8)

    # 1. Бабушка сидит/спит на кровати при калиброванной гомографии -> защёлка patient_in_bed_latched == True
    t = 0.0
    sleep_g = _sleep_grandma_1987(320)
    for _ in range(5):
        t += 0.3
        roles, _ = ident.update([sleep_g], REAL_ZONES, now=t, frame_bgr=frame)
    assert roles[320].role == Role.PATIENT
    assert ident.patient_in_bed_latched is True

    # 2. Входит сиделка (#512, door_visit_active=True), и бабушка (#320) уходит с кровати к ходункам (not pat_on_bed)
    t += 0.3
    cg_0 = _make_pose(512, (0.11, 0.24, 0.22, 0.86), (0.16, 0.35), (0.16, 0.58), nose=(0.16, 0.29), ankle=(0.16, 0.84))
    ident.update([sleep_g, cg_0], REAL_ZONES, now=t, frame_bgr=frame)
    t += 0.3
    cg_1 = _make_pose(512, (0.17, 0.24, 0.28, 0.86), (0.22, 0.35), (0.22, 0.58), nose=(0.22, 0.29), ankle=(0.22, 0.84))
    ident.update([sleep_g, cg_1], REAL_ZONES, now=t, frame_bgr=frame)
    assert ident.door_visit_active is True

    # Даже если у трека #320 сбросить was_on_bed и bed_accum_sec (как после ре-детекции сидящей при calibrated homography),
    # совпадение с _latched_patient_track_id обязано сбросить patient_in_bed_latched при выходе с матраса!
    ident._tracks[320].was_on_bed = False
    ident._tracks[320].bed_accum_sec = 0.0

    grandma_walkers = _make_pose(
        320,
        (0.494, 0.581, 0.686, 0.993),
        (0.580, 0.650),
        (0.590, 0.780),
        nose=(0.575, 0.615),
        knee=(0.595, 0.880),
        ankle=(0.600, 0.970),
        conf=0.88,
    )
    assert is_person_on_bed(grandma_walkers, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False
    t += 0.4
    roles_walk, _ = ident.update([grandma_walkers, cg_1], REAL_ZONES, now=t, frame_bgr=frame)
    assert roles_walk[320].role == Role.PATIENT
    assert ident.patient_in_bed_latched is False

    # 3. Обе ушли, комната пуста, на пустом матрасе 10 с подряд галлюцинирует смятое одеяло (#9001, conf=0.18, mh=None):
    # НЕ захватывает bed_sleep_lock / reid_bed_position / face_patient и НЕ защёлкивает bl!
    blanket_empty = _blanket_roi_9001(9001, bbox=(0.305, 0.510, 0.507, 0.772), conf=0.18)
    assert blanket_empty.mid_hip is None
    for _ in range(25):
        t += 0.5
        roles_empty, _ = ident.update([blanket_empty], REAL_ZONES, now=t, frame_bgr=frame)
        if 9001 in roles_empty:
            assert roles_empty[9001].role != Role.PATIENT
        assert ident.patient_in_bed_latched is False

    # То же для обычного (не-9000) неподвижного трека одеяла с conf=0.28, mh=None
    blanket_plain = _blanket_roi_9001(1987, bbox=(0.305, 0.510, 0.507, 0.772), conf=0.28)
    for _ in range(15):
        t += 0.5
        roles_plain, _ = ident.update([blanket_plain], REAL_ZONES, now=t, frame_bgr=frame)
        if 1987 in roles_plain:
            assert roles_plain[1987].role != Role.PATIENT
        assert ident.patient_in_bed_latched is False

    # 4. Бабушка реально вернулась в кровать (уверенный скелет с тазом, conf=0.88) -> защёлка восстанавливается!
    for _ in range(5):
        t += 0.4
        roles_ret, _ = ident.update([sleep_g], REAL_ZONES, now=t, frame_bgr=frame)
    assert roles_ret[320].role == Role.PATIENT
    assert ident.patient_in_bed_latched is True

    # 5. Закон сохранения под одеялом: бабушка скрылась под одеялом (0 скелетов на кровати),
    # сиделка вошла и вышла одна через дверь -> patient_in_bed_latched обязан остаться True!
    t += 0.4
    ident.update([cg_0], REAL_ZONES, now=t)
    t += 0.4
    ident.update([cg_1], REAL_ZONES, now=t)
    t += 3.5
    ident.update([cg_0], REAL_ZONES, now=t)
    for _ in range(8):
        t += 0.5
        ident.update([], REAL_ZONES, now=t)
    assert ident.patient_in_bed_latched is True

    # 6. Уход PATIENT через дверь при активном door_visit_active=True (ветка L4217):
    # даже если _patient_in_bed_latched был True до выхода в дверь, уход PATIENT через дверь сбрасывает защёлку!
    t += 1.0
    ident._door_visit_active = True
    ident._door_visit_started_at = t - 2.0
    ident._last_visitor_seen_at = t - 0.5
    ident._patient_in_bed_latched = True
    pat_at_door = _make_pose(
        320,
        (0.14, 0.28, 0.25, 0.86),
        (0.195, 0.38),
        (0.195, 0.60),
        nose=(0.195, 0.32),
        ankle=(0.195, 0.84),
        conf=0.88,
    )
    ident._tracks[320].was_on_bed = False
    ident._tracks[320].bed_accum_sec = 0.0
    ident._latched_patient_track_id = 999  # симулируем смену ID, чтобы проверить именно ветку L4217!
    ident._tracks[320].reason = "bed_sleep_lock"
    ident.update([pat_at_door], REAL_ZONES, now=t)
    t += 2.5
    ident.update([], REAL_ZONES, now=t)
    assert ident.patient_in_bed_latched is False


def test_wave15_bc51_far_hallway_passby_and_fast_transient_clear(tmp_path: Path) -> None:
    """
    Z3 п.5 (инцидент bc51):
    - проход снаружи в далёком коридоре (homography.calibrated, floor_xy_cm[1] < -25, [0] < -40, y2 < 0.80)
      НЕ взводит door_visit_active и caregiver_nearby на 1-м кадре (#17885/#17886);
    - при реальном шаге в комнату через порог на следующем кадре (y2 >= 0.80) мгновенно включает door_entry;
    - короткий transient_door_passby при visit_span < 0.6 с сбрасывается за 1.2 с (вместо 2.5 с).
    """
    hom = FloorHomography.from_zones(REAL_ZONES, bed_polygon=REAL_ZONES.bed_polygon)
    assert hom.calibrated is True

    far_hall_17885 = _make_pose(
        17885,
        (0.09, 0.28, 0.18, 0.74),
        (0.135, 0.40),
        (0.135, 0.55),
        nose=(0.135, 0.32),
        ankle=(0.135, 0.72),
        conf=0.86,
    )
    far_hall_17885 = PoseObservation(
        track_id=17885,
        bbox_norm=far_hall_17885.bbox_norm,
        conf=far_hall_17885.conf,
        keypoints_norm=far_hall_17885.keypoints_norm,
        liveness_verified=True,
        confirmed_frames=2,
    )
    m_far = hom.measure_pose(far_hall_17885, bed_polygon=REAL_ZONES.bed_polygon)
    assert m_far.floor_xy_cm is not None
    assert m_far.floor_xy_cm[1] < -25.0 and m_far.floor_xy_cm[0] < -40.0
    assert far_hall_17885.bbox_norm[3] < 0.80

    ident = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        door_visit_idle_timeout_sec=14400.0,
        state_path=tmp_path / "anchor_bc51.json",
    )
    t = 0.0
    # 1. Пустая комната + 1-кадровый проход в далёком коридоре (#17885, #17886) -> 0 вспышек door_visit_active / cg!
    t += 0.5
    roles_1, cg_1 = ident.update([far_hall_17885], REAL_ZONES, now=t)
    assert ident.door_visit_active is False
    assert cg_1 is False
    assert 17885 not in roles_1 or roles_1[17885].role != Role.VISITOR

    t += 0.5
    roles_empty, cg_empty = ident.update([], REAL_ZONES, now=t)
    assert ident.door_visit_active is False
    assert cg_empty is False
    assert roles_empty == {}

    # 2. Если тот же трек на следующем кадре реально переступает порог (y2 = 0.84 >= 0.80) -> мгновенно VISITOR!
    far_hall_17886 = PoseObservation(
        track_id=17886,
        bbox_norm=far_hall_17885.bbox_norm,
        conf=0.86,
        keypoints_norm=far_hall_17885.keypoints_norm,
        liveness_verified=True,
        confirmed_frames=2,
    )
    t += 1.0
    roles_f1, cg_f1 = ident.update([far_hall_17886], REAL_ZONES, now=t)
    assert ident.door_visit_active is False
    assert cg_f1 is False

    step_in_17886 = _make_pose(
        17886,
        (0.14, 0.26, 0.25, 0.84),
        (0.195, 0.36),
        (0.195, 0.58),
        nose=(0.195, 0.30),
        ankle=(0.195, 0.82),
        conf=0.90,
    )
    step_in_17886 = PoseObservation(
        track_id=17886,
        bbox_norm=step_in_17886.bbox_norm,
        conf=step_in_17886.conf,
        keypoints_norm=step_in_17886.keypoints_norm,
        liveness_verified=True,
        confirmed_frames=3,
    )
    t += 0.3
    roles_in, cg_in = ident.update([step_in_17886], REAL_ZONES, now=t)
    assert roles_in[17886].role == Role.VISITOR
    assert roles_in[17886].reason == "door_entry"
    assert ident.door_visit_active is True
    assert cg_in is True

    # 3. Ускоренный transient_door_passby: при visit_span < 0.6 с сбрасывается через 1.2 с (раньше 2.5 с!)
    ident_fast = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        door_visit_idle_timeout_sec=14400.0,
        state_path=tmp_path / "anchor_fast.json",
    )
    tp = 0.0
    for i in range(5):
        tp += 0.3
        ident_fast.update([_jitter_obs(_sleep_grandma_1987(1), 0.001 * (i % 2))], REAL_ZONES, now=tp)

    hall_a = _make_pose(88, (0.11, 0.26, 0.19, 0.82), (0.15, 0.40), (0.15, 0.60), nose=(0.15, 0.30), ankle=(0.15, 0.80))
    hall_b = _make_pose(88, (0.15, 0.27, 0.23, 0.83), (0.19, 0.41), (0.19, 0.61), nose=(0.19, 0.31), ankle=(0.19, 0.81))
    tp += 0.4
    ident_fast.update([_sleep_grandma_1987(1), hall_a], REAL_ZONES, now=tp)
    tp += 0.3  # visit_span = 0.3 с (< 0.6 с)
    ident_fast.update([_sleep_grandma_1987(1), hall_b], REAL_ZONES, now=tp)
    assert ident_fast.door_visit_active is True

    # Через 1.3 с (>= 1.2 с, но < 2.5 с!) портал уже сброшен
    tp += 1.3
    ident_fast.update([_sleep_grandma_1987(1)], REAL_ZONES, now=tp)
    assert ident_fast.door_visit_active is False

    # 4. При более долгом присутствии в проёме (visit_span = 1.0 с >= 0.6 с) 1.3 с ещё НЕ сбрасывают, а >= 2.5 с сбрасывают
    hall_long_a = _make_pose(89, (0.11, 0.26, 0.19, 0.82), (0.15, 0.40), (0.15, 0.60), nose=(0.15, 0.30), ankle=(0.15, 0.80))
    hall_long_b = _make_pose(89, (0.15, 0.27, 0.23, 0.83), (0.19, 0.41), (0.19, 0.61), nose=(0.19, 0.31), ankle=(0.19, 0.81))
    tp += 1.0
    ident_fast.update([_sleep_grandma_1987(1), hall_long_a], REAL_ZONES, now=tp)
    tp += 0.4
    ident_fast.update([_sleep_grandma_1987(1), hall_long_b], REAL_ZONES, now=tp)
    tp += 1.0  # visit_span = 1.0 с >= 0.6 с
    ident_fast.update([_sleep_grandma_1987(1), hall_long_b], REAL_ZONES, now=tp)
    assert ident_fast.door_visit_active is True

    tp += 1.5  # 1.5 с < 2.5 с -> ещё держится
    ident_fast.update([_sleep_grandma_1987(1)], REAL_ZONES, now=tp)
    assert ident_fast.door_visit_active is True

    tp += 1.2  # итого 2.7 с >= 2.5 с -> сброшен
    ident_fast.update([_sleep_grandma_1987(1)], REAL_ZONES, now=tp)
    assert ident_fast.door_visit_active is False
