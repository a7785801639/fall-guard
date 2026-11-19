from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.geometry import (
    PoseObservation,
    are_legs_hanging_outside_bed,
    is_person_on_bed,
    is_standing_in_front_of_bed,
)
from app.patient_id import PatientIdentifier, Role
from app.state_machine import AlertEvent, AlertSeverity, FallStateMachine, PatientState
from tests.test_wave10_feedback_fixes import REAL_ZONES, _build_engine, _jitter_obs, _obs

# Боевые скелеты сидящей на левом крае матраса бабушки из пакетов 19.11.2026 (192.168.1.21)
SEATED_GRANDMA_20261119: dict[str, PoseObservation] = {
    "091043_left_bed": _obs(
        6928,
        (0.2536, 0.4561, 0.4111, 0.8879),
        [
            (0.3343, 0.5342, 0.882),
            (0.3431, 0.5298, 0.915),
            (0.3384, 0.5186, 0.625),
            (0.363, 0.5243, 0.946),
            (0.0, 0.0, 0.0),
            (0.3814, 0.5479, 0.996),
            (0.3233, 0.5147, 0.989),
            (0.3999, 0.6184, 0.994),
            (0.2977, 0.537, 0.966),
            (0.3979, 0.6856, 0.989),
            (0.2681, 0.5722, 0.959),
            (0.3586, 0.6549, 1.0),
            (0.3227, 0.6361, 0.999),
            (0.3393, 0.7443, 0.998),
            (0.3067, 0.7269, 0.996),
            (0.3556, 0.8299, 0.984),
            (0.3313, 0.8172, 0.974),
        ],
    ),
    "091209_sitting_up": _obs(
        6928,
        (0.277, 0.4246, 0.4241, 0.87),
        [
            (0.3656, 0.47, 0.986),
            (0.3723, 0.4625, 0.992),
            (0.3651, 0.461, 0.684),
            (0.3889, 0.4729, 0.988),
            (0.0, 0.0, 0.0),
            (0.4041, 0.5271, 0.998),
            (0.3598, 0.5118, 0.994),
            (0.4106, 0.5965, 0.99),
            (0.3381, 0.5613, 0.879),
            (0.3884, 0.631, 0.98),
            (0.3263, 0.5802, 0.829),
            (0.3669, 0.6266, 0.998),
            (0.3384, 0.6104, 0.995),
            (0.3348, 0.712, 0.993),
            (0.3083, 0.6865, 0.982),
            (0.3144, 0.8372, 0.945),
            (0.2878, 0.8012, 0.902),
        ],
    ),
    "091434_sitting_up": _obs(
        6928,
        (0.2659, 0.4568, 0.4201, 0.8697),
        [
            (0.3681, 0.5259, 0.983),
            (0.3773, 0.5214, 0.985),
            (0.3661, 0.5129, 0.836),
            (0.3941, 0.5212, 0.965),
            (0.0, 0.0, 0.0),
            (0.4029, 0.5605, 0.997),
            (0.3494, 0.5161, 0.997),
            (0.4105, 0.6289, 0.988),
            (0.3136, 0.5456, 0.988),
            (0.3978, 0.6634, 0.984),
            (0.2876, 0.5968, 0.983),
            (0.3554, 0.6512, 0.998),
            (0.3226, 0.6249, 0.998),
            (0.3279, 0.7467, 0.993),
            (0.3018, 0.7186, 0.992),
            (0.3227, 0.84, 0.943),
            (0.3034, 0.8194, 0.941),
        ],
    ),
    "091634_sitting_up": _obs(
        6928,
        (0.2874, 0.4566, 0.4285, 0.8617),
        [
            (0.3768, 0.5123, 0.987),
            (0.3863, 0.5068, 0.989),
            (0.3759, 0.4974, 0.9),
            (0.404, 0.5183, 0.966),
            (0.0, 0.0, 0.0),
            (0.4093, 0.5682, 0.996),
            (0.3646, 0.5165, 0.996),
            (0.4127, 0.6336, 0.979),
            (0.3335, 0.5157, 0.974),
            (0.3878, 0.6345, 0.971),
            (0.3449, 0.4864, 0.962),
            (0.3602, 0.6457, 0.997),
            (0.3319, 0.6136, 0.996),
            (0.3313, 0.7387, 0.988),
            (0.3079, 0.6956, 0.985),
            (0.3099, 0.8432, 0.922),
            (0.2931, 0.801, 0.915),
        ],
    ),
    "091720_left_bed": _obs(
        6957,
        (0.2582, 0.4479, 0.4071, 0.9183),
        [
            (0.353, 0.489, 0.981),
            (0.3602, 0.4825, 0.985),
            (0.3504, 0.4805, 0.808),
            (0.3753, 0.4891, 0.976),
            (0.0, 0.0, 0.0),
            (0.39, 0.5253, 0.998),
            (0.339, 0.5175, 0.996),
            (0.403, 0.5775, 0.988),
            (0.3155, 0.5571, 0.963),
            (0.3754, 0.5806, 0.98),
            (0.3192, 0.5527, 0.939),
            (0.3544, 0.6327, 0.998),
            (0.3223, 0.6221, 0.998),
            (0.3196, 0.731, 0.994),
            (0.2942, 0.7195, 0.99),
            (0.2914, 0.8456, 0.952),
            (0.2837, 0.8333, 0.935),
        ],
    ),
    "091720_sitting_up": _obs(
        6957,
        (0.2615, 0.4484, 0.4118, 0.9172),
        [
            (0.3577, 0.487, 0.984),
            (0.3653, 0.4803, 0.99),
            (0.3555, 0.4788, 0.714),
            (0.3811, 0.4894, 0.988),
            (0.0, 0.0, 0.0),
            (0.3933, 0.5289, 0.998),
            (0.3453, 0.522, 0.996),
            (0.4096, 0.5827, 0.991),
            (0.318, 0.5579, 0.948),
            (0.3847, 0.5977, 0.983),
            (0.3155, 0.5543, 0.915),
            (0.3526, 0.6329, 0.998),
            (0.3223, 0.6222, 0.997),
            (0.3178, 0.7315, 0.992),
            (0.2937, 0.7191, 0.983),
            (0.2925, 0.8477, 0.914),
            (0.2808, 0.8322, 0.867),
        ],
    ),
}


def _grandma_sleep_obs(track_id: int = 6928) -> PoseObservation:
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[0] = (0.59, 0.50, 0.90)
    kps[5] = (0.55, 0.54, 0.95)
    kps[6] = (0.55, 0.51, 0.95)
    kps[11] = (0.42, 0.57, 0.95)
    kps[12] = (0.42, 0.54, 0.95)
    kps[13] = (0.36, 0.60, 0.90)
    kps[14] = (0.36, 0.58, 0.90)
    kps[15] = (0.32, 0.61, 0.88)
    kps[16] = (0.32, 0.59, 0.88)
    return _obs(track_id, (0.30, 0.45, 0.64, 0.66), kps)


def _with_dropped_ankles(obs: PoseObservation, conf: float = 0.20) -> PoseObservation:
    """Снижает уверенность коленей/лодыжек на 1-2 кадра (симуляция перекрытия пледом/ракурса)."""
    kps = list(obs.keypoints_norm)
    for idx in (13, 14, 15, 16):
        kps[idx] = (kps[idx][0], kps[idx][1], conf)
    # Имитируем кратковременный наклон корпуса вперед/вбок, когда носки временно пропадают
    kps[5] = (kps[5][0] + 0.03, kps[5][1] + 0.06, kps[5][2])
    kps[6] = (kps[6][0] + 0.03, kps[6][1] + 0.06, kps[6][2])
    return PoseObservation(
        track_id=obs.track_id,
        bbox_norm=obs.bbox_norm,
        conf=obs.conf,
        keypoints_norm=tuple(kps),
    )


def test_wave13_real_seated_grandma_6_packages_never_standing_in_front_or_left_bed() -> None:
    """
    Волна 13:
    1) Все 6 реальных поз сидящей на матрасе бабушки (angle 10.3°..39.8°) на калиброванном полу REAL_ZONES
       больше не проваливаются в 2D-фолбэк (is_standing_in_front_of_bed == False, is_person_on_bed == True).
    2) При длительном сидении > 120 с со скачками углов и 1-секундными провалами лодыжек —
       0 ложных LEFT_BED и ровно 1 SITTING_UP без стробоскопа каждые 45 с!
    """
    for name, pose in SEATED_GRANDMA_20261119.items():
        assert is_standing_in_front_of_bed(pose, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False, (
            f"Сидячая поза {name} ложно признана стоящей перед кроватью!"
        )
        assert is_person_on_bed(pose, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True, (
            f"Сидячая поза {name} потеряла принадлежность кровати!"
        )

    ident = PatientIdentifier(bed_lock_seconds=12.0, initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(
        sit_confirm_seconds=2.5,
        leave_confirm_seconds=2.0,
        alert_cooldown_seconds=45.0,
    )

    t = 0.0
    for i in range(10):
        t += 0.3
        sleep_obs = _jitter_obs(_grandma_sleep_obs(6928), 0.002 * (i % 2))
        roles, cg = ident.update([sleep_obs], REAL_ZONES, now=t)
        assert fsm.step([sleep_obs], roles, cg, REAL_ZONES, now=t) is None
    assert roles[6928].role == Role.PATIENT
    assert fsm.state == PatientState.IN_BED

    cycle = [
        SEATED_GRANDMA_20261119["091043_left_bed"],
        SEATED_GRANDMA_20261119["091720_left_bed"],
        SEATED_GRANDMA_20261119["091209_sitting_up"],
        SEATED_GRANDMA_20261119["091720_sitting_up"],
        SEATED_GRANDMA_20261119["091434_sitting_up"],
        SEATED_GRANDMA_20261119["091634_sitting_up"],
    ]
    sitting_events: list[AlertEvent] = []
    left_events: list[AlertEvent] = []
    for step_idx in range(280):  # 280 * 0.5s = 140s (> 3 кулдауна по 45 с)
        t += 0.5
        base_pose = cycle[(step_idx // 5) % len(cycle)]
        # На каждых 15-м шаге 2 кадра (= 1.0 с < 3.5 с гистерезиса) лодыжки временно теряются
        if step_idx % 15 in (10, 11) and step_idx >= 12:
            cur_pose = _with_dropped_ankles(base_pose, conf=0.20)
        else:
            cur_pose = _jitter_obs(base_pose, 0.001 * (step_idx % 2))
        roles, cg = ident.update([cur_pose], REAL_ZONES, now=t)
        ev = fsm.step(
            [cur_pose],
            roles,
            cg,
            REAL_ZONES,
            now=t,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )
        assert fsm.state != PatientState.LEFT_BED, f"Ложный LEFT_BED на шаге {step_idx}"
        if ev is not None:
            if ev.state == PatientState.LEFT_BED:
                left_events.append(ev)
            elif ev.state == PatientState.SITTING_UP:
                sitting_events.append(ev)

    assert left_events == []
    assert len(sitting_events) == 1


def test_wave13_night_side_sleep_dangling_shins_silent_and_real_walker_exit_alerts() -> None:
    """
    Волна 13:
    1) Ночной сон глубоко на боку со свешенными вниз голенями (angle >= 60°, малый rise <= 0.085)
       и мелкое поправление подушки (shoulder_lift < 0.085) не поднимают ложный SITTING_UP.
    2) Реальный выход с ходунками на пол вдали от матраса (07:31:27) надёжно даёт LEFT_BED.
    """
    side_sleep_kps = [(0.0, 0.0, 0.0)] * 17
    side_sleep_kps[0] = (0.59, 0.51, 0.90)
    side_sleep_kps[5] = (0.55, 0.54, 0.94)
    side_sleep_kps[6] = (0.55, 0.52, 0.93)
    side_sleep_kps[11] = (0.43, 0.59, 0.95)
    side_sleep_kps[12] = (0.43, 0.57, 0.94)
    side_sleep_kps[13] = (0.37, 0.69, 0.88)
    side_sleep_kps[14] = (0.36, 0.68, 0.87)
    side_sleep_kps[15] = (0.35, 0.835, 0.85)  # голени свешены ниже bed_max_y (0.79135)
    side_sleep_kps[16] = (0.34, 0.825, 0.84)
    side_sleep_dangling = _obs(7001, (0.32, 0.46, 0.63, 0.85), side_sleep_kps)

    assert are_legs_hanging_outside_bed(side_sleep_dangling, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False
    assert is_person_on_bed(side_sleep_dangling, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    # Реальное стояние с ходунками перед кроватью на полу (y2=0.96, рост ~152 см, впереди по Y на полу)
    walker_kps = [(0.0, 0.0, 0.0)] * 17
    walker_kps[0] = (0.33, 0.44, 0.92)
    walker_kps[5] = (0.35, 0.50, 0.95)
    walker_kps[6] = (0.31, 0.50, 0.95)
    walker_kps[11] = (0.35, 0.68, 0.95)
    walker_kps[12] = (0.31, 0.68, 0.95)
    walker_kps[13] = (0.34, 0.82, 0.95)
    walker_kps[14] = (0.31, 0.82, 0.95)
    walker_kps[15] = (0.34, 0.93, 0.93)
    walker_kps[16] = (0.31, 0.93, 0.93)
    real_walker_exit = _obs(7001, (0.28, 0.41, 0.40, 0.95), walker_kps)
    assert is_standing_in_front_of_bed(real_walker_exit, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True
    assert is_person_on_bed(real_walker_exit, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False

    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(sit_confirm_seconds=2.0, leave_confirm_seconds=1.5)
    t = 0.0
    for i in range(20):
        t += 0.5
        p = _jitter_obs(side_sleep_dangling, 0.001 * (i % 2))
        roles, cg = ident.update([p], REAL_ZONES, now=t)
        ev = fsm.step([p], roles, cg, REAL_ZONES, now=t, local_hour=2)
        assert ev is None

    left_ev = None
    for i in range(10):
        t += 0.4
        p_out = _jitter_obs(real_walker_exit, 0.002 * (i % 2))
        roles, cg = ident.update([p_out], REAL_ZONES, now=t)
        ev = fsm.step([p_out], roles, cg, REAL_ZONES, now=t, local_hour=7)
        if ev is not None and ev.state == PatientState.LEFT_BED:
            left_ev = ev
            break
    assert left_ev is not None


def test_wave13_no_midroom_visitor_halo_no_ghost_clone_minus2_and_passby_autoclose(
    tmp_path: Path,
) -> None:
    """
    Волна 13 (причины появления «Своего» посреди комнаты без прохода через дверь):
    1) Сидячая бабушка у левого края кровати (#6982, bbox [0.2906, 0.5138, 0.4913, 0.8970])
       не цепляет ореол двери и не активирует door_visit_active.
    2) Ошибочная защёлка _latched_visitor_bbox поверх бабушки очищается при переходе трека в PATIENT,
       а engine._apply_presence_persistence_locked не рисует клона track_id=-2 (door_visit_seated) поверх бабушки.
    3) Короткий проход по коридору мимо открытой двери (< 3 с без входа в комнату) сбрасывается за 2.5 с.
    """
    ident = PatientIdentifier(
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=1.0,
        door_visit_idle_timeout_sec=14400.0,
    )
    t = 0.0
    for i in range(8):
        t += 0.3
        p = _jitter_obs(_grandma_sleep_obs(1), 0.001 * (i % 2))
        ident.update([p], REAL_ZONES, now=t)

    # Прошёл прогрев > 15 с
    t = 25.0
    kps_6982 = list(SEATED_GRANDMA_20261119["091720_left_bed"].keypoints_norm)
    pose_6982 = PoseObservation(
        track_id=6982,
        bbox_norm=(0.2906, 0.5138, 0.4913, 0.8970),
        conf=0.92,
        keypoints_norm=tuple(kps_6982),
    )
    for i in range(6):
        t += 0.3
        roles, cg = ident.update([_jitter_obs(pose_6982, 0.001 * (i % 2))], REAL_ZONES, now=t)
        assert roles[6982].reason not in ("door_entry", "door_visit_companion")
    assert ident.door_visit_active is False

    # Имитация остаточной защёлки гостя на боксе бабушки + автоочистка при наличии PATIENT
    ident._door_visit_active = True
    ident._door_visit_from_real_door = False
    ident._latched_visitor_track_id = 6982
    ident._latched_visitor_bbox = (0.2906, 0.5138, 0.4913, 0.8970)
    ident._visitor_spot_occupied = True
    frame_room = np.full((180, 320, 3), 85, dtype=np.uint8)
    roles_c, cg_c = ident.update([pose_6982], REAL_ZONES, now=t + 0.5, frame_bgr=frame_room)
    assert roles_c[6982].role == Role.PATIENT
    assert ident.door_visit_active is False
    assert ident.visitor_spot_occupied is False
    assert cg_c is False

    # Проверка _apply_presence_persistence_locked: даже если door_visit_active=True и spot_occupied=True,
    # карточка door_visit_seated никогда не ставится поверх карточки PATIENT!
    engine = _build_engine(tmp_path)
    pat_card = [
        {
            "track_id": 6982,
            "role": Role.PATIENT.value,
            "reason": "face_patient",
            "bbox": [0.2906, 0.5138, 0.4913, 0.8970],
            "keypoints": [],
        }
    ]
    engine.identifier._door_visit_active = True
    engine.identifier._visitor_spot_occupied = True
    engine.identifier._latched_visitor_bbox = (0.2906, 0.5138, 0.4913, 0.8970)
    persisted = engine._apply_presence_persistence_locked(pat_card, PatientState.IN_BED, REAL_ZONES)
    assert not any(c.get("reason") == "door_visit_seated" for c in persisted)

    # Короткий проход по коридору мимо открытой двери (< 2.0 с) без входа внутрь комнаты
    ident_pass = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=1.0,
        door_visit_idle_timeout_sec=14400.0,
    )
    tp = 0.0
    for i in range(6):
        tp += 0.3
        ident_pass.update([_jitter_obs(_grandma_sleep_obs(1), 0.001 * (i % 2))], REAL_ZONES, now=tp)

    tp += 0.5
    hall_t0 = _obs(
        88,
        (0.11, 0.26, 0.19, 0.82),
        [
            (0.15, 0.30, 0.90) if i == 0
            else (0.13, 0.44, 0.90) if i == 5
            else (0.17, 0.44, 0.90) if i == 6
            else (0.13, 0.62, 0.90) if i == 11
            else (0.17, 0.62, 0.90) if i == 12
            else (0.0, 0.0, 0.0)
            for i in range(17)
        ],
    )
    hall_t1 = _obs(
        88,
        (0.16, 0.27, 0.24, 0.83),
        [
            (0.20, 0.31, 0.90) if i == 0
            else (0.18, 0.45, 0.90) if i == 5
            else (0.22, 0.45, 0.90) if i == 6
            else (0.18, 0.63, 0.90) if i == 11
            else (0.22, 0.63, 0.90) if i == 12
            else (0.0, 0.0, 0.0)
            for i in range(17)
        ],
    )
    ident_pass.update([_grandma_sleep_obs(1), hall_t0], REAL_ZONES, now=tp)
    tp += 0.4
    ident_pass.update([_grandma_sleep_obs(1), hall_t1], REAL_ZONES, now=tp)
    assert ident_pass.door_visit_active is True

    # Прохожий скрылся в коридоре -> через >= 2.5 с портал сбрасывается сам без 4-часового зависания
    for _ in range(7):
        tp += 0.5
        ident_pass.update([_grandma_sleep_obs(1)], REAL_ZONES, now=tp)
    assert ident_pass.door_visit_active is False


def test_wave13_false_alarm_on_lone_seated_face_patient_keeps_patient_and_latches_calm_sit(
    tmp_path: Path,
) -> None:
    """
    Волна 13: нажатие «Ложная тревога» по SITTING_UP / LEFT_BED, когда одинокая бабушка
    (face_patient, was_on_bed=True) сидит на кровати, НЕ переводит её в Role.VISITOR
    и защёлкивает спокойное сидение без повторных SITTING_UP.
    """
    engine = _build_engine(tmp_path)
    zones = engine.zones.update(
        bed_polygon=REAL_ZONES.bed_polygon,
        door_polygon=REAL_ZONES.door_polygon,
        ignore_polygons=REAL_ZONES.ignore_polygons,
        floor_polygon=REAL_ZONES.floor_polygon,
        floor_width_cm=REAL_ZONES.floor_width_cm,
        floor_depth_cm=REAL_ZONES.floor_depth_cm,
        mattress_height_cm=REAL_ZONES.mattress_height_cm,
    )

    t = 10.0
    for i in range(8):
        t += 0.3
        sleep_p = _jitter_obs(_grandma_sleep_obs(6928), 0.001 * (i % 2))
        roles, cg = engine.identifier.update([sleep_p], zones=zones, now=t)
        engine.fsm.step([sleep_p], roles, cg, zones, now=t)

    seated_p = SEATED_GRANDMA_20261119["091720_sitting_up"]
    seated_p = PoseObservation(
        track_id=6928,
        bbox_norm=seated_p.bbox_norm,
        conf=seated_p.conf,
        keypoints_norm=seated_p.keypoints_norm,
    )
    for i in range(10):
        t += 0.3
        cur = _jitter_obs(seated_p, 0.001 * (i % 2))
        roles, cg = engine.identifier.update([cur], zones=zones, now=t)
        engine.fsm.step([cur], roles, cg, zones, now=t)

    engine.identifier._tracks[6928].role = Role.PATIENT
    engine.identifier._tracks[6928].reason = "face_patient"
    engine.identifier._tracks[6928].was_on_bed = True
    engine.fsm._state = PatientState.SITTING_UP

    with engine._lock:
        engine._latest_raw_frame = np.full((360, 640, 3), 80, dtype=np.uint8)
        engine._latest_observations = [seated_p]
        engine._latest_identities = dict(engine.identifier._tracks)
        engine._sync_tracks_summary_locked()

    ev = AlertEvent(
        state=PatientState.SITTING_UP,
        severity=AlertSeverity.WARNING,
        title="Пытается встать с кровати",
        description="Сидит на кровати",
        timestamp=t,
        patient_track_id=6928,
    )
    clip = engine.archive.save_event_snapshot(ev, b"\xff\xd8\xff\xe0SIT_JPG", unix_ts=time.time())
    aid = str(clip["id"])
    engine.feedback.register_recent_alert(
        alert_id=aid,
        event=ev,
        annotated_jpeg=b"\xff\xd8\xff\xe0SIT_JPG",
        raw_jpeg=b"\xff\xd8\xff\xe0RAW_JPG",
        context={
            "event_wall_ts": time.time(),
            "telemetry_snapshot": engine.get_telemetry_snapshot(),
            "zones": zones.to_dict(),
            "was_bed_latched": False,
        },
    )

    res = engine.report_false_alert(aid, reporter_name="SM-S942B")
    assert res["ok"] is True
    assert res["reassigned_visitor_track_id"] is None
    assert engine.identifier._tracks[6928].role == Role.PATIENT
    assert engine.identifier.door_visit_active is False
    assert engine.fsm.state == PatientState.IN_BED

    # Дальнейшее спокойное сидение на следующих кадрах не перезапускает SITTING_UP
    t_after = t + 50.0
    for i in range(10):
        t_after += 0.4
        cur = _jitter_obs(seated_p, 0.001 * (i % 2))
        roles_a, cg_a = engine.identifier.update([cur], zones=zones, now=t_after)
        ev_a = engine.fsm.step([cur], roles_a, cg_a, zones, now=t_after)
        assert ev_a is None
        assert engine.fsm.state == PatientState.IN_BED
