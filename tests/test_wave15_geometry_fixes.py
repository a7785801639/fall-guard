from __future__ import annotations

from app.geometry import (
    KP_L_ANKLE,
    KP_L_HIP,
    KP_L_KNEE,
    KP_L_SHOULDER,
    KP_NOSE,
    KP_R_ANKLE,
    KP_R_HIP,
    KP_R_KNEE,
    KP_R_SHOULDER,
    PoseObservation,
    StaticObjectGate,
    are_legs_hanging_outside_bed,
    is_person_on_bed,
    is_pose_touching_door,
    is_standing_in_front_of_bed,
    polygon_distance_norm,
    polygon_y_bounds,
)
from app.homography import FloorHomography
from app.patient_id import PatientIdentifier, Role
from app.state_machine import AlertEvent, FallStateMachine, PatientState
from tests.test_wave10_feedback_fixes import CAREGIVER_FB_POSES, REAL_ZONES, _jitter_obs, _obs
from tests.test_wave13_standup_and_midroom_visitor_fixes import _grandma_sleep_obs

# Боевой скелет #846 из инцидента 20261120_133538_left_bed_a690 (сидящая на кровати бабушка
# с вытянутыми на коврик ногами: floor_xy=(58.6, 76.5) >= 75 см, hip_on_mattress_y=-42.9 см,
# оба колена внутри bed_polygon: y=0.7566 / 0.7707 < bed_max_y=0.79135).
GRANDMA_133538_A690 = _obs(
    846,
    (0.3176, 0.4474, 0.4535, 0.9471),
    [
        (0.385, 0.4935, 0.993),
        (0.3911, 0.4846, 0.994),
        (0.3834, 0.4857, 0.875),
        (0.4079, 0.4888, 0.988),
        (0.0, 0.0, 0.0),
        (0.424, 0.5371, 0.999),
        (0.3727, 0.5316, 0.999),
        (0.4456, 0.6165, 0.996),
        (0.3479, 0.5888, 0.987),
        (0.4178, 0.6085, 0.994),
        (0.3319, 0.6395, 0.978),
        (0.3918, 0.6554, 1.0),
        (0.3581, 0.6509, 1.0),
        (0.3705, 0.7566, 0.999),
        (0.3326, 0.7707, 0.999),
        (0.3899, 0.8485, 0.993),
        (0.3359, 0.892, 0.989),
    ],
)

# Реальный трек #768 (бабушка у кресла-туалета в ~37-41% от двери) из dvr_20261119_160540_687d
GRANDMA_CHAIR_687D = _obs(
    768,
    (0.6112, 0.7168, 0.7461, 0.994),
    [
        (0.6319, 0.8192, 0.786),
        (0.6403, 0.8044, 0.894),
        (0.0, 0.0, 0.0),
        (0.6646, 0.7992, 0.927),
        (0.0, 0.0, 0.0),
        (0.6956, 0.8694, 0.938),
        (0.6604, 0.8237, 0.924),
        (0.6833, 0.9682, 0.58),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ],
)


def _simple_pose(
    track_id: int,
    bbox: tuple[float, float, float, float],
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    knee: tuple[float, float] | None = None,
    ankle: tuple[float, float] | None = None,
    nose: tuple[float, float] | None = None,
) -> PoseObservation:
    kps = [(0.0, 0.0, 0.0)] * 17
    if nose is not None:
        kps[KP_NOSE] = (nose[0], nose[1], 0.92)
    else:
        kps[KP_NOSE] = (shoulder[0], max(0.01, shoulder[1] - 0.06), 0.90)
    kps[KP_L_SHOULDER] = (shoulder[0] + 0.015, shoulder[1], 0.95)
    kps[KP_R_SHOULDER] = (shoulder[0] - 0.015, shoulder[1], 0.95)
    kps[KP_L_HIP] = (hip[0] + 0.015, hip[1], 0.95)
    kps[KP_R_HIP] = (hip[0] - 0.015, hip[1], 0.95)
    if knee is not None:
        kps[KP_L_KNEE] = (knee[0] + 0.012, knee[1], 0.92)
        kps[KP_R_KNEE] = (knee[0] - 0.012, knee[1], 0.92)
    if ankle is not None:
        kps[KP_L_ANKLE] = (ankle[0] + 0.012, ankle[1], 0.92)
        kps[KP_R_ANKLE] = (ankle[0] - 0.012, ankle[1], 0.92)
    return _obs(track_id, bbox, kps)


def test_wave15_seated_grandma_133538_a690_not_standing_in_front_and_no_false_left_bed() -> None:
    """
    Волна 15 (Z1, 20261120_133538_left_bed_a690):
    1) Сидящая на матрасе бабушка #846 с вытянутыми на коврик ногами (floor_xy_y=76.5 >= 75 см,
       hip_on_mattress_y=-42.9 < -15 см, оба колена внутри bed_polygon) НЕ пробивает ветку
       floor_xy_cm[1] >= 75.0: is_standing_in_front_of_bed == False, is_person_on_bed == True.
    2) Проверка обеих веток защиты: (а) таз глубоко на матрасе без видимых коленей;
       (б) таз у самого края матраса (-6 см), но видимое колено (>= 0.45) внутри bed_polygon (< bed_max_y).
    3) В связке PatientIdentifier + FallStateMachine длительное удержание позы #846 даёт SITTING_UP
       и никогда не даёт ложный LEFT_BED.
    """
    hom = FloorHomography.from_zones(REAL_ZONES, bed_polygon=REAL_ZONES.bed_polygon)
    m = hom.measure_pose(GRANDMA_133538_A690, bed_polygon=REAL_ZONES.bed_polygon)
    assert m.floor_xy_cm is not None and m.floor_xy_cm[1] >= 75.0
    assert m.standing_on_floor_in_front is False
    mh = GRANDMA_133538_A690.mid_hip
    assert mh is not None
    hip_m = hom.image_to_plane_z_cm(mh, hom.mattress_height_cm)
    assert hip_m is not None and hip_m[1] < -15.0

    assert is_standing_in_front_of_bed(GRANDMA_133538_A690, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False
    assert is_person_on_bed(GRANDMA_133538_A690, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True
    assert are_legs_hanging_outside_bed(GRANDMA_133538_A690, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    # Вариант (а): колени скрыты пледом (conf=0.10), но таз глубоко на матрасе (hip_on_mattress_y < -15 см)
    kps_no_knees = list(GRANDMA_133538_A690.keypoints_norm)
    kps_no_knees[KP_L_KNEE] = (kps_no_knees[KP_L_KNEE][0], kps_no_knees[KP_L_KNEE][1], 0.10)
    kps_no_knees[KP_R_KNEE] = (kps_no_knees[KP_R_KNEE][0], kps_no_knees[KP_R_KNEE][1], 0.10)
    seated_hidden_knees = _obs(846, GRANDMA_133538_A690.bbox_norm, kps_no_knees)
    assert is_standing_in_front_of_bed(seated_hidden_knees, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False
    assert is_person_on_bed(seated_hidden_knees, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    # Вариант (б): таз на самом переднем крае матраса (hip_on_mattress_y > -15 см и <= 20 см),
    # но левое колено с conf >= 0.45 внутри bed_polygon (y < bed_max_y)
    _, bed_max_y = polygon_y_bounds(REAL_ZONES.bed_polygon)
    kps_edge_hip = list(GRANDMA_133538_A690.keypoints_norm)
    kps_edge_hip[KP_L_HIP] = (0.3918, 0.680, 1.0)
    kps_edge_hip[KP_R_HIP] = (0.3581, 0.680, 1.0)
    kps_edge_hip[KP_R_KNEE] = (0.3326, 0.7707, 0.20)  # только одно видимое колено (>= 0.45) внутри bed_polygon
    seated_edge_with_knee = _obs(846, GRANDMA_133538_A690.bbox_norm, kps_edge_hip)
    mh_edge = seated_edge_with_knee.mid_hip
    assert mh_edge is not None
    hip_m_edge = hom.image_to_plane_z_cm(mh_edge, hom.mattress_height_cm)
    assert hip_m_edge is not None and -15.0 <= hip_m_edge[1] <= 20.0
    kn_l = seated_edge_with_knee.kp(KP_L_KNEE, 0.45)
    assert kn_l is not None and kn_l[1] < bed_max_y
    assert is_standing_in_front_of_bed(seated_edge_with_knee, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False
    assert is_person_on_bed(seated_edge_with_knee, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    # Сквозной прогон FSM: сон -> сидение a690 в течение 6 с -> SITTING_UP, ноль LEFT_BED
    ident = PatientIdentifier(bed_lock_seconds=12.0, initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(sit_confirm_seconds=2.0, leave_confirm_seconds=2.0, alert_cooldown_seconds=45.0)
    t = 0.0
    for i in range(8):
        t += 0.3
        p = _jitter_obs(_grandma_sleep_obs(846), 0.001 * (i % 2))
        roles, cg = ident.update([p], REAL_ZONES, now=t)
        assert fsm.step([p], roles, cg, REAL_ZONES, now=t) is None
    assert roles[846].role == Role.PATIENT
    assert fsm.state == PatientState.IN_BED

    events: list[AlertEvent] = []
    for i in range(20):
        t += 0.3
        p = _jitter_obs(GRANDMA_133538_A690, 0.001 * (i % 2))
        roles, cg = ident.update([p], REAL_ZONES, now=t)
        ev = fsm.step(
            [p],
            roles,
            cg,
            REAL_ZONES,
            now=t,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )
        assert fsm.state != PatientState.LEFT_BED, f"Ложный LEFT_BED на шаге {i}"
        if ev is not None:
            events.append(ev)

    assert any(ev.state == PatientState.SITTING_UP for ev in events)
    assert not any(ev.state == PatientState.LEFT_BED for ev in events)


def test_wave15_real_walker_exit_child_standing_and_wave10_helpers_no_regression() -> None:
    """
    Не-регрессия Z1:
    1) REAL_WALKER_EXIT (Wave 13) и его наклонённый вариант (angle ≈ 30°, floor_xy_y >= 75 см,
       колени вне bed_polygon) -> is_standing_in_front_of_bed == True, is_person_on_bed == False.
    2) child_standing (test_fallguard:3031) -> is_standing_in_front_of_bed == True, is_person_on_bed == False.
    3) Ухаживающие Wave 10 (103826 глубокий наклон 65° с коленом вне кровати, 110430 на стуле
       в 60 см перед кроватью) сохраняют is_standing_in_front_of_bed == True.
    """
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

    # Вариант REAL_WALKER_EXIT с небольшим наклоном (angle ~ 30° >= 28°), который проверяет именно ветку L618
    walker_hunched_kps = list(walker_kps)
    walker_hunched_kps[5] = (0.41, 0.50, 0.95)
    walker_hunched_kps[6] = (0.37, 0.50, 0.95)
    walker_hunched = _obs(7002, (0.28, 0.41, 0.43, 0.95), walker_hunched_kps)
    ang_h = walker_hunched.torso_angle_from_vertical_deg()
    assert ang_h is not None and 28.0 <= ang_h < 45.0
    assert is_standing_in_front_of_bed(walker_hunched, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True
    assert is_person_on_bed(walker_hunched, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False

    # child_standing (точные координаты test_fallguard.py:3023-3033)
    child_standing = _simple_pose(
        track_id=667,
        bbox=(0.4971, 0.5677, 0.5547, 0.8700),
        shoulder=(0.517, 0.625),
        hip=(0.536, 0.726),
        ankle=(0.530, 0.858),
    )
    assert is_standing_in_front_of_bed(child_standing, REAL_ZONES.bed_polygon) is True
    assert is_person_on_bed(child_standing, REAL_ZONES.bed_polygon) is False
    assert is_standing_in_front_of_bed(child_standing, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    # Критические позы Wave 10, проходящие через L618-624
    for name in ("103826_sitting_up", "110430_sitting_up", "102227_sitting_up"):
        pose = CAREGIVER_FB_POSES[name]
        assert is_standing_in_front_of_bed(pose, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True, (
            f"Регрессия Wave 10 на позе {name}"
        )
        assert is_person_on_bed(pose, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False

    # Регрессия Wave 12 (130059): стоящий на прямых ногах с глубоким наклоном к матрасу (angle ~ 56.3°,
    # dy_hip_knee = 0.14 >= 0.125, лодыжки на полу y=0.89) сохраняет is_standing_in_front_of_bed == True
    bent_130059_kps = [(0.0, 0.0, 0.0)] * 17
    bent_130059_kps[0] = (0.35, 0.50, 0.88)
    bent_130059_kps[5] = (0.375, 0.54, 0.90)
    bent_130059_kps[6] = (0.395, 0.54, 0.90)
    bent_130059_kps[11] = (0.48, 0.61, 0.87)
    bent_130059_kps[12] = (0.50, 0.61, 0.87)
    bent_130059_kps[13] = (0.48, 0.75, 0.86)
    bent_130059_kps[14] = (0.50, 0.75, 0.86)
    bent_130059_kps[15] = (0.48, 0.89, 0.85)
    bent_130059_kps[16] = (0.50, 0.89, 0.85)
    bent_130059 = _obs(10, (0.33, 0.45, 0.54, 0.92), bent_130059_kps)
    assert is_standing_in_front_of_bed(bent_130059, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True
    assert is_person_on_bed(bent_130059, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False


def test_wave15_static_object_gate_no_viral_spawned_in_door_far_from_door_687d() -> None:
    """
    Волна 15 (Z1, dvr_20261119_160540_687d):
    Сиделка #766 вошла через дверь (spawned_in_door=True), дошла вглубь комнаты к креслу-туалету
    (~37% от door_polygon) и пропала. Новый трек бабушки #768, появившийся там же даже через 0.5 с,
    НЕ наследует spawned_in_door (is_spawned_in_door(768) == False), сохраняя has_moved=True.
    """
    gate = StaticObjectGate(min_displacement=0.045, ttl_seconds=60.0)
    t = 100.0

    # Шаг 1: сиделка #766 входит через дверь и проходит внутрь комнаты к креслу-туалету
    walk_positions = [
        ((0.12, 0.28, 0.24, 0.84), (0.18, 0.40), (0.18, 0.62)),
        ((0.16, 0.30, 0.28, 0.86), (0.22, 0.42), (0.22, 0.64)),
        ((0.28, 0.38, 0.42, 0.94), (0.35, 0.50), (0.35, 0.72)),
        ((0.46, 0.50, 0.60, 0.97), (0.53, 0.62), (0.53, 0.80)),
        ((0.58, 0.64, 0.73, 0.99), (0.65, 0.78), (0.65, 0.90)),
    ]
    for bbox, sh, hp in walk_positions:
        t += 0.4
        p = _simple_pose(766, bbox, sh, hp)
        gate.update_and_filter([p], zones=REAL_ZONES, now=t)

    assert gate.is_spawned_in_door(766) is True
    assert gate.is_track_moved(766) is True

    # Проверяем, что позиция бабушки у кресла-туалета действительно далеко от двери (<= -0.35)
    d_chair = polygon_distance_norm(GRANDMA_CHAIR_687D.body_center, REAL_ZONES.door_polygon)
    assert d_chair <= -0.35, f"Ожидалась дистанция <= -0.35 от двери, получено {d_chair}"

    # Шаг 2: через 0.5 с на месте пропавшего трека #766 появляется трек бабушки #768
    t += 0.5
    out = gate.update_and_filter([GRANDMA_CHAIR_687D], zones=REAL_ZONES, now=t)
    assert len(out) == 1
    assert out[0].track_id == 768
    assert gate.is_track_moved(768) is True
    assert gate.is_spawned_in_door(768) is False


def test_wave15_static_object_gate_inherits_spawned_in_door_at_door_within_5s_and_blocks_after_5s() -> None:
    """
    Волна 15 (Z1):
    1) Перескок ID гостя вблизи door_polygon (polygon_distance_norm >= -0.10) в пределах dt <= 5.0 с
       сохраняет наследование spawned_in_door=True (даже если новый бокс уже не касается door_polygon).
    2) Тот же перескок ID вблизи door_polygon, но спустя dt > 5.0 с (8.0 с) НЕ наследует spawned_in_door.
    """
    gate = StaticObjectGate(min_displacement=0.045, ttl_seconds=60.0)
    t = 200.0

    # Гость #801 заходит в дверь и делает шаг на порог (x ≈ 0.30, вне самой door_polygon, но >= -0.10)
    door_p0 = _simple_pose(801, (0.11, 0.28, 0.23, 0.84), (0.17, 0.40), (0.17, 0.62))
    door_p1 = _simple_pose(801, (0.16, 0.30, 0.27, 0.86), (0.215, 0.42), (0.215, 0.64))
    door_p2 = _simple_pose(801, (0.24, 0.32, 0.36, 0.88), (0.30, 0.45), (0.30, 0.66))

    gate.update_and_filter([door_p0], zones=REAL_ZONES, now=t)
    t += 0.3
    gate.update_and_filter([door_p1], zones=REAL_ZONES, now=t)
    t += 0.3
    gate.update_and_filter([door_p2], zones=REAL_ZONES, now=t)
    assert gate.is_spawned_in_door(801) is True

    # Перескок ID #801 -> #802 у порога через dt = 2.0 с <= 5.0 с (вне полигона двери, d >= -0.10)
    switched_near_door = _simple_pose(802, (0.26, 0.34, 0.37, 0.89), (0.31, 0.46), (0.31, 0.67))
    assert is_pose_touching_door(switched_near_door, REAL_ZONES.door_polygon, min_overlap=0.20) is False
    d_near = polygon_distance_norm(switched_near_door.body_center, REAL_ZONES.door_polygon)
    assert -0.10 <= d_near < 0.0, f"Ожидалась близость [-0.10, 0) к двери, получено {d_near}"

    t += 2.0
    out_fast = gate.update_and_filter([switched_near_door], zones=REAL_ZONES, now=t)
    assert len(out_fast) == 1
    assert out_fast[0].track_id == 802
    assert gate.is_spawned_in_door(802) is True
    assert out_fast[0].confirmed_frames >= 4
    assert out_fast[0].outside_bed_disp >= 0.05

    # Тот же порог, но спустя dt = 8.0 с > 5.0 с -> #803 не должен наследовать spawned_in_door
    switched_late = _simple_pose(803, (0.26, 0.34, 0.37, 0.89), (0.31, 0.46), (0.31, 0.67))
    t += 8.0
    out_late = gate.update_and_filter([switched_late], zones=REAL_ZONES, now=t)
    assert len(out_late) == 1
    assert out_late[0].track_id == 803
    assert gate.is_track_moved(803) is True
    assert gate.is_spawned_in_door(803) is False
