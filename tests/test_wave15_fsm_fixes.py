from __future__ import annotations

from app.geometry import (
    PoseObservation,
    are_legs_hanging_outside_bed,
    is_person_on_bed,
)
from app.patient_id import Role, TrackIdentity
from app.state_machine import AlertEvent, AlertSeverity, FallStateMachine, PatientState
from app.zones import ZonesConfig

REAL_ZONES_21 = ZonesConfig(
    bed_polygon=[
        [0.59805, 0.79135],
        [0.60234, 0.65649],
        [0.66389, 0.64504],
        [0.69395, 0.56997],
        [0.66890, 0.41858],
        [0.30964, 0.40458],
        [0.25740, 0.57888],
        [0.27815, 0.78880],
    ],
    door_polygon=[
        [0.20659, 0.19211],
        [0.26885, 0.82188],
        [0.21947, 0.99746],
        [0.18297, 0.99746],
        [0.06847, 0.26845],
    ],
    floor_polygon=[
        [0.265, 0.81],
        [0.635, 0.82],
        [0.685, 0.97],
        [0.215, 0.96],
    ],
    floor_width_cm=200.0,
    floor_depth_cm=120.0,
    mattress_height_cm=55.0,
)


def _ident(tid: int, role: Role, reason: str = "bed_sleep_lock") -> TrackIdentity:
    return TrackIdentity(track_id=tid, role=role, reason=reason, has_moved=True)


def _obs_02bb_2735(track_id: int = 2735) -> PoseObservation:
    """Реальные координаты #2735 из инцидента 20261119_181023_fallen_02bb (кресло-туалет у нижнего края)."""
    return PoseObservation(
        track_id=track_id,
        bbox_norm=(0.5430, 0.7545, 0.7368, 0.9974),
        conf=0.84,
        keypoints_norm=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.6063, 0.8549, 0.783),
            (0.6400, 0.8153, 0.480),
            (0.6346, 0.9090, 0.952),
            (0.6678, 0.8091, 0.955),
            (0.6105, 0.9987, 0.607),
            (0.6689, 0.7974, 0.652),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.6821, 0.9921, 0.805),
            (0.7010, 0.9352, 0.839),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )


def _obs_b632_18435(track_id: int = 18435) -> PoseObservation:
    """Реальные координаты #18435 из инцидента 20261120_113218_fallen_b632 (кресло-туалет без видимого таза)."""
    return PoseObservation(
        track_id=track_id,
        bbox_norm=(0.6321, 0.8014, 0.7671, 0.9952),
        conf=0.80,
        keypoints_norm=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.6772, 0.9154, 0.797),
            (0.0, 0.0, 0.0),
            (0.6601, 0.9772, 0.565),
            (0.7290, 0.9349, 0.835),
            (0.0, 0.0, 0.0),
            (0.6917, 0.8974, 0.609),
            (0.0, 0.0, 0.0),
            (0.6713, 0.8992, 0.480),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )


def _make_pose(
    track_id: int,
    bbox: tuple[float, float, float, float],
    *,
    head: tuple[float, float] | None = None,
    shoulder: tuple[float, float] | None = None,
    hip: tuple[float, float] | None = None,
    knee: tuple[float, float] | None = None,
    l_ankle: tuple[float, float, float] | None = None,
    r_ankle: tuple[float, float, float] | None = None,
    conf: float = 0.88,
) -> PoseObservation:
    kps: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 17
    if head is not None:
        kps[0] = (head[0], head[1], 0.90)
        kps[1] = (head[0] - 0.01, head[1] - 0.008, 0.85)
    if shoulder is not None:
        kps[5] = (shoulder[0] - 0.03, shoulder[1], 0.92)
        kps[6] = (shoulder[0] + 0.03, shoulder[1], 0.92)
    if hip is not None:
        kps[11] = (hip[0] - 0.025, hip[1], 0.90)
        kps[12] = (hip[0] + 0.025, hip[1], 0.90)
    if knee is not None:
        kps[13] = (knee[0] - 0.02, knee[1], 0.86)
        kps[14] = (knee[0] + 0.02, knee[1], 0.86)
    if l_ankle is not None:
        kps[15] = l_ankle
    if r_ankle is not None:
        kps[16] = r_ankle
    return PoseObservation(
        track_id=track_id,
        bbox_norm=bbox,
        conf=conf,
        keypoints_norm=tuple(kps),
    )


def test_wave15_is_lying_on_floor_rejects_cropped_chair_torsos_02bb_and_b632() -> None:
    """
    Z4 п.1: реальные позы на кресле-туалете у нижнего среза кадра (02bb #2735 и b632 #18435)
    не считаются лежанием на полу (_is_lying_on_floor == False) и не бьют _check_person_down.
    """
    obs_2735 = _obs_02bb_2735()
    obs_18435 = _obs_b632_18435()

    assert FallStateMachine._is_lying_on_floor(obs_2735, REAL_ZONES_21) is False
    assert FallStateMachine._is_lying_on_floor(obs_18435, REAL_ZONES_21) is False

    # В _check_person_down даже одиночный VISITOR на кресле-туалете не даёт ложный FALLEN за 12 с
    for obs in (obs_2735, obs_18435):
        fsm = FallStateMachine(fall_confirm_seconds=1.2)
        roles = {obs.track_id: _ident(obs.track_id, Role.VISITOR, reason="door_entry")}
        t = 0.0
        for _ in range(48):
            t += 0.25
            ev = fsm.step([obs], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
            assert ev is None
        assert fsm.state != PatientState.FALLEN

    # Если голова НЕ выше плеч у нижнего среза — защита не срабатывает, _is_lying_on_floor == True
    head_down_edge = _make_pose(
        901,
        (0.56, 0.79, 0.72, 0.995),
        head=(0.64, 0.92),
        shoulder=(0.64, 0.87),
        hip=None,
    )
    assert FallStateMachine._is_lying_on_floor(head_down_edge, REAL_ZONES_21) is True

    # Если у нижнего среза видны ноги (колени/лодыжки) — это не обрезанный торс кресла -> True
    with_legs_edge = _make_pose(
        902,
        (0.55, 0.78, 0.72, 0.995),
        head=(0.63, 0.82),
        shoulder=(0.63, 0.86),
        hip=(0.64, 0.93),
        knee=(0.65, 0.97),
    )
    assert FallStateMachine._is_lying_on_floor(with_legs_edge, REAL_ZONES_21) is True


def test_wave15_is_lying_on_floor_preserves_curled_obs_and_edge_falls() -> None:
    """
    Z4 п.1 (не-регрессия):
    - curled_obs (y2 = 0.96) остаётся True;
    - горизонтальное падение у нижнего края (y2 = 0.995, ar = 1.20) остаётся True;
    - заваленное под углом падение у нижнего края (y2 = 0.995, angle = 60°) остаётся True.
    """
    zones_curled = ZonesConfig(
        bed_polygon=[[0.30, 0.42], [0.66, 0.42], [0.62, 0.78], [0.28, 0.78]],
        door_polygon=[],
        floor_polygon=[[0.26, 0.79], [0.66, 0.79], [0.72, 0.98], [0.20, 0.98]],
    )
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
    assert FallStateMachine._is_lying_on_floor(curled_obs, zones_curled) is True

    # Горизонтальное падение у нижнего края: y2 = 0.995, ar = 1.20 (с angle=None и с angle < 48°)
    horiz_edge_no_hip = _make_pose(
        5,
        (0.45, 0.795, 0.69, 0.995),  # w=0.24, h=0.20 -> ar=1.20
        head=(0.50, 0.84),
        shoulder=(0.54, 0.87),
        hip=None,
    )
    assert abs(horiz_edge_no_hip.aspect_ratio_wh - 1.20) < 1e-3
    assert FallStateMachine._is_lying_on_floor(horiz_edge_no_hip, REAL_ZONES_21) is True

    horiz_edge_with_hip = _make_pose(
        6,
        (0.45, 0.795, 0.69, 0.995),  # w=0.24, h=0.20 -> ar=1.20
        head=(0.55, 0.83),
        shoulder=(0.56, 0.86),
        hip=(0.58, 0.95),
    )
    assert abs(horiz_edge_with_hip.aspect_ratio_wh - 1.20) < 1e-3
    assert FallStateMachine._is_lying_on_floor(horiz_edge_with_hip, REAL_ZONES_21) is True

    # Наклонённое падение у нижнего края: y2 = 0.995, angle = 60°, ar = 0.85
    # dx * (16/9) / dy = tan(60°) ≈ 1.732 -> при dy = 0.06, dx = 0.05845
    tilted_edge = _make_pose(
        7,
        (0.50, 0.755, 0.704, 0.995),  # w=0.204, h=0.24 -> ar=0.85
        head=(0.55, 0.82),
        shoulder=(0.56, 0.86),
        hip=(0.61845, 0.92),
    )
    ang = tilted_edge.torso_angle_from_vertical_deg()
    assert ang is not None and 58.0 <= ang <= 62.0
    assert FallStateMachine._is_lying_on_floor(tilted_edge, REAL_ZONES_21) is True


def test_wave15_assisted_seated_foreground_chair_calm_and_relative_slump_0085() -> None:
    """
    Z4 п.2:
    1) Посадка на кресло ближнего плана (y2 >= 0.985, вертикально, голова выше плеч, upper_y >= 0.66)
       при родственнике взводит assisted_seated_on_chair (как с видимым тазом, так и с angle=None);
       после ухода родственника бабушка спокойно сидит без ложного FALLEN/LEFT_BED.
    2) Относительное сползание на >= +0.085 по upper_y штатно поднимает критическую тревогу FALLEN.
    3) Завал головы ниже плеч (head_y >= shoulder_y) на кресле также поднимает FALLEN.
    """
    # Сценарий А: кресло ближнего плана с видимым тазом (уровень upper_y = 0.79, angle ~ 28°)
    fsm_a = FallStateMachine(sit_confirm_seconds=1.2, leave_confirm_seconds=1.5, fall_confirm_seconds=1.0)
    roles = {1: _ident(1, Role.PATIENT)}
    t = 10.0

    fg_chair_calm = _make_pose(
        1,
        (0.55, 0.73, 0.73, 0.997),
        head=(0.63, 0.76),
        shoulder=(0.64, 0.79),
        hip=(0.67, 0.89),
    )
    for _ in range(6):
        t += 0.25
        assert fsm_a.step([fg_chair_calm], roles, caregiver_nearby=True, zones=REAL_ZONES_21, now=t) is None
    assert fsm_a._assisted_seated_latched is True
    assert fsm_a._assisted_seated_on_chair is True

    # Родственник ушёл — бабушка спокойно сидит (и слегка шевелится +0.03 < +0.085) 25 секунд: 0 тревог!
    fg_chair_minor_shift = _make_pose(
        1,
        (0.55, 0.76, 0.73, 0.997),
        head=(0.63, 0.79),
        shoulder=(0.64, 0.82),
        hip=(0.67, 0.92),
    )
    for idx in range(50):
        t += 0.5
        pose = fg_chair_calm if idx % 2 == 0 else fg_chair_minor_shift
        ev = fsm_a.step([pose], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
        assert ev is None
        assert fsm_a.state == PatientState.IN_BED
        assert fsm_a.status_reason == "сидит после ухода родственника"

    # Сползание вниз на +0.09 (>= +0.085) при всё ещё вертикальном торсе -> FALLEN!
    fg_chair_slumped = _make_pose(
        1,
        (0.55, 0.81, 0.73, 0.997),
        head=(0.63, 0.845),
        shoulder=(0.64, 0.88),  # 0.88 - 0.79 = +0.09 >= +0.085
        hip=(0.67, 0.98),
    )
    fall_ev: AlertEvent | None = None
    for _ in range(10):
        t += 0.25
        ev = fsm_a.step([fg_chair_slumped], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
        if ev is not None:
            fall_ev = ev
            break
    assert fall_ev is not None
    assert fall_ev.state == PatientState.FALLEN
    assert fall_ev.severity == AlertSeverity.CRITICAL

    # Сценарий Б: кресло ближнего плана без видимого таза (angle=None, как #18435)
    fsm_b = FallStateMachine(sit_confirm_seconds=1.2, leave_confirm_seconds=1.5, fall_confirm_seconds=1.0)
    t = 100.0
    fg_no_hip = _make_pose(
        1,
        (0.62, 0.79, 0.76, 0.995),
        head=(0.68, 0.84),
        shoulder=(0.69, 0.88),
        hip=None,
    )
    assert fg_no_hip.torso_angle_from_vertical_deg() is None
    for _ in range(6):
        t += 0.25
        assert fsm_b.step([fg_no_hip], roles, caregiver_nearby=True, zones=REAL_ZONES_21, now=t) is None
    for _ in range(30):
        t += 0.5
        assert fsm_b.step([fg_no_hip], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t) is None
        assert fsm_b.state == PatientState.IN_BED
        assert fsm_b.status_reason == "сидит после ухода родственника"

    # Завал головы ниже плеч (head_y >= shoulder_y) -> мгновенно снимает поблажку кресла и бьёт FALLEN
    fg_head_dropped = _make_pose(
        1,
        (0.62, 0.81, 0.76, 0.995),
        head=(0.68, 0.92),
        shoulder=(0.69, 0.89),
        hip=None,
    )
    fall_ev_b: AlertEvent | None = None
    for _ in range(10):
        t += 0.25
        ev = fsm_b.step([fg_head_dropped], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
        if ev is not None:
            fall_ev_b = ev
            break
    assert fall_ev_b is not None
    assert fall_ev_b.state == PatientState.FALLEN


def test_wave15_solo_patient_at_bottom_edge_keeps_L1071_fallen_protection() -> None:
    """
    Отсечённый п.6 из #228: НЕ отключать state_machine.py:1071 для одинокой PATIENT у нижнего среза!
    Без защёлки assisted_seated_on_chair позы 02bb (#2735) и b632 (#18435) как Role.PATIENT
    обязаны по-прежнему переводить FSM в FALLEN.
    """
    for obs in (_obs_02bb_2735(track_id=1), _obs_b632_18435(track_id=1)):
        fsm = FallStateMachine(fall_confirm_seconds=1.0)
        roles = {1: _ident(1, Role.PATIENT)}
        t = 0.0
        fall_ev: AlertEvent | None = None
        for _ in range(12):
            t += 0.25
            ev = fsm.step([obs], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
            if ev is not None:
                fall_ev = ev
                break
        assert fall_ev is not None
        assert fall_ev.state == PatientState.FALLEN
        assert fall_ev.severity == AlertSeverity.CRITICAL
        assert fsm.state == PatientState.FALLEN


def test_wave15_sitting_up_sway_45_to_68_deg_single_alert_and_rearm_on_legs_back() -> None:
    """
    Z4 п.3:
    1) Покачивание сидящей на кровати бабушки (ang 45° <-> 68°) при стопах на полу
       (уверенная лодыжка conf >= 0.45 ниже bed_max_y + 0.02 вне bed_polygon)
       удерживает SITTING_UP и даёт ровно 1 алерт без стробоскопа даже дольше cooldown (45 с).
    2) Возврат ног на матрас на >= 3.5 с возвращает состояние в IN_BED и реармит следующий SITTING_UP.
    3) Неуверенная лодыжка (conf < 0.45) не удерживает SITTING_UP.
    """
    fsm = FallStateMachine(
        sit_confirm_seconds=1.5,
        leave_confirm_seconds=2.0,
        alert_cooldown_seconds=45.0,
    )
    roles = {1: _ident(1, Role.PATIENT)}

    # Сон в кровати (ноги на матрасе, горизонтально)
    sleep_pose = _make_pose(
        1,
        (0.32, 0.48, 0.62, 0.72),
        head=(0.58, 0.52),
        shoulder=(0.54, 0.54),
        hip=(0.41, 0.58),
        knee=(0.38, 0.66),
        l_ankle=(0.35, 0.71, 0.85),
        r_ankle=(0.37, 0.70, 0.85),
    )
    assert is_person_on_bed(sleep_pose, REAL_ZONES_21.bed_polygon, zones=REAL_ZONES_21) is True
    assert are_legs_hanging_outside_bed(sleep_pose, REAL_ZONES_21.bed_polygon, zones=REAL_ZONES_21) is False

    # Посадка на краю кровати, ang ≈ 45°, стопы на полу (y = 0.84 >= 0.79135 + 0.02 = 0.81135)
    # dx * (16/9) / dy = 1.0 -> при dy = 0.10, dx = 0.05625
    sit_45 = _make_pose(
        1,
        (0.36, 0.45, 0.58, 0.88),
        head=(0.53, 0.48),
        shoulder=(0.50625, 0.52),
        hip=(0.45, 0.62),
        knee=(0.44, 0.73),
        l_ankle=(0.43, 0.84, 0.86),
        r_ankle=(0.46, 0.835, 0.85),
    )
    ang_45 = sit_45.torso_angle_from_vertical_deg()
    assert ang_45 is not None and 43.0 <= ang_45 <= 47.0
    assert are_legs_hanging_outside_bed(sit_45, REAL_ZONES_21.bed_polygon, zones=REAL_ZONES_21) is True

    # Откинулась назад, ang ≈ 68°, но стопы остались на полу (are_legs_hanging == False из-за >= 58°!)
    # dx * (16/9) / dy = tan(68°) ≈ 2.475 -> при dy = 0.06, dx = 0.0835
    lean_68_feet_on_floor = _make_pose(
        1,
        (0.35, 0.48, 0.60, 0.88),
        head=(0.57, 0.51),
        shoulder=(0.5335, 0.55),
        hip=(0.45, 0.61),
        knee=(0.44, 0.72),
        l_ankle=(0.43, 0.84, 0.84),
        r_ankle=(0.46, 0.835, 0.82),
    )
    ang_68 = lean_68_feet_on_floor.torso_angle_from_vertical_deg()
    assert ang_68 is not None and 66.0 <= ang_68 <= 70.0
    assert is_person_on_bed(lean_68_feet_on_floor, REAL_ZONES_21.bed_polygon, zones=REAL_ZONES_21) is True
    assert are_legs_hanging_outside_bed(lean_68_feet_on_floor, REAL_ZONES_21.bed_polygon, zones=REAL_ZONES_21) is False

    # Также проверяем вариант с ОДНОЙ уверенной лодыжкой (conf = 0.50 >= 0.45)
    lean_68_one_ankle = _make_pose(
        1,
        (0.35, 0.48, 0.60, 0.88),
        head=(0.57, 0.51),
        shoulder=(0.5335, 0.55),
        hip=(0.45, 0.61),
        knee=(0.44, 0.72),
        l_ankle=(0.43, 0.835, 0.50),
        r_ankle=(0.0, 0.0, 0.0),
    )

    t = 0.0
    # 1. Калибруем сон
    for _ in range(8):
        t += 0.5
        assert fsm.step([sleep_pose], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t) is None
    assert fsm.state == PatientState.IN_BED

    # 2. Первая посадка (ang 45°) -> ровно 1 алерт SITTING_UP
    sitting_events: list[AlertEvent] = []
    for _ in range(8):
        t += 0.3
        ev = fsm.step([sit_45], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
        if ev is not None:
            sitting_events.append(ev)
    assert len(sitting_events) == 1
    assert sitting_events[0].state == PatientState.SITTING_UP
    assert fsm.state == PatientState.SITTING_UP

    # 3. Несколько циклов покачивания 68° <-> 45° суммарно > 120 с (гораздо дольше cooldown 45 с!)
    for cycle in range(3):
        lean_pose = lean_68_one_ankle if cycle == 1 else lean_68_feet_on_floor
        # Откидывается назад на 25 секунд (намного дольше 3.5 с!)
        for _ in range(50):
            t += 0.5
            ev = fsm.step([lean_pose], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
            assert ev is None
            assert fsm.state == PatientState.SITTING_UP
        # Выпрямляется обратно в 45° на 25 секунд -> НЕ должно быть второго алерта!
        for _ in range(50):
            t += 0.5
            ev = fsm.step([sit_45], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
            assert ev is None
            assert fsm.state == PatientState.SITTING_UP

    assert len(sitting_events) == 1

    # 4. Убирает ноги обратно на матрас: за 3.0 с (< 3.5 с) ещё SITTING_UP, за >= 3.5 с -> IN_BED
    for _ in range(6):
        t += 0.5
        assert fsm.step([sleep_pose], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t) is None
    assert fsm.state == PatientState.SITTING_UP

    for _ in range(2):
        t += 0.5
        assert fsm.step([sleep_pose], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t) is None
    assert fsm.state == PatientState.IN_BED

    # 5. После возврата в IN_BED и прохождения cooldown новая посадка штатно реармит SITTING_UP!
    for _ in range(100):
        t += 0.5
        assert fsm.step([sleep_pose], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t) is None
    rearmed_events: list[AlertEvent] = []
    for _ in range(10):
        t += 0.3
        ev = fsm.step([sit_45], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t)
        if ev is not None:
            rearmed_events.append(ev)
    assert len(rearmed_events) == 1
    assert rearmed_events[0].state == PatientState.SITTING_UP

    # 6. При неуверенной лодыжке (conf = 0.35 < 0.45) состояние за >= 3.5 с штатно возвращается в IN_BED
    low_conf_ankle = _make_pose(
        1,
        (0.35, 0.48, 0.60, 0.88),
        head=(0.57, 0.51),
        shoulder=(0.5335, 0.55),
        hip=(0.45, 0.61),
        knee=(0.44, 0.72),
        l_ankle=(0.43, 0.84, 0.35),
        r_ankle=(0.46, 0.835, 0.30),
    )
    for _ in range(10):
        t += 0.5
        assert fsm.step([low_conf_ankle], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t) is None
    assert fsm.state == PatientState.IN_BED


def test_wave15_side_sleep_dangling_from_in_bed_stays_silent() -> None:
    """
    Не-регрессия Wave 13: если бабушка уже в IN_BED (не в SITTING_UP) и спит на боку
    со свешенными голенями (ang >= 60°), гистерезис SITTING_UP -> IN_BED не вмешивается.
    """
    fsm = FallStateMachine(sit_confirm_seconds=1.5)
    roles = {1: _ident(1, Role.PATIENT)}
    side_sleep = _make_pose(
        1,
        (0.32, 0.46, 0.63, 0.85),
        head=(0.59, 0.51),
        shoulder=(0.55, 0.53),
        hip=(0.43, 0.58),
        knee=(0.365, 0.685),
        l_ankle=(0.35, 0.835, 0.85),
        r_ankle=(0.34, 0.825, 0.84),
    )
    t = 0.0
    for _ in range(24):
        t += 0.5
        ev = fsm.step([side_sleep], roles, caregiver_nearby=False, zones=REAL_ZONES_21, now=t, local_hour=2)
        assert ev is None
        assert fsm.state == PatientState.IN_BED
