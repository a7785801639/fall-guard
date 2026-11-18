from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from app.detector import PoseTracker, render_annotated_frame
from app.geometry import (
    KeypointSmoother,
    OpticalKeypointStabilizer,
    PoseObservation,
    StaticObjectGate,
    are_legs_hanging_outside_bed,
    filter_valid_poses,
    is_blanket_sleeper_in_bed,
    is_hand_raised_for_help,
    is_person_on_bed,
    is_standing_in_front_of_bed,
    sanitize_collapsed_bed_keypoints,
)
from app.homography import DEFAULT_FLOOR_POLYGON_STD, FloorHomography, FootFloorMeasurement
from app.patient_id import (
    DEFAULT_DOOR_VISIT_IDLE_TIMEOUT,
    FaceRegistry,
    PatientIdentifier,
    Role,
    TrackIdentity,
    compute_night_ir_appearance,
    compute_torso_appearance,
    cosine_sim,
    is_night_ir_frame,
)
from app.state_machine import AlertSeverity, FallStateMachine, PatientState
from app.zones import ZonesConfig

# Полигон кровати в центре кадра: x=[0.20..0.75], y=[0.30..0.70]
BED_ZONE = ZonesConfig(
    bed_polygon=[[0.20, 0.30], [0.75, 0.30], [0.75, 0.70], [0.20, 0.70]],
    door_polygon=[[0.86, 0.10], [0.99, 0.10], [0.99, 0.85], [0.86, 0.85]],
)


@dataclass(frozen=True)
class _ZonesWithIgnoreProxy:
    """Совместимая обёртка на случай параллельной сборки до/после расширения ZonesConfig."""

    bed_polygon: list[list[float]]
    door_polygon: list[list[float]]
    ignore_polygons: list[list[list[float]]]

    @property
    def has_bed(self) -> bool:
        return len(self.bed_polygon) >= 3

    @property
    def has_door(self) -> bool:
        return len(self.door_polygon) >= 3


def make_zones_with_ignore(
    bed_polygon: list[list[float]],
    door_polygon: list[list[float]],
    ignore_polygons: list[list[list[float]]],
) -> ZonesConfig | _ZonesWithIgnoreProxy:
    if "ignore_polygons" in getattr(ZonesConfig, "__dataclass_fields__", {}):
        return ZonesConfig(  # type: ignore[call-arg]
            bed_polygon=bed_polygon,
            door_polygon=door_polygon,
            ignore_polygons=ignore_polygons,
        )
    return _ZonesWithIgnoreProxy(
        bed_polygon=bed_polygon,
        door_polygon=door_polygon,
        ignore_polygons=ignore_polygons,
    )


def make_pose(
    track_id: int,
    bbox: tuple[float, float, float, float],
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    ankle: tuple[float, float] | None = None,
    nose: tuple[float, float] | None = None,
    knee: tuple[float, float] | None = None,
    headless: bool = False,
    conf: float = 0.92,
) -> PoseObservation:
    kps = [(0.0, 0.0, 0.0)] * 17
    if not headless:
        eff_nose = nose if nose is not None else (shoulder[0], max(0.02, shoulder[1] - 0.05))
        kps[0] = (eff_nose[0], eff_nose[1], 0.90)
        kps[1] = (max(0.01, eff_nose[0] - 0.01), eff_nose[1], 0.88)
        kps[2] = (min(0.99, eff_nose[0] + 0.01), eff_nose[1], 0.88)
    kps[5] = (shoulder[0] - 0.02, shoulder[1], 0.92)
    kps[6] = (shoulder[0] + 0.02, shoulder[1], 0.92)
    kps[11] = (hip[0] - 0.02, hip[1], 0.92)
    kps[12] = (hip[0] + 0.02, hip[1], 0.92)
    if knee:
        kps[13] = (knee[0] - 0.01, knee[1], 0.90)
        kps[14] = (knee[0] + 0.01, knee[1], 0.90)
    if ankle:
        kps[15] = (ankle[0], ankle[1], 0.90)
        kps[16] = (ankle[0] + 0.02, ankle[1], 0.90)
    return PoseObservation(
        track_id=track_id,
        bbox_norm=bbox,
        conf=conf,
        keypoints_norm=tuple(kps),
    )


def make_colored_room_frame(
    persons: list[tuple[PoseObservation, tuple[int, int, int]]],
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    """Создаёт синтетический BGR-кадр комнаты, закрашивая области каждого человека цветом его одежды."""
    frame = np.full((height, width, 3), (38, 42, 46), dtype=np.uint8)
    for obs, bgr_color in persons:
        x1 = int(max(0, obs.bbox_norm[0] * width))
        y1 = int(max(0, obs.bbox_norm[1] * height))
        x2 = int(min(width, obs.bbox_norm[2] * width))
        y2 = int(min(height, obs.bbox_norm[3] * height))
        frame[y1:y2, x1:x2] = bgr_color
        if (y2 - y1) > 4 and (x2 - x1) > 4:
            grad = np.linspace(0.85, 1.15, y2 - y1, dtype=np.float32)[:, None, None]
            patch = np.clip(frame[y1:y2, x1:x2].astype(np.float32) * grad, 0, 255).astype(np.uint8)
            frame[y1:y2, x1:x2] = patch
    return frame


# 1. Сон на боку и повороты в кровати (боковой ракурс)
def test_sleeping_on_side_and_turning_in_bed_triggers_no_alert() -> None:
    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=2.0)
    fsm = FallStateMachine(sit_confirm_seconds=1.5, fall_confirm_seconds=1.5)

    t = 0.0
    for step_idx in range(30):
        t += 0.2
        dy = 0.01 if (step_idx % 6) < 3 else -0.01
        obs = make_pose(
            track_id=1,
            bbox=(0.28, 0.42, 0.68, 0.58),
            shoulder=(0.35, 0.49 + dy),
            hip=(0.55, 0.50),
            ankle=(0.66, 0.51),
        )
        roles, cg = ident.update([obs], BED_ZONE, now=t)
        ev = fsm.step([obs], roles, cg, BED_ZONE, now=t)
        assert ev is None

    assert roles[1].role == Role.PATIENT
    assert fsm.state == PatientState.IN_BED
    assert fsm.baseline_in_bed["calibrated"] is True


# 2. Торцевой / косой ракурс: бабушка спит ВДОЛЬ камеры (головой к изголовью, ногами к камере)
def test_end_on_foreshortened_bed_angle_self_calibrates_and_catches_real_sit_up() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(sit_confirm_seconds=1.2)

    t = 0.0
    for i in range(25):
        t += 0.2
        wobble = 0.008 if (i % 2 == 0) else -0.008
        obs_end_on_sleep = make_pose(
            track_id=1,
            bbox=(0.39, 0.37, 0.57, 0.65),
            shoulder=(0.48 + wobble, 0.42),
            hip=(0.48, 0.52),
            ankle=(0.48, 0.63),
        )
        roles, cg = ident.update([obs_end_on_sleep], BED_ZONE, now=t)
        ev = fsm.step([obs_end_on_sleep], roles, cg, BED_ZONE, now=t)
        assert ev is None, "Сон вдоль оптической оси камеры НЕ должен давать ложную тревогу посадки!"

    assert fsm.state == PatientState.IN_BED
    assert fsm.baseline_in_bed["calibrated"] is True

    sit_event = None
    for _ in range(10):
        t += 0.2
        obs_end_on_sit = make_pose(
            track_id=1,
            bbox=(0.39, 0.26, 0.57, 0.65),
            shoulder=(0.48, 0.31),
            hip=(0.48, 0.52),
            ankle=(0.48, 0.63),
        )
        roles, cg = ident.update([obs_end_on_sit], BED_ZONE, now=t)
        ev = fsm.step([obs_end_on_sit], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            sit_event = ev
            break

    assert sit_event is not None
    assert sit_event.state == PatientState.SITTING_UP
    assert sit_event.severity == AlertSeverity.WARNING


# 3. Посадка в кровати без свешивания ног (сутулая поза под косым углом камеры ~48°)
def test_sitting_up_in_bed_without_hanging_legs_detected_via_baseline_lift() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(sit_confirm_seconds=1.0)

    t = 0.0
    for _ in range(10):
        t += 0.2
        sleep_obs = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.36, 0.49), (0.56, 0.50), ankle=(0.66, 0.51))
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t)
        assert fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t) is None

    fired = None
    for _ in range(10):
        t += 0.2
        sit_in_bed = make_pose(
            track_id=1,
            bbox=(0.38, 0.31, 0.66, 0.58),
            shoulder=(0.46, 0.38),
            hip=(0.53, 0.50),
            ankle=(0.65, 0.52),
        )
        roles, cg = ident.update([sit_in_bed], BED_ZONE, now=t)
        ev = fsm.step([sit_in_bed], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            fired = ev
            break

    assert fired is not None
    assert fired.state == PatientState.SITTING_UP


# 4. Свешивание ног за край кровати (раннее предупреждение до попытки встать)
def test_hanging_legs_over_bed_edge_triggers_early_warning() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(sit_confirm_seconds=1.2)

    t = 0.0
    for _ in range(10):
        t += 0.2
        obs_sleep = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([obs_sleep], BED_ZONE, now=t)
        assert fsm.step([obs_sleep], roles, cg, BED_ZONE, now=t) is None

    fired = None
    for _ in range(10):
        t += 0.2
        obs_hang = make_pose(
            track_id=1,
            bbox=(0.40, 0.30, 0.56, 0.80),
            shoulder=(0.48, 0.36),
            hip=(0.48, 0.56),
            ankle=(0.49, 0.78),
        )
        roles, cg = ident.update([obs_hang], BED_ZONE, now=t)
        ev = fsm.step([obs_hang], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            fired = ev
            break

    assert fired is not None
    assert fired.state == PatientState.SITTING_UP
    assert fired.severity == AlertSeverity.WARNING


# 5. Медленное сползание с кровати на пол на колени/ягодицы (SLIDING / SLOW FALL после инсульта)
def test_slow_sliding_from_bed_onto_floor_triggers_critical_fall() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(fall_confirm_seconds=1.6)

    t = 0.0
    for _ in range(10):
        t += 0.25
        sleep_obs = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t)
        fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t)

    slide_steps = 16
    for step in range(1, slide_steps + 1):
        t += 0.25
        frac = step / float(slide_steps)
        sy = 0.49 + frac * (0.65 - 0.49)
        hy = 0.50 + frac * (0.78 - 0.50)
        sliding_obs = make_pose(
            track_id=1,
            bbox=(0.40, sy - 0.06, 0.58, hy + 0.14),
            shoulder=(0.49, sy),
            hip=(0.50, hy),
            knee=(0.51, min(0.95, hy + 0.10)),
        )
        roles, cg = ident.update([sliding_obs], BED_ZONE, now=t)
        fsm.step([sliding_obs], roles, cg, BED_ZONE, now=t)

    critical_ev = None
    for _ in range(10):
        t += 0.25
        slumped_floor = make_pose(
            track_id=1,
            bbox=(0.40, 0.59, 0.58, 0.92),
            shoulder=(0.49, 0.65),
            hip=(0.50, 0.78),
            knee=(0.51, 0.88),
        )
        roles, cg = ident.update([slumped_floor], BED_ZONE, now=t)
        ev = fsm.step([slumped_floor], roles, cg, BED_ZONE, now=t)
        if ev is not None and ev.state == PatientState.FALLEN:
            critical_ev = ev
            break

    assert fsm.state == PatientState.FALLEN
    assert critical_ev is not None
    assert critical_ev.severity == AlertSeverity.CRITICAL


# 6. Затяжное сидение/лежание на полу у самой кромки кровати (при частичном касании bed_polygon)
def test_slumping_at_bed_edge_with_partial_bed_polygon_overlap_triggers_fall() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(fall_confirm_seconds=1.4)

    t = 0.0
    for _ in range(10):
        t += 0.25
        sleep_obs = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t)
        fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t)

    fall_ev = None
    for _ in range(10):
        t += 0.25
        edge_slump = make_pose(
            track_id=1,
            bbox=(0.38, 0.58, 0.64, 0.88),
            shoulder=(0.48, 0.63),
            hip=(0.50, 0.69),
            ankle=(0.52, 0.86),
        )
        roles, cg = ident.update([edge_slump], BED_ZONE, now=t)
        ev = fsm.step([edge_slump], roles, cg, BED_ZONE, now=t)
        if ev is not None and ev.state == PatientState.FALLEN:
            fall_ev = ev
            break

    assert fall_ev is not None
    assert fall_ev.state == PatientState.FALLEN
    assert fall_ev.severity == AlertSeverity.CRITICAL


# 7. Резкое падение стоя вне кровати (быстрое срабатывание + флаг повтора тревоги)
def test_rapid_standing_fall_outside_bed_triggers_fast_critical_alert() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(fall_confirm_seconds=1.6)

    t = 0.0
    for _ in range(8):
        t += 0.25
        grandma = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([grandma], BED_ZONE, now=t)
        fsm.step([grandma], roles, cg, BED_ZONE, now=t)

    t += 0.2
    standing = make_pose(1, (0.42, 0.32, 0.56, 0.82), (0.49, 0.38), (0.49, 0.58))
    roles, cg = ident.update([standing], BED_ZONE, now=t)
    fsm.step([standing], roles, cg, BED_ZONE, now=t)

    fall_event = None
    elapsed_fall = 0.0
    for _ in range(8):
        t += 0.2
        elapsed_fall += 0.2
        fallen = make_pose(1, (0.32, 0.76, 0.66, 0.94), (0.38, 0.85), (0.58, 0.86))
        roles, cg = ident.update([fallen], BED_ZONE, now=t)
        ev = fsm.step([fallen], roles, cg, BED_ZONE, now=t)
        if ev is not None and ev.state == PatientState.FALLEN:
            fall_event = ev
            break

    assert fall_event is not None
    assert fall_event.severity == AlertSeverity.CRITICAL
    assert elapsed_fall <= 1.2
    assert fsm.fall_unacknowledged is True


# 8. Вход родственника из двери и наклон за предметом на полу — полный игнор
def test_visitor_entering_and_crouching_on_floor_is_ignored() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(fall_confirm_seconds=1.0)

    t = 0.0
    for _ in range(8):
        t += 0.25
        grandma = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([grandma], BED_ZONE, now=t)
        assert fsm.step([grandma], roles, cg, BED_ZONE, now=t) is None

    # Родственник входит через дверь и делает шаг внутрь комнаты (живое смещение >= 0.05)
    t += 0.25
    grandma = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
    visitor_door = make_pose(2, (0.87, 0.20, 0.98, 0.80), (0.92, 0.30), (0.92, 0.52))
    ident.update([grandma, visitor_door], BED_ZONE, now=t)

    t += 0.25
    visitor_step = make_pose(2, (0.76, 0.22, 0.88, 0.82), (0.82, 0.31), (0.82, 0.53))
    roles, _ = ident.update([grandma, visitor_step], BED_ZONE, now=t)
    assert roles[2].role == Role.VISITOR

    for _ in range(12):
        t += 0.25
        visitor_crouch = make_pose(
            track_id=2,
            bbox=(0.30, 0.76, 0.65, 0.94),
            shoulder=(0.36, 0.85),
            hip=(0.56, 0.86),
        )
        roles, cg = ident.update([grandma, visitor_crouch], BED_ZONE, now=t)
        ev = fsm.step([grandma, visitor_crouch], roles, cg, BED_ZONE, now=t)
        assert roles[2].role == Role.VISITOR
        assert ev is None


# 9. Помощь родственника у кровати подавляет ложные тревоги (включая FALLEN) при включённом suppress_sitting_when_caregiver,
#    и пропускает критическое FALLEN, когда подавление при помощнике явно выключено (suppress_sitting_when_caregiver=False)
def test_caregiver_helping_grandma_suppresses_sitting_but_still_alerts_on_fall() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(sit_confirm_seconds=1.0, fall_confirm_seconds=1.0, suppress_sitting_when_caregiver=True)

    t = 0.0
    for _ in range(8):
        t += 0.25
        grandma = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([grandma], BED_ZONE, now=t)
        fsm.step([grandma], roles, cg, BED_ZONE, now=t)

    # Родственник подходит от входа к кровати (живой трек с траекторией движения)
    t += 0.25
    helper_enter = make_pose(2, (0.82, 0.24, 0.96, 0.84), (0.89, 0.34), (0.89, 0.57))
    ident.update([grandma, helper_enter], BED_ZONE, now=t)

    for _ in range(8):
        t += 0.25
        grandma_sit = make_pose(1, (0.40, 0.30, 0.56, 0.75), (0.48, 0.36), (0.48, 0.56), ankle=(0.49, 0.76))
        helper = make_pose(2, (0.50, 0.25, 0.66, 0.85), (0.58, 0.35), (0.58, 0.58))
        roles, cg = ident.update([grandma_sit, helper], BED_ZONE, now=t)
        ev = fsm.step([grandma_sit, helper], roles, cg, BED_ZONE, now=t)
        assert cg is True
        assert ev is None
        assert fsm.state == PatientState.CAREGIVER_ASSIST

    # Пока родственник рядом и suppress_sitting_when_caregiver=True — ложное FALLEN подавляется (CAREGIVER_ASSIST)
    for _ in range(8):
        t += 0.25
        grandma_fallen = make_pose(1, (0.28, 0.76, 0.64, 0.94), (0.34, 0.86), (0.56, 0.87))
        helper = make_pose(2, (0.52, 0.28, 0.68, 0.86), (0.60, 0.38), (0.60, 0.60))
        roles, cg = ident.update([grandma_fallen, helper], BED_ZONE, now=t)
        ev = fsm.step([grandma_fallen, helper], roles, cg, BED_ZONE, now=t)
        assert cg is True
        assert ev is None
        assert fsm.state == PatientState.CAREGIVER_ASSIST

    # А если подавление при помощнике отключено (suppress_sitting_when_caregiver=False) — FALLEN штатно срабатывает
    fsm_strict = FallStateMachine(
        sit_confirm_seconds=1.0,
        fall_confirm_seconds=1.0,
        suppress_sitting_when_caregiver=False,
    )
    fall_ev = None
    for _ in range(8):
        t += 0.25
        grandma_fallen = make_pose(1, (0.28, 0.76, 0.64, 0.94), (0.34, 0.86), (0.56, 0.87))
        helper = make_pose(2, (0.52, 0.28, 0.68, 0.86), (0.60, 0.38), (0.60, 0.60))
        roles, cg = ident.update([grandma_fallen, helper], BED_ZONE, now=t)
        ev = fsm_strict.step([grandma_fallen, helper], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            fall_ev = ev
            break

    assert fall_ev is not None
    assert fall_ev.state == PatientState.FALLEN


# 10. Перекрытие двух треков у кровати и перескок ID (ByteTrack ID switch recovery)
def test_id_switch_recovery_when_visitor_occludes_patient_and_swaps_track_ids() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(leave_confirm_seconds=1.0)

    grandma_color = (130, 70, 210)
    visitor_color = (200, 170, 30)

    t = 0.0
    for _ in range(8):
        t += 0.25
        g_obs = make_pose(1, (0.28, 0.40, 0.66, 0.60), (0.36, 0.49), (0.55, 0.50))
        frame = make_colored_room_frame([(g_obs, grandma_color)])
        roles, cg = ident.update([g_obs], BED_ZONE, now=t, frame_bgr=frame)
        fsm.step([g_obs], roles, cg, BED_ZONE, now=t)

    assert roles[1].role == Role.PATIENT
    assert ident.has_appearance_anchor is True

    # Родственник входит в дверь и шагает к кровати
    t += 0.5
    g_obs = make_pose(1, (0.28, 0.40, 0.66, 0.60), (0.36, 0.49), (0.55, 0.50))
    v_door = make_pose(2, (0.86, 0.18, 0.98, 0.82), (0.92, 0.28), (0.92, 0.52))
    ident.update([g_obs, v_door], BED_ZONE, now=t, frame_bgr=make_colored_room_frame([(g_obs, grandma_color), (v_door, visitor_color)]))

    t += 0.5
    v_walk = make_pose(2, (0.72, 0.20, 0.85, 0.82), (0.78, 0.30), (0.78, 0.53))
    roles, _ = ident.update([g_obs, v_walk], BED_ZONE, now=t, frame_bgr=make_colored_room_frame([(g_obs, grandma_color), (v_walk, visitor_color)]))
    assert roles[2].role == Role.VISITOR

    # После перекрытия у кровати (>= 3.0 с визита) ByteTrack поменял местами track_id=1 и track_id=2 на выходе в дверь
    t += 3.2
    swapped_grandma_in_bed = make_pose(2, (0.28, 0.40, 0.66, 0.60), (0.36, 0.49), (0.55, 0.50))
    swapped_visitor_leaving = make_pose(1, (0.80, 0.20, 0.95, 0.84), (0.87, 0.30), (0.87, 0.55))
    frame_swapped = make_colored_room_frame(
        [(swapped_grandma_in_bed, grandma_color), (swapped_visitor_leaving, visitor_color)]
    )

    roles_after, cg_after = ident.update(
        [swapped_grandma_in_bed, swapped_visitor_leaving],
        BED_ZONE,
        now=t,
        frame_bgr=frame_swapped,
    )
    ev = fsm.step([swapped_grandma_in_bed, swapped_visitor_leaving], roles_after, cg_after, BED_ZONE, now=t)

    assert roles_after[2].role == Role.PATIENT
    assert roles_after[1].role == Role.VISITOR
    assert ident.door_visit_active is True
    assert ev is None
    assert fsm.state == PatientState.CAREGIVER_ASSIST

    # Гость вышел за дверь (>= 2.5 с нет гостей в кадре) -> дверной визит закрыт, статус возвращается в IN_BED
    t += 2.8
    frame_alone = make_colored_room_frame([(swapped_grandma_in_bed, grandma_color)])
    roles_end, cg_end = ident.update([swapped_grandma_in_bed], BED_ZONE, now=t, frame_bgr=frame_alone)
    assert ident.door_visit_active is False
    assert cg_end is False
    assert fsm.step([swapped_grandma_in_bed], roles_end, cg_end, BED_ZONE, now=t) is None
    assert fsm.state == PatientState.IN_BED


# 11. Сохранение и загрузка AppearanceAnchor на диск + проверка срока жизни 12 часов
def test_appearance_anchor_persistence_and_12h_ttl(tmp_path: Path) -> None:
    anchor_file = tmp_path / "patient_anchor.json"
    fake_wall = [1_700_000_000.0]

    ident1 = PatientIdentifier(
        bed_lock_seconds=20.0,
        initial_bed_lock_seconds=1.0,
        state_path=anchor_file,
        wall_clock=lambda: fake_wall[0],
    )
    grandma_color = (60, 180, 220)
    t = 0.0
    for _ in range(6):
        t += 0.25
        obs = make_pose(1, (0.30, 0.40, 0.65, 0.60), (0.38, 0.48), (0.55, 0.50))
        frame = make_colored_room_frame([(obs, grandma_color)])
        roles, _ = ident1.update([obs], BED_ZONE, now=t, frame_bgr=frame)

    assert roles[1].role == Role.PATIENT
    assert anchor_file.exists()

    fake_wall[0] += 2.0 * 3600.0
    ident_restarted = PatientIdentifier(
        bed_lock_seconds=25.0,
        initial_bed_lock_seconds=10.0,
        state_path=anchor_file,
        wall_clock=lambda: fake_wall[0],
    )
    assert ident_restarted.has_appearance_anchor is True
    obs_after_restart = make_pose(42, (0.30, 0.40, 0.65, 0.60), (0.38, 0.48), (0.55, 0.50))
    frame_restart = make_colored_room_frame([(obs_after_restart, grandma_color)])
    roles_fast, _ = ident_restarted.update([obs_after_restart], BED_ZONE, now=0.2, frame_bgr=frame_restart)
    assert roles_fast[42].role == Role.PATIENT
    assert roles_fast[42].reason == "reid_persisted_anchor"

    fake_wall[0] += 13.0 * 3600.0
    ident_expired = PatientIdentifier(
        bed_lock_seconds=25.0,
        initial_bed_lock_seconds=10.0,
        state_path=anchor_file,
        wall_clock=lambda: fake_wall[0],
    )
    assert ident_expired.has_appearance_anchor is False


# 12. Пауза («Я подошёл»), подавление тревог на время паузы и ручное снятие паузы (resume)
def test_pause_acknowledge_suppression_and_resume_lifecycle() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.5)
    fsm = FallStateMachine(fall_confirm_seconds=0.8)

    t = 0.0
    for _ in range(4):
        t += 0.25
        obs = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([obs], BED_ZONE, now=t)
        fsm.step([obs], roles, cg, BED_ZONE, now=t)

    for _ in range(5):
        t += 0.25
        fallen = make_pose(1, (0.32, 0.76, 0.66, 0.94), (0.38, 0.85), (0.58, 0.86))
        roles, cg = ident.update([fallen], BED_ZONE, now=t)
        fsm.step([fallen], roles, cg, BED_ZONE, now=t)

    assert fsm.state == PatientState.FALLEN
    assert fsm.fall_unacknowledged is True

    fsm.acknowledge_and_pause(now=t, pause_seconds=900.0)
    assert fsm.state == PatientState.PAUSED
    assert fsm.fall_unacknowledged is False
    assert 899.0 <= fsm.pause_remaining_sec(t) <= 900.0

    t += 10.0
    fallen = make_pose(1, (0.32, 0.76, 0.66, 0.94), (0.38, 0.85), (0.58, 0.86))
    roles, cg = ident.update([fallen], BED_ZONE, now=t)
    assert fsm.step([fallen], roles, cg, BED_ZONE, now=t) is None
    assert fsm.state == PatientState.PAUSED

    fsm.resume()
    assert fsm.pause_remaining_sec(t) == 0.0
    assert fsm.state == PatientState.IDLE

    t += 0.3
    back_in_bed = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
    roles, cg = ident.update([back_in_bed], BED_ZONE, now=t)
    fsm.step([back_in_bed], roles, cg, BED_ZONE, now=t)
    assert fsm.state == PatientState.IN_BED


# 13. Адаптивное сглаживание ключевых точек: гасит дрожание, но не задерживает падение
def test_adaptive_keypoint_smoother_suppresses_jitter_and_preserves_fast_drop() -> None:
    smoother = KeypointSmoother(alpha_min=0.22, alpha_max=0.94, jitter_threshold=0.01, fast_threshold=0.05)

    raw_ys: list[float] = []
    smooth_ys: list[float] = []
    t = 0.0
    for i in range(20):
        t += 0.2
        jitter = 0.007 if (i % 2 == 0) else -0.007
        raw_y = 0.50 + jitter
        raw_ys.append(raw_y)
        obs = make_pose(1, (0.30, 0.40, 0.65, 0.60), (0.40, raw_y), (0.55, 0.50))
        sm = smoother.smooth_one(obs, now=t)
        assert sm.mid_shoulder is not None
        smooth_ys.append(sm.mid_shoulder[1])

    raw_std = float(np.std(raw_ys[6:]))
    smooth_std = float(np.std(smooth_ys[6:]))
    assert smooth_std < raw_std * 0.45

    t += 0.2
    drop_obs = make_pose(1, (0.30, 0.60, 0.65, 0.88), (0.40, 0.70), (0.55, 0.72))
    sm_drop = smoother.smooth_one(drop_obs, now=t)
    assert sm_drop.mid_shoulder is not None
    assert sm_drop.mid_shoulder[1] >= 0.68


# 14. Самостоятельный выход из кровати стоя (LEFT_BED) и уход из сектора обзора (MISSING)
def test_leaving_bed_alone_and_missing_from_camera_alerts() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(leave_confirm_seconds=1.2, missing_confirm_seconds=3.0)

    t = 0.0
    for _ in range(8):
        t += 0.25
        sleep_obs = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t)
        fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t)

    left_ev = None
    for _ in range(10):
        t += 0.25
        walking = make_pose(1, (0.10, 0.24, 0.24, 0.82), (0.17, 0.32), (0.17, 0.55), ankle=(0.17, 0.80))
        roles, cg = ident.update([walking], BED_ZONE, now=t)
        ev = fsm.step([walking], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            left_ev = ev
            break

    assert left_ev is not None
    assert left_ev.state == PatientState.LEFT_BED
    assert left_ev.severity == AlertSeverity.WARNING

    missing_ev = None
    for _ in range(16):
        t += 0.25
        roles, cg = ident.update([], BED_ZONE, now=t)
        ev = fsm.step([], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            missing_ev = ev
            break

    assert missing_ev is not None
    assert missing_ev.state == PatientState.MISSING
    assert missing_ev.severity == AlertSeverity.WARNING


# 15. Минималистичный HUD (чисто в покое, однострочный баннер в тревоге) + FaceRegistry.enroll_from_frame
def test_render_annotated_frame_and_face_enroll_validation(tmp_path: Path) -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.5)
    g_obs = make_pose(1, (0.28, 0.40, 0.66, 0.60), (0.36, 0.49), (0.55, 0.50))
    v_enter = make_pose(2, (0.88, 0.20, 0.98, 0.80), (0.93, 0.30), (0.93, 0.54))
    v_obs = make_pose(2, (0.76, 0.20, 0.88, 0.80), (0.82, 0.30), (0.82, 0.54))
    frame = make_colored_room_frame([(g_obs, (120, 90, 200)), (v_obs, (200, 160, 50))])

    ident.update([g_obs, v_enter], BED_ZONE, now=0.0, frame_bgr=frame)
    roles, _ = ident.update([g_obs, v_obs], BED_ZONE, now=0.8, frame_bgr=frame)

    # В спокойных режимах верхний левый угол (0..50, 0..220) не закрыт баннером!
    out_calm = render_annotated_frame(frame, [g_obs, v_obs], roles, BED_ZONE, PatientState.IN_BED)
    assert out_calm.shape == frame.shape
    assert np.array_equal(out_calm[:35, :200], frame[:35, :200]), "В покое верхний угол обязан быть чистым без плашек!"

    # В тревожном состоянии FALLEN рисуется компактная однострочная плашка сверху слева
    out_alert = render_annotated_frame(
        frame, [g_obs, v_obs], roles, BED_ZONE, PatientState.FALLEN, status_reason="сползла на пол"
    )
    assert not np.array_equal(out_alert[:35, :200], frame[:35, :200])

    reg = FaceRegistry(faces_dir=tmp_path / "faces", models_dir=tmp_path / "models")
    ok_bad_role, msg_role = reg.enroll_from_frame("invalid_role", frame)
    assert ok_bad_role is False
    assert "patient" in msg_role

    ok_empty, msg_empty = reg.enroll_from_frame(Role.PATIENT, None)
    assert ok_empty is False
    assert "пуст" in msg_empty


# 16. Отсев фантомного трека на скомканном одеяле перед ИК-объективом
def test_phantom_blanket_pose_without_skeleton_anatomy_is_rejected() -> None:
    phantom_kps = tuple((0.25 + 0.01 * i, 0.75, 0.27) for i in range(17))
    phantom_blanket = PoseObservation(
        track_id=1,
        bbox_norm=(0.0, 0.595, 0.611, 0.991),
        conf=0.606,
        keypoints_norm=phantom_kps,
    )
    noise_kps = [(0.0, 0.0, 0.0)] * 17
    noise_kps[13] = (0.20, 0.80, 0.40)
    noise_kps[15] = (0.22, 0.88, 0.41)
    noise_kps[16] = (0.26, 0.89, 0.38)
    phantom_legs_only = PoseObservation(
        track_id=2,
        bbox_norm=(0.05, 0.62, 0.55, 0.98),
        conf=0.52,
        keypoints_norm=tuple(noise_kps),
    )

    assert filter_valid_poses([phantom_blanket, phantom_legs_only]) == []

    ident = PatientIdentifier(initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(sit_confirm_seconds=0.8)
    t = 0.0
    for _ in range(5):
        t += 0.25
        grandma_sleep = make_pose(10, (0.30, 0.42, 0.68, 0.58), (0.36, 0.49), (0.55, 0.50))
        roles, cg = ident.update([grandma_sleep, phantom_blanket], BED_ZONE, now=t)
        assert 1 not in roles
        assert cg is False
        fsm.step([grandma_sleep], roles, cg, BED_ZONE, now=t)

    sit_ev = None
    for _ in range(6):
        t += 0.25
        grandma_sit = make_pose(10, (0.40, 0.30, 0.56, 0.80), (0.48, 0.36), (0.48, 0.56), ankle=(0.49, 0.78))
        roles, cg = ident.update([grandma_sit, phantom_blanket], BED_ZONE, now=t)
        assert cg is False
        ev = fsm.step([grandma_sit], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            sit_ev = ev
            break

    assert sit_ev is not None
    assert sit_ev.state == PatientState.SITTING_UP


# 17. Устойчивость дескриптора одежды в ночном монохромном ИК-режиме (S=0)
def test_monochrome_ir_appearance_does_not_degenerate_in_zero_saturation() -> None:
    grandma_obs = make_pose(1, (0.32, 0.38, 0.62, 0.62), (0.40, 0.48), (0.54, 0.50))
    blanket_obs = make_pose(2, (0.04, 0.62, 0.52, 0.96), (0.18, 0.78), (0.38, 0.80))

    ir_frame = np.full((360, 640, 3), (45, 45, 45), dtype=np.uint8)
    ir_frame[135:225, 205:395] = (102, 102, 102)
    ir_frame[135:225:4, 205:395] = (86, 86, 86)
    ir_frame[225:345, 25:330] = (232, 232, 232)

    vec_grandma_1 = compute_torso_appearance(ir_frame, grandma_obs)
    vec_blanket = compute_torso_appearance(ir_frame, blanket_obs)

    ir_frame_2 = ir_frame.copy()
    ir_frame_2[140:220, 210:390] = np.clip(ir_frame_2[140:220, 210:390].astype(np.int16) + 3, 0, 255).astype(np.uint8)
    vec_grandma_2 = compute_torso_appearance(ir_frame_2, grandma_obs)

    assert vec_grandma_1 is not None and vec_blanket is not None and vec_grandma_2 is not None
    assert cosine_sim(vec_grandma_1, vec_grandma_2) >= 0.92
    assert cosine_sim(vec_grandma_1, vec_blanket) < 0.45


# 18. Полевой регрессионный тест current_bug.jpg: неподвижная куртка/халат на стене у двери
# НЕ попадает в активные треки, НЕ перехватывает reid_appearance и НЕ вызывает LEFT_BED / MISSING!
def test_hanging_jacket_on_wall_is_filtered_as_static_object_and_never_reids() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(leave_confirm_seconds=1.0, missing_confirm_seconds=3.0)
    rng = np.random.default_rng(42)

    # 1. Безголовый «скелет-вешалка» вне кровати мгновенно отсекается ещё в filter_valid_poses
    coat_hanger_headless = make_pose(
        track_id=99,
        bbox=(0.20, 0.38, 0.32, 0.90),
        shoulder=(0.26, 0.48),
        hip=(0.26, 0.68),
        headless=True,
    )
    assert filter_valid_poses([coat_hanger_headless], zones=BED_ZONE) == []

    # 2. Даже если YOLO придумал куртке точки «глаз» на вешалке И у системы есть якорь одежды того же оттенка:
    same_ir_coat_color = (145, 145, 145)
    t = 0.0
    for _ in range(6):
        t += 0.25
        g_sleep = make_pose(1, (0.32, 0.42, 0.66, 0.58), (0.38, 0.49), (0.56, 0.50))
        fr = make_colored_room_frame([(g_sleep, same_ir_coat_color)])
        roles, cg = ident.update([g_sleep], BED_ZONE, now=t, frame_bgr=fr)
        fsm.step([g_sleep], roles, cg, BED_ZONE, now=t)
    assert ident.has_appearance_anchor is True

    # Сбросим состояние ухода в покой и оставим в пустой комнате ТОЛЬКО висящую на стене куртку #345
    fsm.reset_to_idle()
    base_jacket = make_pose(345, (0.205, 0.38, 0.315, 0.90), (0.26, 0.49), (0.26, 0.68), ankle=(0.26, 0.84))
    base_frame = make_colored_room_frame([(base_jacket, same_ir_coat_color)])

    for step_i in range(40):
        t += 0.25
        # Имитируем ИК-шум матрицы ±4 и дрожь ключевых точек ±0.006 на неподвижной куртке
        jx = 0.005 if (step_i % 2 == 0) else -0.005
        jacket_obs = make_pose(
            track_id=345,
            bbox=(0.205, 0.38, 0.315, 0.90),
            shoulder=(0.26 + jx, 0.49),
            hip=(0.26, 0.68),
            ankle=(0.26, 0.84),
        )
        noisy_frame = np.clip(
            base_frame.astype(np.int16) + rng.integers(-4, 5, size=base_frame.shape, dtype=np.int16),
            0,
            255,
        ).astype(np.uint8)
        roles, cg = ident.update([jacket_obs], BED_ZONE, now=t, frame_bgr=noisy_frame)
        ev = fsm.step([jacket_obs], roles, cg, BED_ZONE, now=t)
        assert roles == {}, "Неподвижная куртка на стене НЕ должна попадать в активные человеческие треки!"
        assert ev is None


# 19. Зоны игнорирования вещей (ignore_polygons) и сброс якоря clear_anchor()
def test_ignore_polygons_and_clear_anchor(tmp_path: Path) -> None:
    anchor_file = tmp_path / "patient_anchor.json"
    ident = PatientIdentifier(initial_bed_lock_seconds=0.6, state_path=anchor_file)

    t = 0.0
    for _ in range(5):
        t += 0.25
        g_obs = make_pose(1, (0.32, 0.42, 0.66, 0.58), (0.38, 0.49), (0.56, 0.50))
        fr = make_colored_room_frame([(g_obs, (110, 90, 180))])
        roles, _ = ident.update([g_obs], BED_ZONE, now=t, frame_bgr=fr)

    assert roles[1].role == Role.PATIENT
    assert anchor_file.exists()

    # Пользователь нажал «[🚫 Это вещь]» -> вызывается clear_anchor() и добавляется ignore_polygon на угол стены
    ident.clear_anchor()
    assert ident.has_appearance_anchor is False
    assert not anchor_file.exists()

    zones_ign = make_zones_with_ignore(
        bed_polygon=BED_ZONE.bed_polygon,
        door_polygon=BED_ZONE.door_polygon,
        ignore_polygons=[[[0.18, 0.35], [0.34, 0.35], [0.34, 0.92], [0.18, 0.92]]],
    )
    coat_in_ignore = make_pose(7, (0.20, 0.38, 0.32, 0.90), (0.26, 0.48), (0.26, 0.68))
    assert filter_valid_poses([coat_in_ignore], zones=zones_ign) == []
    roles_after, _ = ident.update([coat_in_ignore], zones_ign, now=t + 1.0)  # type: ignore[arg-type]
    assert roles_after == {}


# 20. Отсутствие спама MISSING каждые 45 секунд (строго 1 срабатывание за эпизод ухода из кровати)
def test_missing_alert_fires_strictly_once_per_bed_episode_without_spam() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(missing_confirm_seconds=2.0, alert_cooldown_seconds=10.0)

    t = 0.0
    for _ in range(6):
        t += 0.25
        sleep_obs = make_pose(1, (0.30, 0.42, 0.66, 0.58), (0.36, 0.49), (0.55, 0.50))
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t)
        fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t)
    assert fsm.state == PatientState.IN_BED

    # Бабушка ушла из комнаты и отсутствует 180 секунд (18 раз дольше cooldown=10s!)
    missing_events = []
    for _ in range(360):
        t += 0.5
        roles, cg = ident.update([], BED_ZONE, now=t)
        ev = fsm.step([], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            missing_events.append(ev)

    assert len(missing_events) == 1, f"Тревога MISSING обязана сработать ровно 1 раз, а не {len(missing_events)}!"
    assert missing_events[0].state == PatientState.MISSING

    # Только когда бабушка реально вернулась в кровать (IN_BED) и снова ушла — взводится новый эпизод
    for _ in range(4):
        t += 0.3
        sleep_obs = make_pose(1, (0.30, 0.42, 0.66, 0.58), (0.36, 0.49), (0.55, 0.50))
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t)
        fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t)
    assert fsm.state == PatientState.IN_BED

    second_episode = []
    for _ in range(20):
        t += 0.5
        roles, cg = ident.update([], BED_ZONE, now=t)
        ev = fsm.step([], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            second_episode.append(ev)
    assert len(second_episode) == 1


# 21. Живой движущийся человек проходит StaticObjectGate за доли секунды и сохраняется после остановки
def test_static_object_gate_allows_moving_human_even_after_stopping() -> None:
    gate = StaticObjectGate(min_displacement=0.05)
    # 1-й кадр вне кровати: смещение 0 -> ждёт подтверждения движения
    p0 = make_pose(5, (0.78, 0.22, 0.90, 0.82), (0.84, 0.32), (0.84, 0.55))
    assert gate.update_and_filter([p0], zones=BED_ZONE, now=1.0) == []

    # 2-й кадр: человек сделал шаг влево на 0.08 (>= 0.05) -> сразу пропускается!
    p1 = make_pose(5, (0.70, 0.22, 0.82, 0.82), (0.76, 0.32), (0.76, 0.55))
    out1 = gate.update_and_filter([p1], zones=BED_ZONE, now=1.2)
    assert len(out1) == 1 and out1[0].track_id == 5

    # Далее человек остановился на месте у кровати — продолжает отслеживаться без потерь!
    for step_t in (1.4, 2.0, 5.0):
        out_stay = gate.update_and_filter([p1], zones=BED_ZONE, now=step_t)
        assert len(out_stay) == 1 and out_stay[0].track_id == 5


# 22. Сквозной конвейер engine.py (infer без zones -> identifier.update с zones):
# неподвижно спящая в кровати бабушка сохраняется как PATIENT, а неподвижная куртка на стене отсекается!
def test_engine_pipeline_without_zones_in_infer_keeps_sleeping_patient_and_drops_coat() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine()
    t = 0.0
    for _ in range(8):
        t += 0.25
        sleeping_grandma = make_pose(10, (0.30, 0.42, 0.66, 0.58), (0.37, 0.49), (0.56, 0.50))
        wall_coat = make_pose(345, (0.04, 0.35, 0.16, 0.88), (0.10, 0.46), (0.10, 0.68), ankle=(0.10, 0.84))
        raw_obs = filter_valid_poses([sleeping_grandma, wall_coat], zones=None)
        roles, cg = ident.update(raw_obs, BED_ZONE, now=t)
        live_obs = [o for o in raw_obs if o.track_id in roles]
        ev = fsm.step(live_obs, roles, cg, BED_ZONE, now=t)
        assert ev is None

    assert set(roles.keys()) == {10}
    assert roles[10].role == Role.PATIENT
    assert fsm.state == PatientState.IN_BED


# 23. TimelineStore: слияние подряд идущих одинаковых состояний при разрыве <= 30 с и новый интервал при > 30 с
def test_timeline_store_merges_consecutive_same_state_within_30s_and_splits_on_gap(tmp_path: Path) -> None:
    from app.timeline import TimelineStore

    store = TimelineStore(tmp_path / "timeline.json", window_sec=21600.0)
    t0 = 1_700_000_000.0

    # Разрывы 10 с, 25 с и ровно 30.0 с (<= 30.0 с) должны слиться в один непрерывный интервал
    store.record("in_bed", True, now_ts=t0)
    store.record("in_bed", True, now_ts=t0 + 10.0)
    store.record("in_bed", True, now_ts=t0 + 35.0)
    store.record("in_bed", True, now_ts=t0 + 65.0)

    snap1 = store.to_dict(now_ts=t0 + 65.0)
    assert len(snap1["spans"]) == 1
    assert snap1["spans"][0]["state"] == "in_bed"
    assert snap1["spans"][0]["tone"] == "calm"
    assert snap1["spans"][0]["label"] == "В кровати"
    assert snap1["spans"][0]["start"] == pytest.approx(t0)
    assert snap1["spans"][0]["end"] == pytest.approx(t0 + 65.0)

    # Разрыв 35.0 с (> 30.0 с) обязан открыть второй отдельный интервал того же состояния
    store.record("in_bed", True, now_ts=t0 + 100.0)
    store.record("in_bed", True, now_ts=t0 + 115.0)

    snap2 = store.to_dict(now_ts=t0 + 115.0)
    assert len(snap2["spans"]) == 2
    assert snap2["spans"][0]["start"] == pytest.approx(t0)
    assert snap2["spans"][0]["end"] == pytest.approx(t0 + 65.0)
    assert snap2["spans"][1]["start"] == pytest.approx(t0 + 100.0)
    assert snap2["spans"][1]["end"] == pytest.approx(t0 + 115.0)


# 24. TimelineStore: разделение интервалов при смене состояния in_bed -> sitting_up -> paused и маппинг тонов
def test_timeline_store_splits_on_state_transitions_and_maps_tones_and_labels(tmp_path: Path) -> None:
    from app.timeline import TimelineStore

    store = TimelineStore(tmp_path / "timeline.json", window_sec=21600.0)
    t0 = 1_700_010_000.0

    store.record("in_bed", True, now_ts=t0)
    store.record("in_bed", True, now_ts=t0 + 20.0)
    store.record("sitting_up", True, now_ts=t0 + 25.0)
    store.record("sitting_up", True, now_ts=t0 + 40.0)
    store.record("paused", True, now_ts=t0 + 45.0)
    store.record("paused", True, now_ts=t0 + 70.0)

    spans = store.to_dict(now_ts=t0 + 70.0)["spans"]
    assert [s["state"] for s in spans] == ["in_bed", "sitting_up", "paused"]
    assert [s["tone"] for s in spans] == ["calm", "warn", "paused"]
    assert [s["label"] for s in spans] == ["В кровати", "Села в кровати", "Пауза"]
    assert spans[0]["start"] == pytest.approx(t0)
    assert t0 + 20.0 <= spans[0]["end"] <= t0 + 25.0
    assert spans[1]["start"] == pytest.approx(t0 + 25.0)
    assert t0 + 40.0 <= spans[1]["end"] <= t0 + 45.0
    assert spans[2]["start"] == pytest.approx(t0 + 45.0)
    assert spans[2]["end"] == pytest.approx(t0 + 70.0)

    # Полная матрица остальных состояний и приоритета offline при cam_online=False
    store2 = TimelineStore(tmp_path / "timeline_all.json", window_sec=21600.0)
    cases = [
        ("in_bed", False, "offline", "offline", "Нет сигнала"),
        ("idle", True, "idle", "calm", "Спокойно"),
        ("caregiver", True, "caregiver", "caregiver", "Рядом свой"),
        ("left_bed", True, "left_bed", "warn", "Встала одна"),
        ("missing", True, "missing", "warn", "Вышла из кадра"),
        ("fallen", True, "fallen", "alarm", "Падение"),
    ]
    for idx, (raw_st, online, exp_st, exp_tone, exp_lbl) in enumerate(cases):
        store2.record(raw_st, online, now_ts=t0 + 200.0 + idx * 10.0)

    spans2 = store2.to_dict(now_ts=t0 + 260.0)["spans"]
    assert [(s["state"], s["tone"], s["label"]) for s in spans2] == [
        (c[2], c[3], c[4]) for c in cases
    ]


# 25. TimelineStore: обрезка интервалов старше окна window_sec и подрезка левой границы start = cutoff
def test_timeline_store_trims_expired_spans_and_clips_left_boundary_to_cutoff(tmp_path: Path) -> None:
    from app.timeline import TimelineStore

    store = TimelineStore(tmp_path / "timeline.json", window_sec=600.0)
    t0 = 1_700_020_000.0

    # 1) Полностью устаревший интервал [t0, t0 + 30] (левее будущего cutoff = t0 + 200)
    store.record("idle", True, now_ts=t0)
    store.record("idle", True, now_ts=t0 + 30.0)

    # 2) Пограничный интервал [t0 + 180, t0 + 235], пересекающий cutoff = t0 + 200
    store.record("in_bed", True, now_ts=t0 + 180.0)
    store.record("in_bed", True, now_ts=t0 + 210.0)
    store.record("in_bed", True, now_ts=t0 + 235.0)

    # 3) Свежий интервал в момент now_ts = t0 + 800 -> cutoff = (t0 + 800) - 600 = t0 + 200
    now_eval = t0 + 800.0
    store.record("sitting_up", True, now_ts=now_eval)

    payload = store.to_dict(now_ts=now_eval)
    assert payload["window_sec"] == pytest.approx(600.0)
    assert payload["start_ts"] == pytest.approx(t0 + 200.0)
    assert payload["now_ts"] == pytest.approx(now_eval)

    spans = payload["spans"]
    assert len(spans) == 2
    assert spans[0]["state"] == "in_bed"
    assert spans[0]["start"] == pytest.approx(t0 + 200.0)
    assert spans[0]["end"] == pytest.approx(t0 + 235.0)
    assert spans[1]["state"] == "sitting_up"
    assert spans[1]["start"] == pytest.approx(now_eval)


# 26. TimelineStore: атомарное сохранение и повторное чтение timeline.json с диска
def test_timeline_store_persists_and_reloads_timeline_json_from_disk(tmp_path: Path) -> None:
    from app.timeline import TimelineStore

    timeline_path = tmp_path / "timeline.json"
    t0 = 1_700_030_000.0

    writer = TimelineStore(timeline_path, window_sec=21600.0)
    writer.record("in_bed", True, now_ts=t0)
    writer.record("in_bed", True, now_ts=t0 + 25.0)
    writer.record("caregiver", True, now_ts=t0 + 30.0)
    writer.record("caregiver", True, now_ts=t0 + 55.0)
    # Смена состояния в конце гарантирует немедленный атомарный сброс актуальных интервалов на диск
    writer.record("paused", True, now_ts=t0 + 60.0)

    assert timeline_path.exists()
    raw_disk = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert isinstance(raw_disk, (dict, list))

    reader = TimelineStore(timeline_path, window_sec=21600.0)
    restored = reader.to_dict(now_ts=t0 + 70.0)
    spans = restored["spans"]
    assert [s["state"] for s in spans] == ["in_bed", "caregiver", "paused"]
    assert [s["tone"] for s in spans] == ["calm", "caregiver", "paused"]
    assert spans[0]["start"] == pytest.approx(t0)
    assert t0 + 25.0 <= spans[0]["end"] <= t0 + 30.0
    assert spans[1]["start"] == pytest.approx(t0 + 30.0)
    assert t0 + 55.0 <= spans[1]["end"] <= t0 + 60.0
    assert spans[2]["start"] == pytest.approx(t0 + 60.0)


def _with_arm_keypoints(
    base_obs: PoseObservation,
    right_elbow: tuple[float, float, float] | None = None,
    right_wrist: tuple[float, float, float] | None = None,
    left_elbow: tuple[float, float, float] | None = None,
    left_wrist: tuple[float, float, float] | None = None,
) -> PoseObservation:
    kps = list(base_obs.keypoints_norm)
    if left_elbow is not None:
        kps[7] = left_elbow
    if right_elbow is not None:
        kps[8] = right_elbow
    if left_wrist is not None:
        kps[9] = left_wrist
    if right_wrist is not None:
        kps[10] = right_wrist
    return PoseObservation(
        track_id=base_obs.track_id,
        bbox_norm=base_obs.bbox_norm,
        conf=base_obs.conf,
        keypoints_norm=tuple(kps),
    )


# 27. Жест вызова помощи поднятой рукой (HELP_CALL): биомеханика, защита от руки под подушкой и HUD-плашка
def test_help_call_gesture_biomechanics_and_pillow_hand_rejection() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(
        sit_confirm_seconds=1.4,
        enable_help_gesture=True,
        help_confirm_seconds=2.0,
        night_strict_mode=False,
    )

    t = 0.0
    # 1. Калибруем эталон сна (shoulder_y ≈ 0.49), рука лежит под подушкой на уровне головы — ложных тревог нет!
    for _ in range(10):
        t += 0.25
        sleep_base = make_pose(1, (0.28, 0.40, 0.68, 0.58), (0.36, 0.49), (0.56, 0.50), nose=(0.34, 0.45))
        pillow_hand = _with_arm_keypoints(
            sleep_base,
            right_elbow=(0.30, 0.46, 0.85),
            right_wrist=(0.25, 0.44, 0.90),  # Вбок и почти на уровне носа — не вызов помощи!
        )
        ok_help, _ = is_hand_raised_for_help(pillow_hand, baseline_shoulder_y=0.49)
        assert ok_help is False
        roles, cg = ident.update([pillow_hand], BED_ZONE, now=t)
        assert fsm.step([pillow_hand], roles, cg, BED_ZONE, now=t) is None

    assert fsm.baseline_in_bed["calibrated"] is True

    # 2. Бабушка приподнимается и поднимает правую руку строго вверх над головой — HELP_CALL приоритетнее SITTING_UP!
    help_ev = None
    last_roles = {}
    last_obs_list: list[PoseObservation] = []
    for _ in range(12):
        t += 0.25
        sit_base = make_pose(1, (0.34, 0.20, 0.64, 0.58), (0.44, 0.40), (0.52, 0.50), nose=(0.44, 0.35))
        raised_hand = _with_arm_keypoints(
            sit_base,
            right_elbow=(0.46, 0.30, 0.88),
            right_wrist=(0.46, 0.22, 0.92),  # lift = 0.18, выше носа и выше baseline_shoulder_y - 0.10
        )
        ok_help, reason = is_hand_raised_for_help(raised_hand, baseline_shoulder_y=0.49)
        assert ok_help is True
        assert "правая рука" in reason.lower()

        roles, cg = ident.update([raised_hand], BED_ZONE, now=t)
        ev = fsm.step([raised_hand], roles, cg, BED_ZONE, now=t)
        last_roles = roles
        last_obs_list = [raised_hand]
        if ev is not None and ev.state == PatientState.HELP_CALL:
            help_ev = ev
            break

    assert help_ev is not None
    assert fsm.state == PatientState.HELP_CALL
    assert help_ev.severity == AlertSeverity.WARNING
    assert help_ev.title == "Зовёт на помощь (поднята рука)"

    # 3. Проверяем отрисовку компактной тревожной плашки HELP_CALL на видео-HUD (в сравнении со спокойным IN_BED)
    blank = np.full((360, 640, 3), 24, dtype=np.uint8)
    calm_hud = render_annotated_frame(blank, last_obs_list, last_roles, BED_ZONE, PatientState.IN_BED)
    help_hud = render_annotated_frame(blank, last_obs_list, last_roles, BED_ZONE, PatientState.HELP_CALL)
    # В левом верхнем углу при HELP_CALL появляется компактный баннер, а при IN_BED угол девственно чист
    assert float(np.mean(np.abs(help_hud[:52, :280].astype(np.float32) - calm_hud[:52, :280].astype(np.float32)))) > 2.0


# 28. Эскалация затяжного отсутствия (prolonged_absence_minutes): однократный CRITICAL и сброс при возврате
def test_prolonged_absence_escalates_once_to_critical_and_resets_on_return() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.6)
    fsm = FallStateMachine(
        leave_confirm_seconds=1.0,
        missing_confirm_seconds=2.0,
        prolonged_absence_minutes=15.0,
        night_strict_mode=False,
    )

    t = 0.0
    for _ in range(5):
        t += 0.25
        sleep_obs = make_pose(1, (0.30, 0.42, 0.66, 0.58), (0.36, 0.49), (0.55, 0.50))
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t)
        fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t)
    assert fsm.state == PatientState.IN_BED

    # Ушла из сектора обзора -> сначала однократный WARNING (MISSING)
    events = []
    for _ in range(12):
        t += 0.5
        roles, cg = ident.update([], BED_ZONE, now=t)
        ev = fsm.step([], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            events.append(ev)
    assert len(events) == 1 and events[0].state == PatientState.MISSING

    # Моделируем 20 минут отсутствия (шагами по 1.0 с, чтобы dt в step шёл непрерывно; либо прыжком реального времени now)
    crit_events = []
    for _ in range(120):
        t += 10.0  # _away_since сравнивается по реальному (now - _away_since) >= 15 * 60!
        roles, cg = ident.update([], BED_ZONE, now=t)
        ev = fsm.step([], roles, cg, BED_ZONE, now=t)
        if ev is not None:
            crit_events.append(ev)

    assert len(crit_events) == 1
    assert crit_events[0].severity == AlertSeverity.CRITICAL
    assert crit_events[0].title == "Долго не возвращается в кровать!"
    assert fsm.fall_unacknowledged is True
    diag_away = fsm.get_care_diagnostics(now=t)
    assert diag_away["away_from_bed_minutes"] >= 15.0

    # Возврат бабушки в кровать сбрасывает счётчик отсутствия и флаг неподтверждённой тревоги
    for _ in range(4):
        t += 0.3
        sleep_obs = make_pose(1, (0.30, 0.42, 0.66, 0.58), (0.36, 0.49), (0.55, 0.50))
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t)
        fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t)

    assert fsm.state == PatientState.IN_BED
    assert fsm.fall_unacknowledged is False
    assert fsm.get_care_diagnostics(now=t)["away_from_bed_minutes"] == 0.0


# 29. Профилактика пролежней (immobility_reminder_minutes): сброс при смене позы и мягкое напоминание
def test_immobility_pressure_ulcer_reminder_and_posture_shift_reset() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.5)
    fsm = FallStateMachine(immobility_reminder_minutes=120.0, night_strict_mode=False)

    t = 0.0
    pose_a = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
    for _ in range(5):
        t += 0.25
        roles, cg = ident.update([pose_a], BED_ZONE, now=t)
        assert fsm.step([pose_a], roles, cg, BED_ZONE, now=t) is None

    # Через 90 минут (< 120 мин) бабушка перевернулась на другой край матраса (смещение >= 0.045)
    t += 90.0 * 60.0
    pose_b = make_pose(1, (0.34, 0.47, 0.72, 0.63), (0.41, 0.55), (0.61, 0.56))
    roles, cg = ident.update([pose_b], BED_ZONE, now=t)
    assert fsm.step([pose_b], roles, cg, BED_ZONE, now=t) is None
    assert fsm.get_care_diagnostics(now=t)["immobility_minutes"] == 0.0

    # Ещё через 90 минут (всего 180 от старта, но 90 после смены позы!) напоминания ещё быть НЕ должно
    t += 90.0 * 60.0
    roles, cg = ident.update([pose_b], BED_ZONE, now=t)
    assert fsm.step([pose_b], roles, cg, BED_ZONE, now=t) is None

    # А ещё через 35 минут (125 мин в позе B >= 120 мин) выдаётся ровно ОДНО мягкое напоминание
    t += 35.0 * 60.0
    roles, cg = ident.update([pose_b], BED_ZONE, now=t)
    ev1 = fsm.step([pose_b], roles, cg, BED_ZONE, now=t)
    assert ev1 is not None
    assert ev1.state == PatientState.IN_BED
    assert ev1.severity == AlertSeverity.WARNING
    assert ev1.title == "Напоминание об уходе: смена позы"

    # На следующем кадре повторного спама нет
    t += 60.0
    roles, cg = ident.update([pose_b], BED_ZONE, now=t)
    assert fsm.step([pose_b], roles, cg, BED_ZONE, now=t) is None


# 30. Живой индикатор микро-активности (last_movement_sec_ago): устойчивость к шуму <0.012 и реакция на шевеление
def test_micro_movement_tracking_updates_on_subtle_limb_shift_and_ignores_jitter() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.4)
    fsm = FallStateMachine(night_strict_mode=False)

    base = make_pose(1, (0.30, 0.42, 0.66, 0.58), (0.36, 0.49), (0.55, 0.50), ankle=(0.64, 0.51))
    roles, cg = ident.update([base], BED_ZONE, now=10.0)
    fsm.step([base], roles, cg, BED_ZONE, now=10.0)
    roles, cg = ident.update([base], BED_ZONE, now=10.5)
    fsm.step([base], roles, cg, BED_ZONE, now=10.5)

    # 30 секунд микро-дрожи +/-0.004 (< 0.012) — движения нет, счётчик копится
    t = 10.5
    for i in range(15):
        t += 2.0
        jitter = 0.004 if (i % 2 == 0) else -0.004
        j_obs = make_pose(1, (0.30, 0.42, 0.66, 0.58), (0.36 + jitter, 0.49), (0.55, 0.50), ankle=(0.64, 0.51))
        roles, cg = ident.update([j_obs], BED_ZONE, now=t)
        fsm.step([j_obs], roles, cg, BED_ZONE, now=t)

    diag_still = fsm.get_care_diagnostics(now=t)
    assert diag_still["last_movement_sec_ago"] is not None
    assert diag_still["last_movement_sec_ago"] >= 29.5

    # Плавное шевеление стопой на 3 шага по +0.006 (суммарно +0.018 >= 0.012 от якоря покоя!)
    for step_k in (1, 2, 3):
        t += 0.5
        m_obs = make_pose(
            1,
            (0.30, 0.42, 0.66, 0.58),
            (0.36, 0.49),
            (0.55, 0.50),
            ankle=(0.64, 0.51 + 0.006 * step_k),
        )
        roles, cg = ident.update([m_obs], BED_ZONE, now=t)
        fsm.step([m_obs], roles, cg, BED_ZONE, now=t)

    diag_moved = fsm.get_care_diagnostics(now=t)
    assert diag_moved["last_movement_sec_ago"] == pytest.approx(0.0, abs=0.6)


# 31. Ночной строгий режим (22:00–07:00): ускорение перехвата посадки и выхода ночью относительно дня
def test_night_strict_mode_tightens_thresholds_at_night_and_relaxes_by_day() -> None:
    ident_day = PatientIdentifier(initial_bed_lock_seconds=0.4)
    ident_night = PatientIdentifier(initial_bed_lock_seconds=0.4)
    fsm_day = FallStateMachine(
        sit_confirm_seconds=2.0,
        leave_confirm_seconds=2.4,
        night_strict_mode=True,
        night_start_hour=22,
        night_end_hour=7,
    )
    fsm_night = FallStateMachine(
        sit_confirm_seconds=2.0,
        leave_confirm_seconds=2.4,
        night_strict_mode=True,
        night_start_hour=22,
        night_end_hour=7,
    )

    assert fsm_day.get_care_diagnostics(local_hour=14)["night_active"] is False
    assert fsm_day.get_care_diagnostics(local_hour=14)["effective_sit_confirm_sec"] == pytest.approx(2.0)
    assert fsm_night.get_care_diagnostics(local_hour=2)["night_active"] is True
    assert fsm_night.get_care_diagnostics(local_hour=2)["effective_sit_confirm_sec"] == pytest.approx(1.44)
    assert fsm_night.get_care_diagnostics(local_hour=2)["effective_leave_confirm_sec"] == pytest.approx(1.80)

    t = 0.0
    for _ in range(5):
        t += 0.2
        s_obs = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        r_d, cg_d = ident_day.update([s_obs], BED_ZONE, now=t)
        r_n, cg_n = ident_night.update([s_obs], BED_ZONE, now=t)
        fsm_day.step([s_obs], r_d, cg_d, BED_ZONE, now=t, local_hour=14)
        fsm_night.step([s_obs], r_n, cg_n, BED_ZONE, now=t, local_hour=2)

    # Свешивает ноги на протяжении 1.6 с (>= 1.44 с ночью, но < 2.0 с днём!)
    day_ev = None
    night_ev = None
    for _ in range(8):
        t += 0.2
        hang_obs = make_pose(1, (0.40, 0.30, 0.56, 0.80), (0.48, 0.36), (0.48, 0.56), ankle=(0.49, 0.78))
        r_d, cg_d = ident_day.update([hang_obs], BED_ZONE, now=t)
        r_n, cg_n = ident_night.update([hang_obs], BED_ZONE, now=t)
        ev_d = fsm_day.step([hang_obs], r_d, cg_d, BED_ZONE, now=t, local_hour=14)
        ev_n = fsm_night.step([hang_obs], r_n, cg_n, BED_ZONE, now=t, local_hour=2)
        if ev_d is not None:
            day_ev = ev_d
        if ev_n is not None:
            night_ev = ev_n

    assert night_ev is not None and night_ev.state == PatientState.SITTING_UP
    assert day_ev is None and fsm_day.state != PatientState.SITTING_UP


# 32. Горячее обновление настроек (update_runtime_prefs), reset_baseline и полнота get_care_diagnostics
def test_runtime_prefs_hot_update_reset_baseline_and_care_diagnostics() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.4)
    fsm = FallStateMachine()

    t = 0.0
    for _ in range(6):
        t += 0.2
        s_obs = make_pose(1, (0.28, 0.42, 0.68, 0.58), (0.35, 0.49), (0.55, 0.50))
        roles, cg = ident.update([s_obs], BED_ZONE, now=t)
        fsm.step([s_obs], roles, cg, BED_ZONE, now=t, local_hour=13)
    assert fsm.baseline_in_bed["calibrated"] is True

    fsm.reset_baseline()
    assert fsm.baseline_in_bed["calibrated"] is False
    assert fsm.baseline_in_bed["samples"] == 0

    applied = fsm.update_runtime_prefs(
        sensitivity_preset="gentle",  # Лишний ключ из RuntimePrefsStore игнорируется без ошибки
        sit_confirm_seconds=2.8,
        leave_confirm_seconds=3.0,
        fall_confirm_seconds=2.2,
        enable_help_gesture=False,
        prolonged_absence_minutes=30.0,
        immobility_reminder_minutes=180.0,
        night_strict_mode=False,
    )
    assert applied["sit_confirm_seconds"] == pytest.approx(2.8)
    assert applied["enable_help_gesture"] is False
    assert applied["prolonged_absence_minutes"] == pytest.approx(30.0)
    assert applied["immobility_reminder_minutes"] == pytest.approx(180.0)

    diag = fsm.get_care_diagnostics(now=t, local_hour=23)
    expected_keys = {
        "baseline",
        "night_active",
        "effective_sit_confirm_sec",
        "effective_leave_confirm_sec",
        "immobility_minutes",
        "away_from_bed_minutes",
        "last_movement_sec_ago",
    }
    assert expected_keys.issubset(diag.keys())
    assert diag["night_active"] is False
    assert diag["effective_sit_confirm_sec"] == pytest.approx(2.8)
    assert diag["effective_leave_confirm_sec"] == pytest.approx(3.0)


# 33. Сон под плотным одеялом при ярком контровом свете (conf=0.227, бёдра скрыты одеялом) + 0% сработок на куртку
def test_blanket_sleeper_in_bed_detected_at_low_conf_without_hips_while_wall_coat_rejected() -> None:
    real_zones = ZonesConfig(
        bed_polygon=[[0.29, 0.73], [0.74, 0.68], [0.67, 0.99], [0.20, 0.99]],
        door_polygon=[[0.01, 0.08], [0.20, 0.28], [0.17, 0.99], [0.01, 0.99]],
    )

    # Точный слепок сырого выхода YOLO11s-pose на дневном кадре 1920x1080 со слепящим окном:
    # видны голова на подушке и плечи под краем одеяла (conf бокса 0.227), бёдра 11..12 скрыты (< 0.30)
    grandma_kps = [(0.0, 0.0, 0.0)] * 17
    grandma_kps[0] = (0.556, 0.819, 0.82)
    grandma_kps[1] = (0.562, 0.816, 0.87)
    grandma_kps[2] = (0.560, 0.814, 0.47)
    grandma_kps[3] = (0.561, 0.825, 0.81)
    grandma_kps[4] = (0.565, 0.822, 0.16)
    grandma_kps[5] = (0.534, 0.858, 0.78)
    grandma_kps[6] = (0.545, 0.848, 0.62)
    grandma_kps[7] = (0.531, 0.873, 0.52)
    grandma_kps[9] = (0.560, 0.853, 0.44)
    grandma_kps[11] = (0.453, 0.961, 0.20)
    grandma_kps[12] = (0.458, 0.955, 0.16)
    grandma_blanket = PoseObservation(
        track_id=1,
        bbox_norm=(0.516, 0.797, 0.588, 0.890),
        conf=0.227,
        keypoints_norm=tuple(grandma_kps),
    )

    # Висящая куртка/халат на левой стене (conf=0.410 вне кровати, и вариант conf=0.560 неподвижный)
    coat_low_conf = make_pose(
        track_id=2,
        bbox=(0.135, 0.355, 0.261, 0.821),
        shoulder=(0.190, 0.476),
        hip=(0.188, 0.654),
        conf=0.410,
        nose=(0.203, 0.418),
    )
    coat_high_conf = make_pose(
        track_id=3,
        bbox=(0.135, 0.355, 0.261, 0.821),
        shoulder=(0.190, 0.476),
        hip=(0.188, 0.654),
        conf=0.560,
        nose=(0.203, 0.418),
    )

    # Скомканное одеяло внутри кровати без уверенного лица (head conf = 0.12)
    fold_kps = [(0.0, 0.0, 0.0)] * 17
    fold_kps[0] = (0.45, 0.84, 0.12)
    fold_kps[5] = (0.43, 0.87, 0.65)
    fold_kps[6] = (0.48, 0.88, 0.61)
    fold_kps[7] = (0.41, 0.91, 0.45)
    fold_kps[11] = (0.39, 0.94, 0.52)
    empty_blanket_fold = PoseObservation(
        track_id=4,
        bbox_norm=(0.36, 0.82, 0.52, 0.96),
        conf=0.310,
        keypoints_norm=tuple(fold_kps),
    )

    assert is_person_on_bed(grandma_blanket, real_zones.bed_polygon) is True
    assert is_blanket_sleeper_in_bed(grandma_blanket, real_zones.bed_polygon) is True
    assert is_blanket_sleeper_in_bed(coat_low_conf, real_zones.bed_polygon) is False
    assert is_blanket_sleeper_in_bed(empty_blanket_fold, real_zones.bed_polygon) is False

    too_low_conf = PoseObservation(
        track_id=12,
        bbox_norm=(0.516, 0.797, 0.588, 0.890),
        conf=0.05,
        keypoints_norm=tuple(grandma_kps),
    )
    assert is_blanket_sleeper_in_bed(too_low_conf, real_zones.bed_polygon) is False

    filtered = filter_valid_poses(
        [grandma_blanket, coat_low_conf, coat_high_conf, empty_blanket_fold],
        min_box_conf=0.48,
        zones=real_zones,
    )
    # Куртка с conf=0.41 и скомканное одеяло отсекаются сразу на анатомическом фильтре,
    # а куртка с conf=0.56 далее отсекается темпоральным фильтром неподвижных вещей StaticObjectGate!
    assert {o.track_id for o in filtered} == {1, 3}

    ident = PatientIdentifier(initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(sit_confirm_seconds=1.2, night_strict_mode=False)

    t = 0.0
    for _ in range(8):
        t += 0.25
        roles, cg = ident.update(
            [grandma_blanket, coat_low_conf, coat_high_conf, empty_blanket_fold],
            real_zones,
            now=t,
        )
        ev = fsm.step([grandma_blanket], roles, cg, real_zones, now=t, local_hour=12)
        assert ev is None

    # Строго только бабушка (track 1) прошла все барьеры и перевела автомат в IN_BED с калибровкой сна!
    assert list(roles.keys()) == [1]
    assert roles[1].role == Role.PATIENT
    assert cg is False
    assert fsm.state == PatientState.IN_BED
    assert fsm.baseline_in_bed["calibrated"] is True

    # Бабушка откидывает одеяло и садится на матрасе (плечи поднимаются с y=0.853 до y=0.690) -> SITTING_UP!
    sit_ev = None
    for _ in range(8):
        t += 0.25
        grandma_sitting = make_pose(
            track_id=1,
            bbox=(0.43, 0.60, 0.59, 0.94),
            shoulder=(0.51, 0.69),
            hip=(0.48, 0.89),
            conf=0.74,
            nose=(0.51, 0.63),
        )
        roles, cg = ident.update([grandma_sitting, coat_high_conf], real_zones, now=t)
        ev = fsm.step([grandma_sitting], roles, cg, real_zones, now=t, local_hour=12)
        if ev is not None:
            sit_ev = ev
            break

    assert sit_ev is not None
    assert sit_ev.state == PatientState.SITTING_UP
    assert fsm.state == PatientState.SITTING_UP


# 34. PoseTracker Bed-ROI Fallback: бесшовное восстановление спящей под одеялом и сохранение track_id
def test_pose_tracker_bed_roi_fallback_recovers_blanket_sleeper_and_preserves_track_id(tmp_path: Path) -> None:
    class _FakeTensor:
        def __init__(self, arr: list | np.ndarray) -> None:
            self._arr = np.asarray(arr, dtype=np.float32)

        def int(self) -> "_FakeTensor":
            return _FakeTensor(self._arr.astype(np.int32))

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeBoxes:
        def __init__(self, xyxy: list, conf: list, ids: list | None = None) -> None:
            self.xyxy = _FakeTensor(xyxy)
            self.conf = _FakeTensor(conf)
            self.id = _FakeTensor(ids) if ids is not None else None

        def __len__(self) -> int:
            return int(len(self.conf.numpy()))

    class _FakeKeypoints:
        def __init__(self, xy: list, conf: list) -> None:
            self.xy = _FakeTensor(xy)
            self.conf = _FakeTensor(conf)

    class _FakeResult:
        def __init__(self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None) -> None:
            self.boxes = boxes
            self.keypoints = keypoints

    class _FakeModel:
        def __init__(self) -> None:
            self.track_calls = 0
            self.predict_calls = 0
            self.mode = "full_frame_visible"

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            self.track_calls += 1
            h, w = source.shape[:2]
            if self.mode == "full_frame_visible":
                xy = [[0.0, 0.0]] * 17
                cf = [0.0] * 17
                xy[0], cf[0] = [0.556 * w, 0.819 * h], 0.84
                xy[1], cf[1] = [0.562 * w, 0.816 * h], 0.88
                xy[3], cf[3] = [0.561 * w, 0.825 * h], 0.80
                xy[5], cf[5] = [0.534 * w, 0.858 * h], 0.79
                xy[6], cf[6] = [0.545 * w, 0.848 * h], 0.64
                return [
                    _FakeResult(
                        _FakeBoxes([[0.516 * w, 0.797 * h, 0.588 * w, 0.890 * h]], [0.24], [7]),
                        _FakeKeypoints([xy], [cf]),
                    )
                ]
            if self.mode == "glare_only_coat_on_full":
                # На полном кадре из-за засветки окна видна только куртка на стене (track 42, conf=0.41)
                xy = [[0.0, 0.0]] * 17
                cf = [0.0] * 17
                xy[0], cf[0] = [0.203 * w, 0.418 * h], 0.60
                xy[5], cf[5] = [0.180 * w, 0.476 * h], 0.80
                xy[6], cf[6] = [0.200 * w, 0.476 * h], 0.80
                xy[11], cf[11] = [0.180 * w, 0.654 * h], 0.75
                xy[12], cf[12] = [0.196 * w, 0.654 * h], 0.75
                return [
                    _FakeResult(
                        _FakeBoxes([[0.135 * w, 0.355 * h, 0.261 * w, 0.821 * h]], [0.41], [42]),
                        _FakeKeypoints([xy], [cf]),
                    )
                ]
            return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            self.predict_calls += 1
            if self.mode == "empty_room":
                return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]
            # Окно 0 кропа кровати для [0.20..0.74] x [0.68..0.99] на 1920x1080 начинается в rx1=288, ry1=497
            rx1, ry1 = 288.0, 497.0
            W, H = 1920.0, 1080.0
            xy = [[0.0, 0.0]] * 17
            cf = [0.0] * 17
            xy[0], cf[0] = [0.556 * W - rx1, 0.819 * H - ry1], 0.96
            xy[1], cf[1] = [0.562 * W - rx1, 0.816 * H - ry1], 0.95
            xy[2], cf[2] = [0.560 * W - rx1, 0.814 * H - ry1], 0.89
            xy[5], cf[5] = [0.534 * W - rx1, 0.858 * H - ry1], 0.95
            xy[6], cf[6] = [0.545 * W - rx1, 0.848 * H - ry1], 0.90
            xy[11], cf[11] = [0.453 * W - rx1, 0.961 * H - ry1], 0.49
            bx = [0.421 * W - rx1, 0.797 * H - ry1, 0.588 * W - rx1, 0.988 * H - ry1]
            return [_FakeResult(_FakeBoxes([bx], [0.433]), _FakeKeypoints([xy], [cf]))]

    real_zones = ZonesConfig(
        bed_polygon=[[0.29, 0.73], [0.74, 0.68], [0.67, 0.99], [0.20, 0.99]],
        door_polygon=[[0.01, 0.08], [0.20, 0.28], [0.17, 0.99], [0.01, 0.99]],
    )
    tracker = PoseTracker("yolo11s-pose.pt", tmp_path, conf=0.48, smooth_keypoints=True, roi_probe_interval=0.0)
    fake_model = _FakeModel()
    tracker._model = fake_model  # type: ignore[assignment]
    tracker._runtime_label = "ONNX Runtime (CPU AVX2, passive)"
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Шаг 1: полный кадр сразу нашёл бабушку с track_id=7 -> ROI predict НЕ вызывается (0 мс оверхеда)
    obs1 = tracker.infer(frame, zones=real_zones)
    assert len(obs1) == 1
    assert obs1[0].track_id == 7
    assert fake_model.predict_calls == 0

    # Шаг 2: засветка скрыла бабушку на полном кадре (там только куртка 42) -> ROI fallback вытаскивает бабушку с track_id=7!
    fake_model.mode = "glare_only_coat_on_full"
    obs2 = tracker.infer(frame, zones=real_zones)
    assert fake_model.predict_calls == 1
    assert len(obs2) == 1
    assert obs2[0].track_id == 7
    assert is_person_on_bed(obs2[0], real_zones.bed_polygon) is True

    # Шаг 3: кровать реально пуста -> оба прохода возвращают пустой список без фантомов
    fake_model.mode = "empty_room"
    obs3 = tracker.infer(frame, zones=real_zones)
    assert obs3 == []


# 35. Защита от окклюзии родственником у кровати: переход в CAREGIVER_ASSIST вместо ложного MISSING
def test_visitor_near_bed_occluding_patient_switches_to_caregiver_assist_not_missing() -> None:
    ident = PatientIdentifier(initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(missing_confirm_seconds=3.0, suppress_sitting_when_caregiver=True)

    t = 0.0
    for _ in range(6):
        t += 0.25
        grandma = make_pose(1, (0.36, 0.42, 0.68, 0.58), (0.44, 0.49), (0.60, 0.50))
        roles, cg = ident.update([grandma], BED_ZONE, now=t)
        assert fsm.step([grandma], roles, cg, BED_ZONE, now=t) is None
    assert fsm.state == PatientState.IN_BED

    # Родственница входит через дверь и подходит к бортику кровати
    t += 0.25
    v_door = make_pose(2, (0.88, 0.18, 0.98, 0.82), (0.93, 0.28), (0.93, 0.54))
    ident.update([grandma, v_door], BED_ZONE, now=t)
    t += 0.25
    v_near_bed = make_pose(2, (0.22, 0.34, 0.36, 0.88), (0.29, 0.44), (0.29, 0.66), ankle=(0.29, 0.86))
    roles, cg = ident.update([grandma, v_near_bed], BED_ZONE, now=t)
    assert roles[2].role == Role.VISITOR
    assert cg is True

    # Родственница на 15 секунд (>> missing_confirm_seconds=3.0 с) полностью закрыла бабушку спиной у кровати
    for _ in range(30):
        t += 0.5
        roles, cg = ident.update([v_near_bed], BED_ZONE, now=t)
        ev = fsm.step([v_near_bed], roles, cg, BED_ZONE, now=t)
        assert cg is True
        assert ev is None, "При ухаживающем родственнике у кровати НЕ должен срабатывать ложный MISSING!"
        assert fsm.state == PatientState.CAREGIVER_ASSIST
        assert fsm.status_reason == "рядом родственник · уход у кровати"

    # Родственница уходит обратно через полигон двери и покидает комнату -> бесшовный возврат в IN_BED
    t += 0.4
    v_exit = make_pose(2, (0.88, 0.18, 0.98, 0.82), (0.93, 0.28), (0.93, 0.54))
    ident.update([v_exit], BED_ZONE, now=t)
    t += 2.6
    for _ in range(4):
        t += 0.3
        grandma = make_pose(1, (0.36, 0.42, 0.68, 0.58), (0.44, 0.49), (0.60, 0.50))
        roles, cg = ident.update([grandma], BED_ZONE, now=t)
        ev = fsm.step([grandma], roles, cg, BED_ZONE, now=t)
        assert ev is None
    assert ident.door_visit_active is False
    assert fsm.state == PatientState.IN_BED


# 36. Совместное нахождение у кровати в PoseTracker: ROI-спящая + полнокадровый гость без коллизии track_id
def test_pose_tracker_co_presence_merges_roi_sleeper_with_full_frame_visitor_without_id_collision(
    tmp_path: Path,
) -> None:
    class _FakeTensor:
        def __init__(self, arr: list | np.ndarray) -> None:
            self._arr = np.asarray(arr, dtype=np.float32)

        def int(self) -> "_FakeTensor":
            return _FakeTensor(self._arr.astype(np.int32))

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeBoxes:
        def __init__(self, xyxy: list, conf: list, ids: list | None = None) -> None:
            self.xyxy = _FakeTensor(xyxy)
            self.conf = _FakeTensor(conf)
            self.id = _FakeTensor(ids) if ids is not None else None

        def __len__(self) -> int:
            return int(len(self.conf.numpy()))

    class _FakeKeypoints:
        def __init__(self, xy: list, conf: list) -> None:
            self.xy = _FakeTensor(xy)
            self.conf = _FakeTensor(conf)

    class _FakeResult:
        def __init__(self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None) -> None:
            self.boxes = boxes
            self.keypoints = keypoints

    class _FakeCoPresenceModel:
        def __init__(self) -> None:
            self.phase = "grandma_alone"
            self.predict_calls = 0

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            h, w = source.shape[:2]
            if self.phase == "grandma_alone":
                xy = [[0.0, 0.0]] * 17
                cf = [0.0] * 17
                xy[4], cf[4] = [0.548 * w, 0.520 * h], 0.82
                xy[5], cf[5] = [0.530 * w, 0.533 * h], 0.90
                xy[6], cf[6] = [0.550 * w, 0.537 * h], 0.97
                xy[7], cf[7] = [0.522 * w, 0.560 * h], 0.68
                xy[8], cf[8] = [0.555 * w, 0.555 * h], 0.97
                xy[11], cf[11] = [0.525 * w, 0.620 * h], 0.92
                xy[12], cf[12] = [0.548 * w, 0.625 * h], 0.95
                return [
                    _FakeResult(
                        _FakeBoxes([[0.515 * w, 0.501 * h, 0.577 * w, 0.656 * h]], [0.252], [7]),
                        _FakeKeypoints([xy], [cf]),
                    )
                ]
            # На шаге 2 гостья села на левый край кровати и перехватила track_id=7 (вертикальный торс, ноги на полу)
            v_xy = [[0.0, 0.0]] * 17
            v_cf = [0.0] * 17
            v_xy[0], v_cf[0] = [0.345 * w, 0.445 * h], 0.96
            v_xy[1], v_cf[1] = [0.340 * w, 0.440 * h], 0.95
            v_xy[2], v_cf[2] = [0.350 * w, 0.440 * h], 0.95
            v_xy[5], v_cf[5] = [0.330 * w, 0.500 * h], 0.98
            v_xy[6], v_cf[6] = [0.365 * w, 0.502 * h], 0.98
            v_xy[11], v_cf[11] = [0.332 * w, 0.680 * h], 0.96
            v_xy[12], v_cf[12] = [0.365 * w, 0.682 * h], 0.96
            v_xy[15], v_cf[15] = [0.340 * w, 0.890 * h], 0.92
            v_xy[16], v_cf[16] = [0.365 * w, 0.892 * h], 0.92
            return [
                _FakeResult(
                    _FakeBoxes([[0.295 * w, 0.404 * h, 0.411 * w, 0.932 * h]], [0.84], [7]),
                    _FakeKeypoints([v_xy], [v_cf]),
                )
            ]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            self.predict_calls += 1
            # Для зон bed_polygon min_x=0.26, min_y=0.41 окно 0 начинается в rx1=403, ry1=205 на 1920x1080
            rx1, ry1 = 403.0, 205.0
            W, H = 1920.0, 1080.0
            # В ROI видны и дубликат сидящей гостьи слева, и спящая справа бабушка
            dup_xy = [[0.0, 0.0]] * 17
            dup_cf = [0.0] * 17
            dup_xy[0], dup_cf[0] = [0.346 * W - rx1, 0.446 * H - ry1], 0.95
            dup_xy[1], dup_cf[1] = [0.341 * W - rx1, 0.441 * H - ry1], 0.94
            dup_xy[5], dup_cf[5] = [0.330 * W - rx1, 0.500 * H - ry1], 0.96
            dup_xy[6], dup_cf[6] = [0.365 * W - rx1, 0.502 * H - ry1], 0.96
            dup_xy[11], dup_cf[11] = [0.332 * W - rx1, 0.680 * H - ry1], 0.92
            dup_bx = [0.295 * W - rx1, 0.404 * H - ry1, 0.411 * W - rx1, 0.790 * H - ry1]

            g_xy = [[0.0, 0.0]] * 17
            g_cf = [0.0] * 17
            g_xy[4], g_cf[4] = [0.548 * W - rx1, 0.520 * H - ry1], 0.84
            g_xy[5], g_cf[5] = [0.530 * W - rx1, 0.533 * H - ry1], 0.91
            g_xy[6], g_cf[6] = [0.550 * W - rx1, 0.537 * H - ry1], 0.96
            g_xy[8], g_cf[8] = [0.555 * W - rx1, 0.555 * H - ry1], 0.90
            g_xy[11], g_cf[11] = [0.525 * W - rx1, 0.620 * H - ry1], 0.88
            g_xy[12], g_cf[12] = [0.548 * W - rx1, 0.625 * H - ry1], 0.91
            g_bx = [0.515 * W - rx1, 0.501 * H - ry1, 0.577 * W - rx1, 0.656 * H - ry1]

            return [
                _FakeResult(
                    _FakeBoxes([dup_bx, g_bx], [0.88, 0.61]),
                    _FakeKeypoints([dup_xy, g_xy], [dup_cf, g_cf]),
                )
            ]

    zones_1303 = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )
    tracker = PoseTracker("yolo11s-pose.pt", tmp_path, conf=0.48, smooth_keypoints=False, roi_probe_interval=0.0)
    fake_model = _FakeCoPresenceModel()
    tracker._model = fake_model  # type: ignore[assignment]
    tracker._roi_model = fake_model  # type: ignore[assignment]
    tracker._runtime_label = "ONNX Runtime (CPU AVX2, passive)"
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Шаг 1: бабушка спит с рукой на лице (открыто ухо kp4=0.82 + оба плеча) -> видна сразу на полном кадре как #7
    obs_alone = tracker.infer(frame, zones=zones_1303)
    assert len(obs_alone) == 1
    assert obs_alone[0].track_id == 7
    assert fake_model.predict_calls == 0

    # Шаг 2: гостья села на левый край кровати с track_id=7 -> ROI НЕ отключился и выдал бабушке бесконфликтный #9001!
    fake_model.phase = "visitor_on_edge"
    obs_both = tracker.infer(frame, zones=zones_1303)
    assert fake_model.predict_calls >= 1
    assert len(obs_both) == 2
    ids = {o.track_id for o in obs_both}
    assert ids == {7, 9001}


# 37. Регрессионный тест инцидента 13:02–13:07: перескок ID у гостя, сидящего со свешенными ногами на краю кровати,
# НИКОГДА не превращает гостя в PATIENT (bed_sleep_lock) и НЕ загрязняет patient_anchor.json!
def test_visitor_sitting_on_bed_edge_id_switch_never_becomes_patient(tmp_path: Path) -> None:
    anchor_file = tmp_path / "patient_anchor.json"
    ident = PatientIdentifier(
        bed_lock_seconds=20.0,
        initial_bed_lock_seconds=2.0,
        state_path=anchor_file,
    )
    fsm = FallStateMachine(sit_confirm_seconds=1.2, missing_confirm_seconds=5.0)

    zones_1303 = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )
    grandma_color = (210, 195, 60)   # светло-бирюзовое/жёлтое одеяло/одежда бабушки
    visitor_color = (45, 35, 165)    # контрастная тёмно-бордовая одежда родственницы

    # 1. Бабушка спокойно спит на правой половине кровати -> захватывается как PATIENT и сохраняет якорь
    t = 0.0
    for _ in range(12):
        t += 0.25
        g_sleep = make_pose(1, (0.48, 0.48, 0.64, 0.64), (0.53, 0.53), (0.56, 0.61))
        fr = make_colored_room_frame([(g_sleep, grandma_color)])
        roles, cg = ident.update([g_sleep], zones_1303, now=t, frame_bgr=fr)
        assert fsm.step([g_sleep], roles, cg, zones_1303, now=t) is None

    assert roles[1].role == Role.PATIENT
    saved_before = np.asarray(json.loads(anchor_file.read_text(encoding="utf-8"))["anchor_vec"], dtype=np.float32)

    # 2. Родственница входит через дверь (#12 -> Role.VISITOR) и садится на левый край кровати со спущенными на пол ногами
    t += 0.5
    v_door = make_pose(12, (0.12, 0.28, 0.24, 0.86), (0.18, 0.38), (0.18, 0.60), ankle=(0.18, 0.84))
    fr_door = make_colored_room_frame([(g_sleep, grandma_color), (v_door, visitor_color)])
    ident.update([g_sleep, v_door], zones_1303, now=t, frame_bgr=fr_door)

    t += 0.5
    v_sit_12 = make_pose(12, (0.295, 0.404, 0.411, 0.932), (0.35, 0.49), (0.36, 0.68), ankle=(0.35, 0.90))
    fr_sit_12 = make_colored_room_frame([(v_sit_12, visitor_color)])
    roles_12, cg_12 = ident.update([v_sit_12], zones_1303, now=t, frame_bgr=fr_sit_12)
    assert roles_12[12].role == Role.VISITOR
    assert cg_12 is True

    # 3. ByteTrack обрывает #12 и выдаёт сидящей на том же краю кровати родственнице новый track_id=30 на 400 секунд!
    for _ in range(400):
        t += 1.0
        v_sit_30 = make_pose(30, (0.298, 0.406, 0.414, 0.932), (0.352, 0.492), (0.362, 0.682), ankle=(0.352, 0.90))
        fr_sit_30 = make_colored_room_frame([(v_sit_30, visitor_color)])
        roles_30, cg_30 = ident.update([v_sit_30], zones_1303, now=t, frame_bgr=fr_sit_30)
        ev = fsm.step([v_sit_30], roles_30, cg_30, zones_1303, now=t)
        assert roles_30[30].role == Role.VISITOR
        assert roles_30[30].reason == "visitor_id_switch_continuity"
        assert roles_30[30].bed_accum_sec == 0.0
        assert cg_30 is True
        assert ev is None
        assert fsm.state == PatientState.CAREGIVER_ASSIST

    # 4. Сидящий со свешенными ногами НЕ копит спальный bed_accum_sec, а по Аксиоме 1 без дверного визита
    # единственная сидящая в кровати бабушка охраняется как PATIENT (никогда не переводится в VISITOR!)
    ident_fresh = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=1.0)
    t2 = 0.0
    for step_k in range(8):
        t2 += 0.5
        dx = 0.06 if step_k >= 1 else 0.0
        sit_edge = make_pose(77, (0.29 + dx, 0.40, 0.41 + dx, 0.93), (0.35 + dx, 0.49), (0.36 + dx, 0.68), ankle=(0.35 + dx, 0.90))
        roles_f, _ = ident_fresh.update([sit_edge], zones_1303, now=t2)
        if 77 in roles_f:
            assert roles_f[77].role != Role.VISITOR
            assert roles_f[77].bed_accum_sec == 0.0
    assert roles_f[77].role == Role.PATIENT

    # 5. Проверяем защиту якоря от загрязнения чужой одеждой даже при ручной подстановке без совпадения цвета
    saved_after = np.asarray(json.loads(anchor_file.read_text(encoding="utf-8"))["anchor_vec"], dtype=np.float32)
    assert cosine_sim(saved_before, saved_after) >= 0.999


# 38. Регрессия боевого снимка diag_1400.jpg: спящая одна бабушка НИКОГДА не получает visitor_id_switch_continuity
# и при любом ошибочном VISITOR самоисцеляется за <= 3.0 с через sole_bed_sleeper_recovery!
def test_diag_1400_sole_bed_sleeper_never_poisoned_by_recent_visitors_and_self_heals_in_3s() -> None:
    zones_1400 = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )
    ident = PatientIdentifier(
        bed_lock_seconds=15.0,
        initial_bed_lock_seconds=2.0,
        door_visit_idle_timeout_sec=2.0,
    )
    fsm = FallStateMachine(sit_confirm_seconds=1.5, missing_confirm_seconds=5.0)
    shared_blanket_color = (80, 155, 195)

    # 1. Родственник вошёл из двери (#5) и постоял/наклонился у самого края кровати, оставив запись в _recent_visitors
    t = 0.0
    v_door = make_pose(5, (0.12, 0.26, 0.24, 0.84), (0.18, 0.36), (0.18, 0.58), ankle=(0.18, 0.82))
    fr1 = make_colored_room_frame([(v_door, shared_blanket_color)])
    ident.update([v_door], zones_1400, now=t, frame_bgr=fr1)

    t += 0.5
    v_edge = make_pose(5, (0.33, 0.36, 0.46, 0.88), (0.39, 0.46), (0.40, 0.66), ankle=(0.39, 0.86))
    fr2 = make_colored_room_frame([(v_edge, shared_blanket_color)])
    r2, _ = ident.update([v_edge], zones_1400, now=t, frame_bgr=fr2)
    assert r2[5].role == Role.VISITOR
    assert 5 in ident._recent_visitors

    # 2. Родственник ушёл, а в комнате одна бабушка лежит горизонтально на матрасе (точные координаты diag_1400.jpg:
    # bbox = [0.3401, 0.5237, 0.5911, 0.6875], angle ~ 77.9 град, рядом с прошлой позицией #5 и тем же тоном ткани!)
    for _ in range(10):
        t += 0.25
        g_sleep = make_pose(
            track_id=10001,
            bbox=(0.3401, 0.5237, 0.5911, 0.6875),
            shoulder=(0.545, 0.575),
            hip=(0.415, 0.603),
            conf=0.31,
            nose=(0.568, 0.560),
        )
        fr_g = make_colored_room_frame([(g_sleep, shared_blanket_color)])
        roles, cg = ident.update([g_sleep], zones_1400, now=t, frame_bgr=fr_g)
        ev = fsm.step([g_sleep], roles, cg, zones_1400, now=t)
        assert roles[10001].reason != "visitor_id_switch_continuity"
        assert 10001 not in ident._recent_visitors
        assert ev is None

    assert roles[10001].role == Role.PATIENT
    assert cg is False
    assert fsm.state == PatientState.IN_BED

    # 3. Искусственно имитируем залипший ложный VISITOR ("visitor_id_switch_continuity") на новом треке спящей бабушки
    # даже при отсутствии/потере предыдущего якоря одежды: sole_bed_sleeper_recovery обязан за <= 3.0 с вернуть PATIENT!
    t += 4.0
    g_sleep_err = make_pose(
        track_id=10002,
        bbox=(0.3401, 0.5237, 0.5911, 0.6875),
        shoulder=(0.545, 0.575),
        hip=(0.415, 0.603),
        conf=0.31,
        nose=(0.568, 0.560),
    )
    fr_err = make_colored_room_frame([(g_sleep_err, (210, 45, 45))])
    ident._anchor_vec = None
    ident._ever_locked_patient = False
    ident.update([g_sleep_err], zones_1400, now=t, frame_bgr=fr_err)
    # Принудительно портим роль трека в памяти, как это произошло в 14:00 до фикса
    ident._tracks[10002].role = Role.VISITOR
    ident._tracks[10002].reason = "visitor_id_switch_continuity"
    ident._tracks[10002].bed_accum_sec = 0.0
    ident._last_patient_track_id = None

    elapsed = 0.0
    recovered = False
    while elapsed < 3.0:
        t += 0.25
        elapsed += 0.25
        roles_rec, cg_rec = ident.update([g_sleep_err], zones_1400, now=t, frame_bgr=fr_err)
        fsm.step([g_sleep_err], roles_rec, cg_rec, zones_1400, now=t)
        if roles_rec[10002].role == Role.PATIENT:
            recovered = True
            break

    assert recovered is True
    assert elapsed <= 3.0
    assert roles_rec[10002].reason == "sole_bed_sleeper_recovery"
    assert cg_rec is False
    assert fsm.state == PatientState.IN_BED
    assert 10002 not in ident._recent_visitors


# 39. Стабильные монотонные ID (_anon_tracks >= 10001 без 10000+i) и сквозной _last_bed_track_id в PoseTracker
def test_pose_tracker_anon_tracks_monotonic_ids_and_persistent_bed_track_id(tmp_path: Path) -> None:
    class _FakeTensor:
        def __init__(self, arr: list | np.ndarray) -> None:
            self._arr = np.asarray(arr, dtype=np.float32)

        def int(self) -> "_FakeTensor":
            return _FakeTensor(self._arr.astype(np.int32))

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeBoxes:
        def __init__(self, xyxy: list, conf: list, ids: list | None = None) -> None:
            self.xyxy = _FakeTensor(xyxy)
            self.conf = _FakeTensor(conf)
            self.id = _FakeTensor(ids) if ids is not None else None

        def __len__(self) -> int:
            return int(len(self.conf.numpy()))

    class _FakeKeypoints:
        def __init__(self, xy: list, conf: list) -> None:
            self.xy = _FakeTensor(xy)
            self.conf = _FakeTensor(conf)

    class _FakeResult:
        def __init__(self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None) -> None:
            self.boxes = boxes
            self.keypoints = keypoints

    def _sleep_kps(w: float, h: float) -> tuple[list, list]:
        xy = [[0.0, 0.0]] * 17
        cf = [0.0] * 17
        xy[0], cf[0] = [0.555 * w, 0.560 * h], 0.88
        xy[1], cf[1] = [0.560 * w, 0.556 * h], 0.86
        xy[5], cf[5] = [0.535 * w, 0.575 * h], 0.91
        xy[6], cf[6] = [0.548 * w, 0.582 * h], 0.89
        xy[11], cf[11] = [0.420 * w, 0.602 * h], 0.85
        xy[12], cf[12] = [0.432 * w, 0.610 * h], 0.84
        return xy, cf

    def _guest_kps(w: float, h: float, cx: float) -> tuple[list, list]:
        xy = [[0.0, 0.0]] * 17
        cf = [0.0] * 17
        xy[0], cf[0] = [cx * w, 0.36 * h], 0.93
        xy[1], cf[1] = [(cx - 0.006) * w, 0.354 * h], 0.91
        xy[2], cf[2] = [(cx + 0.006) * w, 0.354 * h], 0.91
        xy[5], cf[5] = [(cx - 0.02) * w, 0.44 * h], 0.95
        xy[6], cf[6] = [(cx + 0.02) * w, 0.44 * h], 0.95
        xy[11], cf[11] = [(cx - 0.02) * w, 0.64 * h], 0.92
        xy[12], cf[12] = [(cx + 0.02) * w, 0.64 * h], 0.92
        return xy, cf

    class _FakeAnonModel:
        def __init__(self) -> None:
            self.step = 1

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            h, w = source.shape[:2]
            s_xy, s_cf = _sleep_kps(w, h)
            s_bx = [0.340 * w, 0.524 * h, 0.591 * w, 0.688 * h]
            if self.step == 0:
                g_xy, g_cf = _guest_kps(w, h, 0.14)
                g_bx = [0.09 * w, 0.30 * h, 0.19 * w, 0.84 * h]
                return [_FakeResult(_FakeBoxes([g_bx, s_bx], [0.62, 0.29], None), _FakeKeypoints([g_xy, s_xy], [g_cf, s_cf]))]
            if self.step == 1:
                # Два анонимных бокса с ids=None: шагнувший гость слева + спящая бабушка справа
                g_xy, g_cf = _guest_kps(w, h, 0.20)
                g_bx = [0.15 * w, 0.30 * h, 0.25 * w, 0.84 * h]
                return [_FakeResult(_FakeBoxes([g_bx, s_bx], [0.62, 0.29], None), _FakeKeypoints([g_xy, s_xy], [g_cf, s_cf]))]
            if self.step == 2:
                # Порядок в выдаче YOLO поменялся местами, ids по-прежнему None -> ID обязаны сохраниться по IoU/dist!
                g_xy, g_cf = _guest_kps(w, h, 0.22)
                g_bx = [0.17 * w, 0.30 * h, 0.27 * w, 0.84 * h]
                return [_FakeResult(_FakeBoxes([s_bx, g_bx], [0.30, 0.64], None), _FakeKeypoints([s_xy, g_xy], [s_cf, g_cf]))]
            if self.step == 3:
                # На полном кадре бабушка пропала (пусто), вытаскивается через _infer_bed_roi_fallback -> тот же bed ID!
                return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]
            if self.step == 4:
                g_xy, g_cf = _guest_kps(w, h, 0.12)
                g_bx = [0.07 * w, 0.30 * h, 0.17 * w, 0.84 * h]
                return [_FakeResult(_FakeBoxes([g_bx], [0.60], None), _FakeKeypoints([g_xy], [g_cf]))]
            # Шаг 5: после истечения TTL анонимных треков новый гость у двери получает НОВЫЙ монотонный ID (не старый!)
            g_xy, g_cf = _guest_kps(w, h, 0.19)
            g_bx = [0.14 * w, 0.30 * h, 0.24 * w, 0.84 * h]
            return [_FakeResult(_FakeBoxes([g_bx], [0.60], None), _FakeKeypoints([g_xy], [g_cf]))]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            rx1, ry1 = 403.0, 205.0
            W, H = 1920.0, 1080.0
            s_xy, s_cf = _sleep_kps(W, H)
            roi_xy = [[pt[0] - rx1 if cf_v > 0 else 0.0, pt[1] - ry1 if cf_v > 0 else 0.0] for pt, cf_v in zip(s_xy, s_cf)]
            roi_bx = [0.340 * W - rx1, 0.524 * H - ry1, 0.591 * W - rx1, 0.688 * H - ry1]
            return [_FakeResult(_FakeBoxes([roi_bx], [0.46], None), _FakeKeypoints([roi_xy], [s_cf]))]

    zones_1400 = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )
    tracker = PoseTracker("yolo11s-pose.pt", tmp_path, conf=0.48, smooth_keypoints=False, roi_probe_interval=0.0)
    fake = _FakeAnonModel()
    tracker._model = fake  # type: ignore[assignment]
    tracker._roi_model = fake  # type: ignore[assignment]
    tracker._runtime_label = "ONNX Runtime (CPU AVX2, passive)"
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    fake.step = 0
    tracker.infer(frame, zones=zones_1400)
    fake.step = 1
    obs1 = tracker.infer(frame, zones=zones_1400)
    assert len(obs1) == 2
    assert all(o.track_id != 10000 and o.track_id >= 10001 for o in obs1)
    guest_id_1 = min(obs1, key=lambda o: o.body_center[0]).track_id
    bed_id_1 = max(obs1, key=lambda o: o.body_center[0]).track_id
    assert guest_id_1 != bed_id_1

    # Шаг 2: даже при перестановке порядка боксов в ids=None оба трека удерживают свои прежние ID
    fake.step = 2
    obs2 = tracker.infer(frame, zones=zones_1400)
    assert len(obs2) == 2
    guest_id_2 = min(obs2, key=lambda o: o.body_center[0]).track_id
    bed_id_2 = max(obs2, key=lambda o: o.body_center[0]).track_id
    assert guest_id_2 == guest_id_1
    assert bed_id_2 == bed_id_1

    # Шаг 3: ROI fallback бесшовно наследует тот же _last_bed_track_id
    fake.step = 3
    obs3 = tracker.infer(frame, zones=zones_1400)
    assert len(obs3) == 1
    assert obs3[0].track_id == bed_id_1

    # Шаг 4..5: состариваем историю _anon_tracks и _liveness_gate -> новый анонимный гость получает строго новый ID!
    for rec in tracker._anon_tracks.values():
        rec.last_ts -= 10.0
    tracker._liveness_gate.reset()
    fake.step = 4
    tracker.infer(frame, zones=zones_1400)
    fake.step = 5
    obs4 = tracker.infer(frame, zones=zones_1400)
    guest_only = [o for o in obs4 if o.body_center[0] < 0.25]
    assert len(guest_only) == 1
    assert guest_only[0].track_id > max(guest_id_1, bed_id_1)


# 40. Ручная отметка «Свой» (force_assign_visitor): 8-часовая галерея visitor_vecs, очистка грязного якоря и автоповышение бабушки
def test_force_assign_visitor_saves_8h_outfit_cleans_tainted_anchor_and_promotes_bed_sleeper(
    tmp_path: Path,
) -> None:
    anchor_file = tmp_path / "patient_anchor.json"
    clock_now = [1_700_000_000.0]
    ident = PatientIdentifier(
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=2.0,
        state_path=anchor_file,
        wall_clock=lambda: clock_now[0],
    )
    grandma_shirt = (195, 175, 55)
    visitor_sweater = (35, 30, 175)

    # Имитируем ошибку прошлого: родственник (#20) случайно был назначен бабушкой и загрязнил _anchor_vec,
    # пока настоящая бабушка (#21) спокойно лежит на кровати (4 кадра подтверждённого шага #20 вне матраса)
    t = 10.0
    g_obs = make_pose(21, (0.46, 0.48, 0.66, 0.64), (0.53, 0.54), (0.60, 0.56))
    for step_x, step_dt in ((0.22, -0.9), (0.24, -0.6), (0.26, -0.3)):
        v_step = make_pose(
            20,
            (step_x, 0.36, step_x + 0.12, 0.90),
            (step_x + 0.06, 0.45),
            (step_x + 0.06, 0.68),
            ankle=(step_x + 0.06, 0.88),
        )
        ident.update(
            [v_step, g_obs],
            BED_ZONE,
            now=t + step_dt,
            frame_bgr=make_colored_room_frame([(v_step, visitor_sweater), (g_obs, grandma_shirt)]),
        )
    v_obs = make_pose(20, (0.28, 0.36, 0.40, 0.90), (0.34, 0.45), (0.34, 0.68), ankle=(0.34, 0.88))
    frame = make_colored_room_frame([(v_obs, visitor_sweater), (g_obs, grandma_shirt)])

    ident.update([v_obs, g_obs], BED_ZONE, now=t, frame_bgr=frame)
    assert ident.force_assign_patient(20, now=t) is True
    assert ident.has_appearance_anchor is True

    # Оператор нажимает «Свой» на треке #20!
    assert ident.force_assign_visitor(20, now=t + 0.2) is True
    assert ident.visitor_outfit_count == 1
    assert ident._tracks[20].role == Role.VISITOR
    assert ident._tracks[20].reason == "manual_visitor_override"
    # Лежащая на кровати бабушка (#21) мгновенно повышена до Role.PATIENT с чистым якорем её рубашки!
    assert ident._tracks[21].role == Role.PATIENT
    assert ident._tracks[21].reason == "promoted_after_visitor_override"

    raw_saved = json.loads(anchor_file.read_text(encoding="utf-8"))
    assert len(raw_saved["visitor_vecs"]) == 1

    # Проверяем, что воротниковый дескриптор различает две разные рубашки даже под одинаковым одеялом на животе
    frame_same_blanket_a = np.full((360, 640, 3), 60, dtype=np.uint8)
    frame_same_blanket_b = np.full((360, 640, 3), 60, dtype=np.uint8)
    test_obs = make_pose(99, (0.35, 0.40, 0.65, 0.70), (0.48, 0.46), (0.50, 0.62))
    # Одинаковый низ торса (одеяло), но контрастный верх вокруг плеч!
    frame_same_blanket_a[140:195, 250:370] = (210, 190, 40)
    frame_same_blanket_a[195:255, 250:370] = (90, 90, 90)
    frame_same_blanket_b[140:195, 250:370] = (30, 40, 200)
    frame_same_blanket_b[195:255, 250:370] = (90, 90, 90)
    vec_a = compute_torso_appearance(frame_same_blanket_a, test_obs)
    vec_b = compute_torso_appearance(frame_same_blanket_b, test_obs)
    assert cosine_sim(vec_a, vec_b) < 0.72

    # Перезапускаем контейнер через 2 часа (< 8 ч TTL): родственник в том же свитере на новом track_id=88
    # мгновенно распознаётся как Role.VISITOR ("visitor_outfit_match")!
    clock_now[0] += 2.0 * 3600.0
    ident_reloaded = PatientIdentifier(
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=2.0,
        state_path=anchor_file,
        wall_clock=lambda: clock_now[0],
    )
    assert ident_reloaded.visitor_outfit_count == 1
    for idx_step, vx in enumerate((0.40, 0.42, 0.44), start=1):
        v_step88 = make_pose(
            88,
            (vx, 0.34, vx + 0.12, 0.86),
            (vx + 0.06, 0.43),
            (vx + 0.06, 0.65),
            ankle=(vx + 0.06, 0.84),
        )
        ident_reloaded.update(
            [v_step88],
            BED_ZONE,
            now=0.25 * idx_step,
            frame_bgr=make_colored_room_frame([(v_step88, visitor_sweater)]),
        )
    v_new_step4 = make_pose(88, (0.46, 0.34, 0.58, 0.86), (0.52, 0.43), (0.52, 0.65), ankle=(0.52, 0.84))
    fr_re4 = make_colored_room_frame([(v_new_step4, visitor_sweater)])
    roles_re, _ = ident_reloaded.update([v_new_step4], BED_ZONE, now=1.25, frame_bgr=fr_re4)
    assert roles_re[88].role == Role.VISITOR
    assert roles_re[88].reason == "visitor_outfit_match"

    # А через 9 часов (> 8 ч TTL) запись одежды гостя автоматически истекает
    clock_now[0] += 7.0 * 3600.0
    assert ident_reloaded.visitor_outfit_count == 0


# 41. Скидывание одеяла при посадке с активным _anchor_vec: spatiotemporal_bed_continuity сохраняет роль бабушки
def test_throwing_off_blanket_when_sitting_up_keeps_patient_role_via_spatiotemporal_continuity() -> None:
    ident = PatientIdentifier(bed_lock_seconds=12.0, initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(sit_confirm_seconds=1.2, night_strict_mode=False)

    yellow_blanket = (40, 205, 225)  # Тёпло-жёлтое одеяло во время сна
    blue_shirt = (210, 55, 35)       # Контрастная синяя ночная рубашка после скидывания одеяла

    # 1. Бабушка спит под жёлтым одеялом (#1) -> захватывается как PATIENT и сохраняет _anchor_vec
    t = 0.0
    for _ in range(8):
        t += 0.25
        sleep_obs = make_pose(1, (0.34, 0.44, 0.66, 0.60), (0.42, 0.50), (0.56, 0.52))
        fr_s = make_colored_room_frame([(sleep_obs, yellow_blanket)])
        roles, cg = ident.update([sleep_obs], BED_ZONE, now=t, frame_bgr=fr_s)
        assert fsm.step([sleep_obs], roles, cg, BED_ZONE, now=t) is None

    assert roles[1].role == Role.PATIENT
    assert ident.has_appearance_anchor is True
    assert fsm.state == PatientState.IN_BED

    # 2. Бабушка скидывает одеяло и садится на матрасе со свешенными ногами; из-за резкой смены силуэта
    # ByteTrack выдал новый track_id=9, а цвет торса сменился с жёлтого на синий (cosine_sim << 0.78)!
    sit_event = None
    for step_i in range(8):
        t += 0.25
        sit_obs = make_pose(
            track_id=9,
            bbox=(0.38, 0.30, 0.56, 0.80),
            shoulder=(0.47, 0.36),
            hip=(0.47, 0.56),
            ankle=(0.48, 0.78),
        )
        fr_sit = make_colored_room_frame([(sit_obs, blue_shirt)])
        roles_sit, cg_sit = ident.update([sit_obs], BED_ZONE, now=t, frame_bgr=fr_sit)
        assert roles_sit[9].role == Role.PATIENT
        if step_i == 0:
            assert roles_sit[9].reason == "spatiotemporal_bed_continuity"
        ev = fsm.step([sit_obs], roles_sit, cg_sit, BED_ZONE, now=t)
        if ev is not None:
            sit_event = ev

    assert sit_event is not None
    assert sit_event.state == PatientState.SITTING_UP
    assert fsm.state == PatientState.SITTING_UP


# 42. Мгновенный отсев входа у двери/края без 8-секундной задержки и двухперсонный решатель _arbitrate_multi_person_roles
def test_multi_person_arbitration_and_instant_border_standing_entry_without_8s_block() -> None:
    zones_room = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )
    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=1.0)

    # 1. Бабушка спала в правой части кровати (x=0.53, y=0.56)
    t = 0.0
    for _ in range(6):
        t += 0.25
        g = make_pose(1, (0.46, 0.48, 0.64, 0.64), (0.52, 0.53), (0.58, 0.56))
        roles, _ = ident.update([g], zones_room, now=t)
    assert roles[1].role == Role.PATIENT

    # 2. Всего через 0.5 с (<< старых 8.0 с!) бабушка скрыта одеялом, а в левом дверном коридоре (x=0.25..0.29)
    # появляется вертикально стоящий человек (#33) -> мгновенно получает Role.VISITOR!
    t += 0.5
    entrant_t0 = make_pose(33, (0.19, 0.26, 0.29, 0.84), (0.24, 0.35), (0.24, 0.57), ankle=(0.24, 0.82))
    ident.update([entrant_t0], zones_room, now=t)
    t += 0.3
    entrant_t1 = make_pose(33, (0.25, 0.28, 0.35, 0.86), (0.30, 0.37), (0.30, 0.59), ankle=(0.30, 0.84))
    roles_ent, _ = ident.update([entrant_t1], zones_room, now=t)
    assert roles_ent[33].role == Role.VISITOR
    assert roles_ent[33].reason in ("door_entry", "border_standing_entry")

    # 3. Одновременное появление двух новых треков: лежащий на кровати (#40) и стоящий у края (#41) ->
    # двухперсонный арбитраж _arbitrate_multi_person_roles однозначно отдаёт PATIENT лежащему #40 и VISITOR #41!
    t += 0.4
    cand_bed = make_pose(40, (0.45, 0.48, 0.64, 0.64), (0.51, 0.53), (0.58, 0.56))
    cand_stand_a = make_pose(41, (0.20, 0.32, 0.32, 0.88), (0.26, 0.42), (0.26, 0.64), ankle=(0.26, 0.86))
    ident.update([cand_bed, cand_stand_a], zones_room, now=t)
    t += 0.4
    cand_stand_b = make_pose(41, (0.26, 0.32, 0.38, 0.88), (0.32, 0.42), (0.32, 0.64), ankle=(0.32, 0.86))
    roles_pair, cg_pair = ident.update([cand_bed, cand_stand_b], zones_room, now=t)
    assert roles_pair[40].role == Role.PATIENT
    assert roles_pair[41].role == Role.VISITOR
    assert cg_pair is True


# 43. Распознавание лица на подушке с поворотом кропа головы ±90° и сниженным порогом YuNet 0.48
def test_face_registry_threshold_048_and_rotated_pillow_head_crop_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = FaceRegistry(faces_dir=tmp_path / "faces", models_dir=tmp_path / "models", threshold=0.38)
    monkeypatch.setattr(reg, "ensure_models", lambda: True)
    reg._models_ready = True

    expected_emb = np.ones(128, dtype=np.float32)
    expected_emb /= float(np.linalg.norm(expected_emb))
    seen_orientations: list[str] = []

    # Синтетический кадр 360x640: вокруг головы на подушке ставим асимметричный маркер, который после
    # поворота на +90° (ROTATE_90_CLOCKWISE) оказывается в верхней полосе кропа — как лицо на боку!
    frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    obs_pillow = make_pose(
        track_id=1,
        bbox=(0.34, 0.50, 0.60, 0.70),
        shoulder=(0.52, 0.58),
        hip=(0.40, 0.60),
        nose=(0.55, 0.56),
    )
    hc = obs_pillow.head_center
    assert hc is not None
    hx, hy = int(hc[0] * 640), int(hc[1] * 360)
    # Маркер СЛЕВА от центра головы -> на 0° он строго слева, а после ROTATE_90_CLOCKWISE переходит ВВЕРХ!
    frame[hy - 3 : hy + 3, hx - 42 : hx - 32] = (250, 220, 180)

    def _fake_detect_best(img_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        top_band_bright = float(img_bgr[: max(8, img_bgr.shape[0] // 4), :, 0].max()) >= 240.0
        left_band_bright = float(img_bgr[:, : max(8, img_bgr.shape[1] // 4), 0].max()) >= 240.0
        if left_band_bright and not top_band_bright:
            seen_orientations.append("0_deg_missed")
            return None, None
        if top_band_bright:
            seen_orientations.append("rotated_90_hit")
            h_i, w_i = img_bgr.shape[:2]
            return expected_emb.copy(), (w_i // 4, h_i // 4, w_i // 2, h_i // 2)
        return None, None

    monkeypatch.setattr(reg, "_detect_best_face_and_embedding", _fake_detect_best)

    ok, saved_name = reg.enroll_from_frame(Role.PATIENT, frame, obs=obs_pillow)
    assert ok is True
    assert saved_name.endswith(".jpg")
    assert "0_deg_missed" in seen_orientations
    assert "rotated_90_hit" in seen_orientations

    matched_role, score = reg.match_person_crop(frame, obs_pillow)
    assert matched_role == Role.PATIENT
    assert score >= 0.99


# 44. Стационарный ракурс 14:25 (permanent_mount_1425.jpg): родственник спиной к камере (kpts 0..4 = 0)
# на переднем плане детектируется как VISITOR (Свой), сидящая на кровати бабушка — PATIENT, а куртка на стене = 0!
def test_back_turned_foreground_visitor_detected_while_sitting_grandma_stays_patient_and_coat_rejected() -> None:
    zones_1425 = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )
    # 1. Девушка сидит строго спиной к камере на переднем плане (kpts 0..4 все 0.0, видны плечи и локти, y2=0.994)
    back_kps = [(0.0, 0.0, 0.0)] * 17
    back_kps[5] = (0.250, 0.835, 0.98)
    back_kps[6] = (0.335, 0.836, 0.98)
    back_kps[7] = (0.243, 0.950, 0.62)
    back_kps[8] = (0.375, 0.953, 0.72)
    back_visitor = PoseObservation(
        track_id=10002,
        bbox_norm=(0.233, 0.660, 0.401, 0.994),
        conf=0.89,
        keypoints_norm=tuple(back_kps),
    )
    # 2. Бабушка сидит на краю кровати лицом к нам со свешенными ногами
    grandma_sitting = make_pose(
        track_id=1,
        bbox=(0.3485, 0.4411, 0.4903, 0.8991),
        shoulder=(0.404, 0.534),
        hip=(0.420, 0.647),
        knee=(0.393, 0.733),
        ankle=(0.417, 0.842),
        conf=0.90,
    )
    # 3. Неподвижная куртка без головы на стене (y2=0.58)
    wall_coat = make_pose(
        track_id=99,
        bbox=(0.242, 0.183, 0.336, 0.580),
        shoulder=(0.280, 0.245),
        hip=(0.284, 0.392),
        conf=0.58,
        headless=True,
    )

    valid = filter_valid_poses([grandma_sitting, back_visitor, wall_coat], min_box_conf=0.48, zones=zones_1425)
    assert {o.track_id for o in valid} == {1, 10002}

    ident = PatientIdentifier(bed_lock_seconds=12.0, initial_bed_lock_seconds=1.5)
    fsm = FallStateMachine(sit_confirm_seconds=1.2, suppress_sitting_when_caregiver=True)

    # На 1-м одиночном кадре недверный второй трек ещё не прошёл 4-кадровый гейт движения (Задача 7),
    # а после подтверждённого подхода и посадки за 4 кадра (disp >= 0.045) распознаётся как VISITOR!
    def _make_back_step(dx: float) -> PoseObservation:
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[5] = (0.250 + dx, 0.835, 0.98)
        kps[6] = (0.335 + dx, 0.836, 0.98)
        kps[7] = (0.243 + dx, 0.950, 0.62)
        kps[8] = (0.375 + dx, 0.953, 0.72)
        return PoseObservation(
            track_id=10002,
            bbox_norm=(0.233 + dx, 0.660, 0.401 + dx, 0.994),
            conf=0.89,
            keypoints_norm=tuple(kps),
        )

    roles_f1, cg_f1 = ident.update([grandma_sitting, _make_back_step(-0.06), wall_coat], zones_1425, now=0.25)
    assert set(roles_f1.keys()) == {1}
    assert cg_f1 is False

    ident.update([grandma_sitting, _make_back_step(-0.04), wall_coat], zones_1425, now=0.50)
    ident.update([grandma_sitting, _make_back_step(-0.02), wall_coat], zones_1425, now=0.75)
    roles, cg = ident.update([grandma_sitting, back_visitor, wall_coat], zones_1425, now=1.0)
    assert set(roles.keys()) == {1, 10002}
    assert roles[1].role == Role.PATIENT
    assert roles[10002].role == Role.VISITOR
    assert cg is True
    assert fsm.step([grandma_sitting, back_visitor], roles, cg, zones_1425, now=1.0) is None


# 45. Волна 4 (живой кейс deep_blanket_1502.jpg): укрывание пёстрым одеялом с головой (box_conf=0.077..0.089
# при чёткой голове + обоих плечах в WIN 0) и защита от ложного IN_BED -> MISSING за 12 с без фазы посадки!
def test_deep_blanket_sub_010_conf_sleeper_and_deep_bed_missing_suppression() -> None:
    zones_1502 = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )
    # Точный слепок WIN[0][1] с живого кадра deep_blanket_1502.jpg (conf=0.077, нос+глаз+ухо + оба плеча на подушке)
    deep_kps = [(0.0, 0.0, 0.0)] * 17
    deep_kps[0] = (0.551, 0.556, 0.429)
    deep_kps[1] = (0.554, 0.550, 0.545)
    deep_kps[3] = (0.564, 0.548, 0.450)
    deep_kps[5] = (0.589, 0.579, 0.501)
    deep_kps[6] = (0.545, 0.565, 0.340)
    deep_sleeper = PoseObservation(
        track_id=9001,
        bbox_norm=(0.5324, 0.5108, 0.6221, 0.6639),
        conf=0.077,
        keypoints_norm=tuple(deep_kps),
    )
    # Шумная пустая складка одеяла с conf=0.091 без выраженной головы и обоих плеч — обязана отсекаться!
    fold_kps = [(0.0, 0.0, 0.0)] * 17
    fold_kps[0] = (0.552, 0.577, 0.329)
    fold_kps[6] = (0.540, 0.607, 0.462)
    noise_fold = PoseObservation(
        track_id=9002,
        bbox_norm=(0.6021, 0.6418, 0.7015, 0.8057),
        conf=0.091,
        keypoints_norm=tuple(fold_kps),
    )

    assert is_blanket_sleeper_in_bed(deep_sleeper, zones_1502.bed_polygon) is True
    assert is_blanket_sleeper_in_bed(noise_fold, zones_1502.bed_polygon) is False
    valid = filter_valid_poses([deep_sleeper, noise_fold], min_box_conf=0.48, zones=zones_1502)
    assert [o.track_id for o in valid] == [9001]

    ident = PatientIdentifier(bed_lock_seconds=12.0, initial_bed_lock_seconds=1.5)
    fsm = FallStateMachine(sit_confirm_seconds=1.5, missing_confirm_seconds=12.0, night_strict_mode=False)

    t = 0.0
    for _ in range(6):
        t += 0.4
        roles, cg = ident.update([deep_sleeper], zones_1502, now=t)
        assert fsm.step([deep_sleeper], roles, cg, zones_1502, now=t) is None
    assert roles[9001].role == Role.PATIENT
    assert fsm.state == PatientState.IN_BED

    # Даже если бабушка полностью накрыла макушку одеялом на 30 секунд без фазы посадки/края кровати —
    # ложный MISSING за 12 с НЕ срабатывает, состояние удерживает IN_BED!
    for _ in range(60):
        t += 0.5
        roles, cg = ident.update([], zones_1502, now=t)
        ev = fsm.step([], roles, cg, zones_1502, now=t)
        assert ev is None
        assert fsm.state == PatientState.IN_BED

    # А если перед исчезновением пациентка реально села со свешенными ногами (SITTING_UP) и ушла — MISSING срабатывает за 12 с!
    for _ in range(5):
        t += 0.4
        sit_obs = make_pose(
            track_id=9001,
            bbox=(0.35, 0.44, 0.49, 0.90),
            shoulder=(0.41, 0.53),
            hip=(0.42, 0.65),
            knee=(0.40, 0.73),
            ankle=(0.41, 0.85),
            conf=0.88,
        )
        roles, cg = ident.update([sit_obs], zones_1502, now=t)
        fsm.step([sit_obs], roles, cg, zones_1502, now=t)
    assert fsm.state == PatientState.SITTING_UP

    missing_ev = None
    for _ in range(30):
        t += 0.5
        roles, cg = ident.update([], zones_1502, now=t)
        ev = fsm.step([], roles, cg, zones_1502, now=t)
        if ev is not None:
            missing_ev = ev
            break
    assert missing_ev is not None and missing_ev.state == PatientState.MISSING


# 46. Волна 5 (Правило двери 1 + 3): вход через дверь -> тихий визит CAREGIVER_ASSIST -> выход обратно в дверь -> возврат охраны
def test_door_rule_1_plus_3_full_visit_lifecycle_and_exit_reactivation(tmp_path: Path) -> None:
    anchor_path = tmp_path / "patient_anchor.json"
    ident = PatientIdentifier(
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=1.0,
        state_path=anchor_path,
        door_visit_idle_timeout_sec=90.0,
    )
    fsm = FallStateMachine(
        sit_confirm_seconds=1.0,
        missing_confirm_seconds=3.0,
        suppress_sitting_when_caregiver=True,
        night_strict_mode=False,
    )
    grandma_col = (200, 185, 65)
    visitor_col = (40, 45, 190)

    # 1. Бабушка одна спит в кровати -> Role.PATIENT, IN_BED, door_visit_active=False
    t = 0.0
    for _ in range(6):
        t += 0.25
        g_sleep = make_pose(1, (0.30, 0.42, 0.66, 0.58), (0.38, 0.49), (0.56, 0.50))
        fr = make_colored_room_frame([(g_sleep, grandma_col)])
        roles, cg = ident.update([g_sleep], BED_ZONE, now=t, frame_bgr=fr)
        assert fsm.step([g_sleep], roles, cg, BED_ZONE, now=t) is None
    assert roles[1].role == Role.PATIENT
    assert ident.door_visit_active is False
    assert fsm.state == PatientState.IN_BED

    # 2. Родственник входит через полигон двери -> Role.VISITOR (reason="door_entry") + door_visit_active=True
    t += 0.3
    v_enter_0 = make_pose(2, (0.88, 0.18, 0.98, 0.82), (0.93, 0.28), (0.93, 0.54))
    ident.update(
        [g_sleep, v_enter_0],
        BED_ZONE,
        now=t,
        frame_bgr=make_colored_room_frame([(g_sleep, grandma_col), (v_enter_0, visitor_col)]),
    )
    t += 0.3
    v_enter_1 = make_pose(2, (0.76, 0.20, 0.88, 0.82), (0.82, 0.30), (0.82, 0.55))
    roles_v, cg_v = ident.update(
        [g_sleep, v_enter_1],
        BED_ZONE,
        now=t,
        frame_bgr=make_colored_room_frame([(g_sleep, grandma_col), (v_enter_1, visitor_col)]),
    )
    assert roles_v[2].role == Role.VISITOR
    assert roles_v[2].reason == "door_entry"
    assert ident.door_visit_active is True
    assert cg_v is True

    # 2б. Гость подходит к кровати и у него перескакивает track_id (2 -> 22) на краю кровати — остаётся VISITOR!
    t += 0.4
    v_bed_edge_2 = make_pose(2, (0.64, 0.28, 0.78, 0.86), (0.71, 0.38), (0.71, 0.60), ankle=(0.71, 0.84))
    ident.update(
        [g_sleep, v_bed_edge_2],
        BED_ZONE,
        now=t,
        frame_bgr=make_colored_room_frame([(g_sleep, grandma_col), (v_bed_edge_2, visitor_col)]),
    )
    t += 0.4
    v_bed_edge_22 = make_pose(22, (0.65, 0.28, 0.79, 0.86), (0.72, 0.38), (0.72, 0.60), ankle=(0.72, 0.84))
    roles_sw, _ = ident.update(
        [g_sleep, v_bed_edge_22],
        BED_ZONE,
        now=t,
        frame_bgr=make_colored_room_frame([(g_sleep, grandma_col), (v_bed_edge_22, visitor_col)]),
    )
    assert roles_sw[22].role == Role.VISITOR

    # 3. Пока визит активен, даже если гость отошёл в мёртвую зону (нет в кадре 6 с < 90 с!),
    # посадка бабушки и кратковременная окклюзия НЕ дают ложных тревог — держится CAREGIVER_ASSIST!
    for _ in range(12):
        t += 0.5
        g_sit = make_pose(1, (0.38, 0.30, 0.56, 0.76), (0.47, 0.36), (0.47, 0.56), ankle=(0.48, 0.77))
        roles_mid, cg_mid = ident.update([g_sit], BED_ZONE, now=t)
        ev_mid = fsm.step([g_sit], roles_mid, cg_mid, BED_ZONE, now=t)
        assert ident.door_visit_active is True
        assert cg_mid is True
        assert ev_mid is None
        assert fsm.state == PatientState.CAREGIVER_ASSIST

    # 4. Родственник (>= 3.0 с в комнате) снова проходит через door_polygon на выход и исчезает на >= 2.5 с
    t += 0.4
    v_exit_0 = make_pose(2, (0.78, 0.20, 0.90, 0.82), (0.84, 0.30), (0.84, 0.55))
    ident.update([g_sleep, v_exit_0], BED_ZONE, now=t)
    t += 0.4
    v_exit_door = make_pose(2, (0.88, 0.18, 0.98, 0.82), (0.93, 0.28), (0.93, 0.54))
    ident.update([g_sleep, v_exit_door], BED_ZONE, now=t)
    assert ident.door_visit_active is True  # Ещё в двери — визит пока активен

    t += 2.6
    roles_after_exit, cg_after_exit = ident.update([g_sleep], BED_ZONE, now=t)
    assert ident.door_visit_active is False
    assert cg_after_exit is False
    assert fsm.step([g_sleep], roles_after_exit, cg_after_exit, BED_ZONE, now=t) is None
    assert fsm.state == PatientState.IN_BED

    # 5. Теперь гость ушёл — самостоятельная посадка бабушки МГНОВЕННО вызывает тревогу SITTING_UP!
    sit_alert = None
    for _ in range(6):
        t += 0.25
        g_sit_alone = make_pose(1, (0.38, 0.30, 0.56, 0.76), (0.47, 0.36), (0.47, 0.56), ankle=(0.48, 0.77))
        roles_al, cg_al = ident.update([g_sit_alone], BED_ZONE, now=t)
        ev_al = fsm.step([g_sit_alone], roles_al, cg_al, BED_ZONE, now=t)
        if ev_al is not None:
            sit_alert = ev_al
            break
    assert sit_alert is not None
    assert sit_alert.state == PatientState.SITTING_UP


# 47. Волна 5: страховочный watchdog 90 с дверного визита, clear_door_visit() и Аксиома 1 (кровать без двери = Бабушка)
def test_door_visit_90s_watchdog_clear_door_visit_and_bed_without_door_is_always_patient() -> None:
    ident = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        door_visit_idle_timeout_sec=90.0,
    )
    t = 0.0
    g_sleep = make_pose(1, (0.30, 0.42, 0.66, 0.58), (0.38, 0.49), (0.56, 0.50))
    for _ in range(5):
        t += 0.25
        ident.update([g_sleep], BED_ZONE, now=t)

    # Гость вошёл через дверь, но ночью вышел мимо полигона двери в слепой зоне
    t += 0.3
    ident.update([g_sleep, make_pose(5, (0.88, 0.18, 0.98, 0.82), (0.93, 0.28), (0.93, 0.54))], BED_ZONE, now=t)
    t += 0.3
    ident.update([g_sleep, make_pose(5, (0.76, 0.20, 0.88, 0.82), (0.82, 0.30), (0.82, 0.54))], BED_ZONE, now=t)
    assert ident.door_visit_active is True

    # Через 60 с без гостей визит ещё удерживается, а через 90.5 с watchdog сбрасывает залипание!
    t += 60.0
    _, cg_60 = ident.update([g_sleep], BED_ZONE, now=t)
    assert ident.door_visit_active is True
    assert cg_60 is True

    t += 30.5
    _, cg_90 = ident.update([g_sleep], BED_ZONE, now=t)
    assert ident.door_visit_active is False
    assert cg_90 is False

    # Проверка ручного сброса clear_door_visit()
    t += 1.0
    ident.update([g_sleep, make_pose(6, (0.88, 0.18, 0.98, 0.82), (0.93, 0.28), (0.93, 0.54))], BED_ZONE, now=t)
    t += 0.3
    ident.update([g_sleep, make_pose(6, (0.76, 0.20, 0.88, 0.82), (0.82, 0.30), (0.82, 0.54))], BED_ZONE, now=t)
    assert ident.door_visit_active is True
    ident.clear_door_visit()
    assert ident.door_visit_active is False


# 48. Волна 5: санация смятых на подушке ключевых точек (устранение ложного angle ~ 2.1° при крошечном боксе головы)
def test_sanitize_collapsed_bed_keypoints_eliminates_false_vertical_angle_and_preserves_in_bed() -> None:
    # Сырой баг YOLO на подушке: в маленьком боксе головы w=0.09, h=0.09 свалены все 17 точек,
    # из-за чего вектор hip->shoulder стоял почти вертикально (angle ~ 2.1 град)!
    crumpled_kps = [(0.0, 0.0, 0.0)] * 17
    crumpled_kps[0] = (0.552, 0.542, 0.88)   # нос
    crumpled_kps[1] = (0.548, 0.538, 0.86)   # левый глаз
    crumpled_kps[2] = (0.556, 0.538, 0.84)   # правый глаз
    crumpled_kps[5] = (0.524, 0.555, 0.78)   # левое плечо (физиологическая ширина ±0.028)
    crumpled_kps[6] = (0.580, 0.556, 0.76)   # правое плечо
    # Фантомно сваленные на подушку бёдра, колени и лодыжки прямо под подбородком (dy = 0.04 -> angle ~ 2°)
    crumpled_kps[11] = (0.545, 0.595, 0.52)
    crumpled_kps[12] = (0.559, 0.596, 0.50)
    crumpled_kps[13] = (0.546, 0.602, 0.44)
    crumpled_kps[15] = (0.548, 0.608, 0.41)

    raw_crumpled = PoseObservation(
        track_id=1,
        bbox_norm=(0.510, 0.515, 0.600, 0.605),
        conf=0.38,
        keypoints_norm=tuple(crumpled_kps),
    )
    raw_angle = raw_crumpled.torso_angle_from_vertical_deg()
    assert raw_angle is not None and raw_angle < 5.0

    cleaned = sanitize_collapsed_bed_keypoints(raw_crumpled, BED_ZONE.bed_polygon)
    assert cleaned.mid_shoulder is not None
    assert cleaned.head_center is not None
    assert cleaned.mid_hip is None
    assert cleaned.lower_limbs == []
    assert cleaned.torso_angle_from_vertical_deg() is None
    assert is_blanket_sleeper_in_bed(cleaned, BED_ZONE.bed_polygon) is True

    ident = PatientIdentifier(initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(sit_confirm_seconds=1.0)
    t = 0.0
    for _ in range(10):
        t += 0.25
        roles, cg = ident.update([raw_crumpled], BED_ZONE, now=t)
        ev = fsm.step([raw_crumpled], roles, cg, BED_ZONE, now=t)
        assert ev is None
    assert roles[1].role == Role.PATIENT
    assert fsm.state == PatientState.IN_BED


# 49. Волна 5: монолитная стабилизация _bed_envelope_bbox спящей бабушки и чистый HUD без микро-скелета на лице
def test_bed_envelope_bbox_stabilizer_and_clean_hud_without_face_micro_skeleton(tmp_path: Path) -> None:
    tracker = PoseTracker("yolo11s-pose.pt", tmp_path, smooth_keypoints=False)
    zones = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )

    # 1. Эталонная полноразмерная рамка сна -> затем резкое сжатие сырого бокса до маленькой головы на подушке
    full_sleep = make_pose(
        track_id=1,
        bbox=(0.34, 0.52, 0.59, 0.69),
        shoulder=(0.54, 0.57),
        hip=(0.42, 0.60),
        conf=0.55,
    )
    st1 = tracker._stabilize_bed_sleeper_bbox(full_sleep, zones.bed_polygon, now=1.0)
    assert st1.bbox_width >= 0.24

    head_only_kps = [(0.0, 0.0, 0.0)] * 17
    head_only_kps[0] = (0.555, 0.555, 0.85)
    head_only_kps[1] = (0.560, 0.550, 0.82)
    head_only_kps[5] = (0.527, 0.570, 0.76)
    head_only_kps[6] = (0.583, 0.575, 0.74)
    tiny_head_obs = PoseObservation(
        track_id=1,
        bbox_norm=(0.520, 0.525, 0.595, 0.600),  # сжался до 0.075 x 0.075!
        conf=0.28,
        keypoints_norm=tuple(head_only_kps),
    )
    st2 = tracker._stabilize_bed_sleeper_bbox(tiny_head_obs, zones.bed_polygon, now=1.2)
    assert st2.bbox_width >= 0.23
    assert st2.bbox_height >= 0.15
    assert abs(st2.bbox_norm[0] - st1.bbox_norm[0]) < 0.02

    # 2. Холодный старт сразу с глубоко укрытой головы (без предыдущего полного бокса) тоже строит спальную рамку
    tracker.reset_smoother()
    st_cold = tracker._stabilize_bed_sleeper_bbox(tiny_head_obs, zones.bed_polygon, now=5.0)
    assert st_cold.bbox_width >= 0.23
    assert st_cold.bbox_height >= 0.15

    # 3. Проверка HUD render_annotated_frame: внутри области лица (0.535..0.570 x 0.540..0.585) у спящей под одеялом
    # бабушки НЕ рисуются зелёные кружки/линии микро-скелета (чистое лицо без визуального мусора!)
    base_frame = np.full((360, 640, 3), 50, dtype=np.uint8)
    ident = PatientIdentifier(initial_bed_lock_seconds=0.2)
    roles, _ = ident.update([st2], zones, now=1.0)
    roles[1].role = Role.PATIENT

    rendered = render_annotated_frame(
        frame_bgr=base_frame,
        observations=[st2],
        identities=roles,
        zones=ZonesConfig(bed_polygon=[], door_polygon=[]),  # без заливки полигона, чтобы проверить чистоту пикселей лица
        state=PatientState.IN_BED,
    )
    fx1, fy1 = int(0.535 * 640), int(0.545 * 360)
    fx2, fy2 = int(0.568 * 640), int(0.582 * 360)
    face_patch = rendered[fy1:fy2, fx1:fx2]
    # Внутри области лица ни один пиксель не закрашен зелёным скелетом (все пиксели остались ровным фоном 50)!
    assert int(np.max(np.abs(face_patch.astype(np.int16) - 50))) == 0


# 50. Волна 5: нативная загрузка TensorRT FP16 (.engine), передача rect=False и безопасный fallback на .pt
def test_pose_tracker_tensorrt_fp16_engine_loading_rect_false_and_pt_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    import ultralytics

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "yolo11m-pose.pt").write_bytes(b"PT_WEIGHTS")
    (models_dir / "yolo11m-pose.engine").write_bytes(b"TRT_ENGINE")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA GeForce RTX 3060")

    recorded_calls: list[tuple[str, dict[str, object]]] = []
    fail_engine = [False]

    class _FakeLoadedYOLO:
        def __init__(self, path_str: str, **init_kw: object) -> None:
            self.path_str = path_str
            self.init_kw = init_kw
            recorded_calls.append(("init", {"path": path_str, **init_kw}))

        def predict(self, source: np.ndarray, **kw: object) -> list:
            recorded_calls.append(("predict", {"path": self.path_str, **kw}))
            if fail_engine[0] and self.path_str.endswith(".engine"):
                raise RuntimeError("TensorRT version mismatch")
            return []

        def track(self, source: np.ndarray, **kw: object) -> list:
            recorded_calls.append(("track", {"path": self.path_str, **kw}))
            return []

    monkeypatch.setattr(ultralytics, "YOLO", _FakeLoadedYOLO)

    # 1. Штатная активация готового TensorRT .engine рядом с .pt + проверка rect=False и device=0
    trk_trt = PoseTracker("yolo11m-pose.pt", models_dir, device="cuda")
    trk_trt.load()
    assert trk_trt.runtime_label == "TensorRT FP16 (NVIDIA GeForce RTX 3060)"
    assert trk_trt._loaded_is_engine is True

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    trk_trt.infer(frame, zones=BED_ZONE)
    track_calls = [kw for kind, kw in recorded_calls if kind == "track"]
    assert len(track_calls) == 1
    assert str(track_calls[0]["path"]).endswith("yolo11m-pose.engine")
    assert track_calls[0]["device"] == 0
    assert track_calls[0]["rect"] is False

    # 2. Если .engine битый или от другой версии TensorRT -> автоматический откат на PyTorch .pt без падения!
    recorded_calls.clear()
    fail_engine[0] = True
    trk_fb = PoseTracker("yolo11m-pose.engine", models_dir, device="cuda")
    trk_fb.load()
    assert trk_fb.runtime_label == "CUDA (NVIDIA GeForce RTX 3060)"
    assert trk_fb._loaded_is_engine is False
    assert str(trk_fb._loaded_model_path).endswith("yolo11m-pose.pt")


# 51. Реальный горизонтальный сон лицом от камеры (hc=None, как в diag_1400) сохраняет полный скелет и валидность
def test_side_sleeper_turned_away_without_head_keeps_horizontal_hips_and_passes_pose_filter() -> None:
    from app.geometry import is_valid_human_pose

    real_bed_zone = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[],
    )
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[5] = (0.534, 0.626, 0.60)
    pts[6] = (0.538, 0.588, 0.63)
    pts[7] = (0.494, 0.674, 0.53)
    pts[8] = (0.496, 0.605, 0.49)
    pts[9] = (0.451, 0.645, 0.62)
    pts[10] = (0.460, 0.590, 0.55)
    pts[11] = (0.481, 0.630, 0.60)
    pts[12] = (0.483, 0.618, 0.60)
    pts[13] = (0.453, 0.653, 0.51)
    pts[14] = (0.456, 0.611, 0.44)
    pts[15] = (0.436, 0.650, 0.36)
    pts[16] = (0.438, 0.605, 0.32)

    raw_obs = PoseObservation(
        track_id=9001,
        bbox_norm=(0.410, 0.545, 0.572, 0.720),
        conf=0.275,
        keypoints_norm=tuple(pts),
    )
    san_obs = sanitize_collapsed_bed_keypoints(raw_obs, real_bed_zone.bed_polygon)
    assert san_obs.mid_hip is not None
    assert san_obs.valid_keypoint_count(0.30) == 12
    assert is_valid_human_pose(san_obs, min_kp_conf=0.30, min_valid_kps=4, min_box_conf=0.48, zones=real_bed_zone) is True


# 52. Боевая регрессия 17:18 на 192.168.1.100: родственница + стоящий на полу перед бортом кровати ребёнок (track_id=667)
def test_live_incident_1718_visitor_with_standing_child_in_front_of_bed_zero_false_alarms(
    tmp_path: Path,
) -> None:
    from app.geometry import are_legs_hanging_outside_bed, is_standing_in_front_of_bed

    real_zones = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
        ignore_polygons=[
            [[0.71599, 0.0311], [0.69716, 0.60407], [0.81019, 0.85287], [0.99994, 0.91388], [0.99725, 0.00598], [0.71599, 0.0012]],
            [[0.70052, 0.41627], [0.30959, 0.4067], [0.24567, 0.35885], [0.22818, 0.16627], [0.06871, 0.26555], [0.1831, 0.9988], [0.00076, 0.9988], [0.00076, 0.00239], [0.71398, 0.0]],
        ],
    )
    anchor_file = tmp_path / "patient_anchor.json"
    ident = PatientIdentifier(
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=1.5,
        state_path=anchor_file,
    )
    fsm = FallStateMachine(
        sit_confirm_seconds=1.5,
        fall_confirm_seconds=1.2,
        suppress_sitting_when_caregiver=True,
    )

    grandma_color = (185, 165, 95)
    woman_color = (45, 55, 165)
    child_color = (65, 175, 70)

    # 1. Бабушка сначала спокойно спала в кровати, сформировав эталонный якорь одежды и baseline сна
    t = 0.0
    for _ in range(8):
        t += 0.3
        g_sleep = make_pose(1, (0.34, 0.52, 0.59, 0.69), (0.54, 0.57), (0.42, 0.60))
        fr_g = make_colored_room_frame([(g_sleep, grandma_color)])
        roles_g, cg_g = ident.update([g_sleep], real_zones, now=t, frame_bgr=fr_g)
        assert fsm.step([g_sleep], roles_g, cg_g, real_zones, now=t) is None
    saved_grandma_anchor = ident._anchor_vec.copy()
    assert saved_grandma_anchor is not None

    # 2. Вход через дверной проём на границе с ignore_polygons[1]: трек НЕ вырезается игнор-зоной,
    # проходит первичный StaticObjectGate детектора за малый шаг в двери (0.03) и НЕ ждёт второго
    # смещения в PatientIdentifier -> рождается прямо в door_polygon и взводит door_visit_active=True!
    det_gate = StaticObjectGate()
    t += 0.15
    door_guest_t0 = make_pose(672, (0.09, 0.26, 0.22, 0.84), (0.155, 0.36), (0.155, 0.58), ankle=(0.155, 0.82))
    assert len(filter_valid_poses([door_guest_t0], zones=real_zones)) == 1
    assert det_gate.update_and_filter([door_guest_t0], zones=real_zones, now=t) == []

    t += 0.15
    door_guest = make_pose(672, (0.12, 0.26, 0.25, 0.84), (0.185, 0.36), (0.185, 0.58), ankle=(0.185, 0.82))
    det_live = det_gate.update_and_filter([door_guest], zones=real_zones, now=t)
    assert len(det_live) == 1
    assert det_live[0].liveness_verified is True
    assert det_live[0].spawn_center_hint == door_guest_t0.body_center

    fr_door = make_colored_room_frame([(det_live[0], woman_color)])
    roles_d, cg_d = ident.update(det_live, real_zones, now=t, frame_bgr=fr_door)
    assert 672 in roles_d
    assert roles_d[672].role == Role.VISITOR
    assert roles_d[672].reason == "door_entry"
    assert ident.door_visit_active is True
    assert cg_d is True

    # 3. Точные боевые координаты 17:18:22: женщина (672) и маленький ребёнок (667), стоящий вертикально
    # на полу вплотную ПЕРЕД бортом кровати (bbox=[0.4971, 0.5677, 0.5547, 0.8700], angle≈18.6°, ar≈0.19)
    woman_near_bed = make_pose(672, (0.4067, 0.4127, 0.5229, 0.8687), (0.4648, 0.50), (0.4658, 0.69), ankle=(0.465, 0.85))
    for step_idx, cx_x1 in enumerate((0.4371, 0.4571, 0.4771), start=1):
        child_step = make_pose(
            track_id=667,
            bbox=(cx_x1, 0.5677, cx_x1 + 0.0576, 0.8700),
            shoulder=(cx_x1 + 0.020, 0.625),
            hip=(cx_x1 + 0.039, 0.726),
            ankle=(cx_x1 + 0.033, 0.858),
        )
        ident.update(
            [woman_near_bed, child_step],
            real_zones,
            now=t + 0.08 * step_idx,
            frame_bgr=make_colored_room_frame([(woman_near_bed, woman_color), (child_step, child_color)]),
        )
    t += 0.24
    child_standing = make_pose(
        track_id=667,
        bbox=(0.4971, 0.5677, 0.5547, 0.8700),
        shoulder=(0.517, 0.625),
        hip=(0.536, 0.726),  # hip_y=0.726 внутри нижнего края bed_polygon (max_y=0.79), angle≈18.6° с 16:9!
        ankle=(0.530, 0.858),
    )
    assert 17.5 <= (child_standing.torso_angle_from_vertical_deg() or 0.0) <= 19.5
    assert is_standing_in_front_of_bed(child_standing, real_zones.bed_polygon) is True
    assert is_person_on_bed(child_standing, real_zones.bed_polygon) is False
    assert are_legs_hanging_outside_bed(child_standing, real_zones.bed_polygon) is False

    for _ in range(10):
        t += 0.3
        fr_pair = make_colored_room_frame([(woman_near_bed, woman_color), (child_standing, child_color)])
        roles_p, cg_p = ident.update([woman_near_bed, child_standing], real_zones, now=t, frame_bgr=fr_pair)
        ev_p = fsm.step([woman_near_bed, child_standing], roles_p, cg_p, real_zones, now=t)
        assert ev_p is None, f"Ложная тревога на стоящем перед кроватью ребёнке: {ev_p}"
        assert roles_p[672].role == Role.VISITOR
        assert roles_p[667].role != Role.PATIENT
        assert cg_p is True
        assert ident.door_visit_active is True

    # 4. Пользователь нажимает «Свой» на женщине (672), а затем на ребёнке (667):
    # ребёнок НИКОГДА не повышается в PATIENT, а якорь бабушки _anchor_vec остаётся нетронутым!
    assert ident.force_assign_visitor(672, now=t) is True
    assert ident._tracks[667].role != Role.PATIENT
    assert cosine_sim(ident._anchor_vec, saved_grandma_anchor) >= 0.999

    assert ident.force_assign_visitor(667, now=t + 0.1) is True
    assert ident._tracks[667].role == Role.VISITOR
    assert ident._tracks[667].reason == "manual_visitor_override"
    assert ident.door_visit_active is True
    assert cosine_sim(ident._anchor_vec, saved_grandma_anchor) >= 0.999

    # 5. Эшелонированная защита FallStateMachine: даже если бы стоящий ребёнок имел роль PATIENT,
    # вертикально стоящая поза (angle≈18.6°, ar≈0.19) НЕ должна давать ложный FALLEN!
    fake_pat_roles = {667: ident._tracks[667]}
    old_role = ident._tracks[667].role
    ident._tracks[667].role = Role.PATIENT
    for _ in range(10):
        t += 0.3
        assert fsm.step([child_standing], fake_pat_roles, True, real_zones, now=t) is None
    assert fsm.state != PatientState.FALLEN
    ident._tracks[667].role = old_role


# 53. Боевая регрессия 17:48–18:06 (ночной Ч/Б ИК на yolo11m-pose.engine):
# дублирующие боксы одной спящей под одеялом бабушки (Снимок А IoU=0.72 и Снимок Б — складка одеяла/ног)
# подавляются в PoseTracker и PatientIdentifier, НЕ зажигая ложный caregiver / door_visit_active!
def test_live_night_ir_bed_sleeper_duplicate_boxes_never_trigger_false_caregiver_or_door_visit(
    tmp_path: Path,
) -> None:
    import time as _time
    class _FakeTensor:
        def __init__(self, arr: list | np.ndarray) -> None:
            self._arr = np.asarray(arr, dtype=np.float32)

        def int(self) -> "_FakeTensor":
            return _FakeTensor(self._arr.astype(np.int32))

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeBoxes:
        def __init__(self, xyxy: list, conf: list, ids: list | None = None) -> None:
            self.xyxy = _FakeTensor(xyxy)
            self.conf = _FakeTensor(conf)
            self.id = _FakeTensor(ids) if ids is not None else None

        def __len__(self) -> int:
            return int(len(self.conf.numpy()))

    class _FakeKeypoints:
        def __init__(self, xy: list, conf: list) -> None:
            self.xy = _FakeTensor(xy)
            self.conf = _FakeTensor(conf)

    class _FakeResult:
        def __init__(self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None) -> None:
            self.boxes = boxes
            self.keypoints = keypoints

    def _horiz_kps(w: float, h: float, sh: tuple[float, float], hp: tuple[float, float]) -> tuple[list, list]:
        xy = [[0.0, 0.0]] * 17
        cf = [0.0] * 17
        nx = min(0.98, sh[0] + 0.025)
        ny = max(0.02, sh[1] - 0.012)
        xy[0], cf[0] = [nx * w, ny * h], 0.86
        xy[1], cf[1] = [(nx + 0.005) * w, (ny - 0.004) * h], 0.84
        xy[5], cf[5] = [sh[0] * w, (sh[1] - 0.008) * h], 0.88
        xy[6], cf[6] = [sh[0] * w, (sh[1] + 0.008) * h], 0.87
        xy[11], cf[11] = [hp[0] * w, (hp[1] - 0.008) * h], 0.82
        xy[12], cf[12] = [hp[0] * w, (hp[1] + 0.008) * h], 0.81
        return xy, cf

    class _FakeNightIrModel:
        def __init__(self) -> None:
            self.mode = "snap_a"

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            h, w = source.shape[:2]
            if self.mode == "snap_a":
                # Снимок А: два полнокадровых лежачих бокса на одной спящей бабушке (IoU = 0.72)
                bx_main = [0.3199 * w, 0.5184 * h, 0.5943 * w, 0.6632 * h]
                bx_dup = [0.3322 * w, 0.4954 * h, 0.5938 * w, 0.6334 * h]
                kp1, cf1 = _horiz_kps(w, h, (0.540, 0.578), (0.415, 0.598))
                kp2, cf2 = _horiz_kps(w, h, (0.536, 0.562), (0.420, 0.580))
                return [
                    _FakeResult(
                        _FakeBoxes([bx_main, bx_dup], [0.61, 0.54], [964, 965]),
                        _FakeKeypoints([kp1, kp2], [cf1, cf2]),
                    )
                ]
            # Снимок Б: верхняя половина тела + горизонтальный фрагмент одеяла/ног (angle_deg ~ 86.3°)
            bx_main = [0.3178 * w, 0.4928 * h, 0.5975 * w, 0.6338 * h]
            bx_fold = [0.2748 * w, 0.5731 * h, 0.6202 * w, 0.7437 * h]
            kp1, cf1 = _horiz_kps(w, h, (0.542, 0.558), (0.418, 0.575))
            # Намеренно ставим таз складки одеяла на скошенную границу правой выемки bed_polygon (x=0.608)
            kp2, cf2 = _horiz_kps(w, h, (0.460, 0.654), (0.608, 0.660))
            return [
                _FakeResult(
                    _FakeBoxes([bx_main, bx_fold], [0.59, 0.52], [964, 963]),
                    _FakeKeypoints([kp1, kp2], [cf1, cf2]),
                )
            ]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]

    real_zones = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )

    # Строго монохромный Ч/Б ИК-кадр (R == G == B, нулевая цветовая насыщенность S == 0)
    rng = np.random.default_rng(1748)
    gray = rng.integers(75, 165, size=(360, 640), dtype=np.uint8)
    ir_frame = np.stack([gray, gray, gray], axis=-1)

    tracker = PoseTracker("yolo11m-pose.pt", tmp_path, conf=0.48, smooth_keypoints=False, roi_probe_interval=0.0)
    fake_model = _FakeNightIrModel()
    tracker._model = fake_model  # type: ignore[assignment]
    tracker._roi_model = fake_model  # type: ignore[assignment]
    tracker._runtime_label = "TensorRT FP16 (CUDA GPU)"
    tracker._last_bed_track_id = 10201
    tracker._last_bed_bbox = (0.3199, 0.5184, 0.5943, 0.6632)
    tracker._last_bed_seen_ts = _time.monotonic()

    # 1. Проверяем подавление дублей в самом PoseTracker.infer (и Снимок А, и Снимок Б)
    fake_model.mode = "snap_a"
    det_a = tracker.infer(ir_frame, zones=real_zones)
    assert len(det_a) == 1
    assert det_a[0].track_id == 10201

    fake_model.mode = "snap_b"
    det_b = tracker.infer(ir_frame, zones=real_zones)
    assert len(det_b) == 1
    assert det_b[0].track_id == 10201

    # 2. Проверяем эшелонированную защиту в PatientIdentifier + Ч/Б ИК устойчивость visitor_vecs:
    # предварительно сохраняем в visitor_vecs и дневной цветной свитер, и монохромный серый ИК-патч одеяла!
    anchor_path = tmp_path / "patient_anchor.json"
    ident = PatientIdentifier(
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=1.5,
        door_visit_idle_timeout_sec=10.0,
        state_path=anchor_path,
    )
    fsm = FallStateMachine(sit_confirm_seconds=1.5, suppress_sitting_when_caregiver=True)

    probe_sleep = make_pose(10201, (0.3199, 0.5184, 0.5943, 0.6632), (0.540, 0.578), (0.415, 0.598))
    ir_blanket_vec = compute_torso_appearance(ir_frame, probe_sleep)
    assert ir_blanket_vec is not None
    day_frame = make_colored_room_frame([(probe_sleep, (45, 55, 170))])
    day_vis_vec = compute_torso_appearance(day_frame, probe_sleep)
    assert day_vis_vec is not None
    ident._remember_visitor_outfit(day_vis_vec)
    ident._remember_visitor_outfit(ir_blanket_vec)

    # Подаём напрямую нефильтрованные пары боксов из Снимка А и Снимка Б в PatientIdentifier.update
    t = 0.0
    for step_idx in range(12):
        t += 0.3
        obs_10201_a = make_pose(10201, (0.3199, 0.5184, 0.5943, 0.6632), (0.540, 0.578), (0.415, 0.598), conf=0.61)
        obs_965_a = make_pose(965, (0.3322, 0.4954, 0.5938, 0.6334), (0.536, 0.562), (0.420, 0.580), conf=0.54)
        obs_10201_b = make_pose(10201, (0.3178, 0.4928, 0.5975, 0.6338), (0.542, 0.558), (0.418, 0.575), conf=0.59)
        obs_963_b = make_pose(963, (0.2748, 0.5731, 0.6202, 0.7437), (0.460, 0.654), (0.608, 0.660), conf=0.52)

        batch = [obs_10201_a, obs_965_a] if (step_idx % 2 == 0) else [obs_10201_b, obs_963_b]
        roles, cg = ident.update(batch, real_zones, now=t, frame_bgr=ir_frame)
        ev = fsm.step(batch[:1], roles, cg, real_zones, now=t)
        assert ev is None
        assert list(roles.keys()) == [10201]
        assert cg is False
        assert ident.door_visit_active is False
        assert fsm.state != PatientState.CAREGIVER_ASSIST

    assert roles[10201].role == Role.PATIENT
    assert fsm.state == PatientState.IN_BED

    # 3. Проверяем, что неподвижный фантом у двери в valid_obs (висящий на косяке халат)
    # НЕ замораживает закрытие реального door_visit_active после ухода гостя!
    coat_phantom = make_pose(888, (0.20, 0.22, 0.27, 0.72), (0.235, 0.31), (0.235, 0.50), ankle=(0.235, 0.70))
    t += 0.5
    v_in_0 = make_pose(701, (0.09, 0.30, 0.19, 0.86), (0.14, 0.39), (0.14, 0.60), ankle=(0.14, 0.84))
    v_in_1 = make_pose(701, (0.14, 0.30, 0.24, 0.86), (0.19, 0.39), (0.19, 0.60), ankle=(0.19, 0.84))
    ident.update([obs_10201_a, coat_phantom, v_in_0], real_zones, now=t, frame_bgr=ir_frame)
    t += 0.3
    roles_v, cg_v = ident.update([obs_10201_a, coat_phantom, v_in_1], real_zones, now=t, frame_bgr=ir_frame)
    assert 888 not in roles_v
    assert roles_v[701].role == Role.VISITOR
    assert ident.door_visit_active is True
    assert cg_v is True

    # Через 3.2 с гость вышел через дверь, а неподвижный халат (coat_phantom) остался в valid_obs у двери
    t += 3.2
    v_out = make_pose(701, (0.10, 0.30, 0.20, 0.86), (0.15, 0.39), (0.15, 0.60), ankle=(0.15, 0.84))
    ident.update([obs_10201_a, coat_phantom, v_out], real_zones, now=t, frame_bgr=ir_frame)

    for _ in range(10):
        t += 0.35
        roles_after, cg_after = ident.update([obs_10201_a, coat_phantom], real_zones, now=t, frame_bgr=ir_frame)
    assert 888 not in roles_after
    assert ident.door_visit_active is False
    assert cg_after is False
    assert roles_after[10201].role == Role.PATIENT


# 54. Волна 6 (Задача 2): 3D-гомография плоскости пола (cv2.findHomography + матрас Z=55 см) —
# точный расчёт (floor_x_cm, floor_y_cm), физического роста стоящего взрослого vs ребёнка и устранение 2D-параллакса!
def test_wave6_floor_homography_3d_height_cm_floor_xy_and_mattress_parallax() -> None:
    bed_16_31 = [[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]]
    door_16_31 = [[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]]
    hom = FloorHomography(
        floor_polygon=DEFAULT_FLOOR_POLYGON_STD,
        floor_width_cm=200.0,
        floor_depth_cm=120.0,
        mattress_height_cm=55.0,
        bed_polygon=bed_16_31,
    )
    assert hom.is_valid is True
    assert hom.is_calibrated is True

    # Обратимость проекции пол <-> кадр на всех 4 углах калиброванного прямоугольника пола 200x120 см
    for wx, wy in ((0.0, 0.0), (200.0, 0.0), (200.0, 120.0), (0.0, 120.0), (100.0, 60.0)):
        uv = hom.world_3d_to_image_uv(wx, wy, 0.0)
        assert uv is not None
        back_xy = hom.image_to_floor_cm(uv)
        assert back_xy is not None
        assert abs(back_xy[0] - wx) < 0.8
        assert abs(back_xy[1] - wy) < 0.8

    # Генерируем проективно-точные скелеты взрослого (172 см) и ребёнка (120 см), стоящих на полу (Z=0) перед кроватью
    def _build_projected_standing_pose(tid: int, wx: float, wy: float, true_h_cm: float) -> PoseObservation:
        foot_uv = hom.world_3d_to_image_uv(wx, wy, 0.0)
        ankle_uv = hom.world_3d_to_image_uv(wx, wy, 6.0)
        hip_uv = hom.world_3d_to_image_uv(wx, wy, true_h_cm * 0.52)
        sh_uv = hom.world_3d_to_image_uv(wx, wy, true_h_cm * 0.82)
        nose_uv = hom.world_3d_to_image_uv(wx, wy, true_h_cm * 0.93)
        top_uv = hom.world_3d_to_image_uv(wx, wy, true_h_cm)
        assert foot_uv and ankle_uv and hip_uv and sh_uv and nose_uv and top_uv
        half_w = 0.042 if true_h_cm >= 150.0 else 0.030
        return make_pose(
            track_id=tid,
            bbox=(
                max(0.01, foot_uv[0] - half_w),
                max(0.01, top_uv[1]),
                min(0.99, foot_uv[0] + half_w),
                min(0.99, foot_uv[1]),
            ),
            shoulder=sh_uv,
            hip=hip_uv,
            ankle=ankle_uv,
            nose=nose_uv,
        )

    adult_obs = _build_projected_standing_pose(501, 100.0, 22.0, 172.0)
    child_obs = _build_projected_standing_pose(502, 115.0, 18.0, 120.0)

    m_adult = hom.measure_pose(adult_obs, bed_polygon=bed_16_31)
    m_child = hom.measure_pose(child_obs, bed_polygon=bed_16_31)
    assert isinstance(m_adult, FootFloorMeasurement)
    assert isinstance(m_adult, dict)
    assert m_adult.height_cm is not None and 158.0 <= m_adult.height_cm <= 185.0
    assert m_adult["height_cm"] == m_adult.height_cm
    assert m_adult.floor_xy_cm is not None
    assert abs(m_adult.floor_xy_cm[0] - 100.0) <= 6.0
    assert abs(m_adult.floor_xy_cm[1] - 22.0) <= 6.0

    assert m_child.height_cm is not None and 108.0 <= m_child.height_cm <= 134.0
    assert (m_adult.height_cm - m_child.height_cm) >= 35.0
    # Хотя 2D-таз стоящего у борта ребёнка проецируется поверх нижнего края bed_polygon, 3D-гомография отделяет пол Z=0 от матраса Z=55 см!
    assert m_child.standing_on_floor_in_front is True

    zones_hom = ZonesConfig(
        bed_polygon=bed_16_31,
        door_polygon=door_16_31,
        floor_polygon=DEFAULT_FLOOR_POLYGON_STD,
    )
    assert is_standing_in_front_of_bed(child_obs, bed_16_31, zones=zones_hom) is True
    assert is_person_on_bed(child_obs, bed_16_31, zones=zones_hom) is False

    # Сидящая НА матрасе (Z=55 см) бабушка со свешенными ногами НЕ считается стоящей на полу перед кроватью
    grandma_seated = make_pose(
        track_id=1,
        bbox=(0.35, 0.44, 0.49, 0.89),
        shoulder=(0.41, 0.53),
        hip=(0.42, 0.65),
        knee=(0.40, 0.74),
        ankle=(0.42, 0.84),
    )
    assert is_person_on_bed(grandma_seated, bed_16_31, zones=zones_hom) is True
    assert is_standing_in_front_of_bed(grandma_seated, bed_16_31, zones=zones_hom) is False

    # Проверяем запись height_cm и floor_xy_cm в TrackIdentity и рендер бейджа «Свой · ... см»
    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=1.0)
    t = 0.0
    for wx_step in (135.0, 155.0, 175.0, 195.0):
        t += 0.25
        a_step = _build_projected_standing_pose(501, wx_step, 22.0, 172.0)
        roles, _ = ident.update([grandma_seated, a_step], zones_hom, now=t)
    assert roles[501].role == Role.VISITOR
    assert roles[501].height_cm is not None and 155.0 <= roles[501].height_cm <= 188.0
    assert roles[501].floor_xy_cm is not None

    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered = render_annotated_frame(
        canvas,
        [grandma_seated, adult_obs],
        roles,
        zones_hom,
        PatientState.CAREGIVER_ASSIST,
        status_reason="рядом родственник",
    )
    assert rendered.shape == (360, 640, 3)
    assert int(rendered.sum()) > 0


# 55. Волна 6 (Задача 3): Детект ночного Ч/Б ИК-режима (mean(S) < 12 в HSV) и строгая изоляция
# дневного RGB/HSV-якоря (48D) от ночного текстурного LBP+Sobel+CLAHE якоря (64D) на диске и в памяти!
def test_wave6_night_ir_detection_and_isolated_day_rgb_vs_night_lbp_anchors(tmp_path: Path) -> None:
    # 1. Детект Ч/Б ИК-кадра по насыщенности S в HSV
    day_frame = np.full((360, 640, 3), (45, 95, 165), dtype=np.uint8)
    ir_base = np.full((360, 640), 110, dtype=np.uint8)
    night_ir_frame = np.stack([ir_base, ir_base + 1, ir_base], axis=-1)  # лёгкий шум кодека S << 12
    assert is_night_ir_frame(day_frame) is False
    assert is_night_ir_frame(night_ir_frame) is True

    # 2. Ночной 64D LBP+Sobel+CLAHE дескриптор уверенно различает вязаный/клетчатый свитер гостя
    # и гладкую светлую ночную сорочку бабушки даже при АБСОЛЮТНО НУЛЕВОЙ цветности (R == G == B)!
    obs_test = make_pose(1, (0.34, 0.44, 0.62, 0.66), (0.48, 0.50), (0.48, 0.60))

    def _make_textured_ir_frame(pattern: str) -> np.ndarray:
        g = np.full((360, 640), 85, dtype=np.uint8)
        yy, xx = np.ogrid[:360, :640]
        if pattern == "smooth_light_gown":
            # Светлая гладкая ткань с плавным градиентом
            patch = np.clip(185 + ((xx - 300) // 12), 160, 215).astype(np.uint8)
            g[140:250, 200:410] = np.broadcast_to(patch, (360, 640))[140:250, 200:410]
        elif pattern == "dark_knit_checker":
            # Контрастная мелкая клетка / крупная вязка тёмного свитера
            checker = np.where(((xx // 4) + (yy // 4)) % 2 == 0, 42, 158).astype(np.uint8)
            g[140:250, 200:410] = checker[140:250, 200:410]
        return np.stack([g, g, g], axis=-1)

    ir_gown_a = _make_textured_ir_frame("smooth_light_gown")
    ir_gown_b = np.clip(ir_gown_a.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    ir_knit = _make_textured_ir_frame("dark_knit_checker")

    vec_gown_a = compute_night_ir_appearance(ir_gown_a, obs_test)
    vec_gown_b = compute_night_ir_appearance(ir_gown_b, obs_test)
    vec_knit = compute_night_ir_appearance(ir_knit, obs_test)
    assert vec_gown_a is not None and vec_gown_a.shape == (64,)
    assert vec_knit is not None and vec_knit.shape == (64,)
    assert cosine_sim(vec_gown_a, vec_gown_b) >= 0.96
    assert cosine_sim(vec_gown_a, vec_knit) < 0.78

    # 3. Полный цикл День -> Ночь ИК -> День с сохранением в patient_anchor.json:
    # ночное обучение НИ НА БИТ не меняет anchor_vec_day, а дневное — не трогает anchor_vec_night!
    anchor_file = tmp_path / "patient_anchor.json"
    clock = [1_710_000_000.0]
    ident = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=1.0,
        state_path=anchor_file,
        wall_clock=lambda: clock[0],
    )

    grandma_bed = make_pose(1, (0.34, 0.44, 0.62, 0.66), (0.48, 0.50), (0.48, 0.60))
    day_room = make_colored_room_frame([(grandma_bed, (195, 160, 45))])
    t = 0.0
    for _ in range(6):
        t += 0.25
        ident.update([grandma_bed], BED_ZONE, now=t, frame_bgr=day_room)

    assert ident.is_night_ir is False
    assert ident.anchor_day_ready is True
    assert ident.anchor_night_ready is False
    saved_day_vec = ident._anchor_vec_day.copy()  # type: ignore[union-attr]
    assert saved_day_vec.shape == (48,)

    # Камера переключается в ночной Ч/Б ИК: обучается 64D ночной якорь + гость в вязаном свитере помечается «Свой»
    clock[0] += 600.0
    for _ in range(6):
        t += 0.25
        ident.update([grandma_bed], BED_ZONE, now=t, frame_bgr=ir_gown_a)

    assert ident.is_night_ir is True
    assert ident.anchor_day_ready is True
    assert ident.anchor_night_ready is True
    assert ident._anchor_vec_night is not None and ident._anchor_vec_night.shape == (64,)
    # Дневной якорь остался тождественно неизменным (cosine_sim == 1.0)!
    assert np.allclose(ident._anchor_vec_day, saved_day_vec, atol=1e-6)

    # Входит ночной гость через дверь (door_polygon в BED_ZONE справа x=[0.86..0.99]) в клетчатом свитере
    vis_d0 = make_pose(77, (0.86, 0.22, 0.96, 0.82), (0.91, 0.32), (0.91, 0.54), ankle=(0.91, 0.80))
    vis_d1 = make_pose(77, (0.78, 0.24, 0.88, 0.84), (0.83, 0.34), (0.83, 0.56), ankle=(0.83, 0.82))
    ir_with_vis0 = ir_gown_a.copy()
    ir_with_vis1 = ir_gown_a.copy()
    yy, xx = np.ogrid[:360, :640]
    checker_full = np.where(((xx // 4) + (yy // 4)) % 2 == 0, 40, 162).astype(np.uint8)
    ir_with_vis0[80:300, 545:620] = np.stack([checker_full]*3, axis=-1)[80:300, 545:620]
    ir_with_vis1[85:305, 495:570] = np.stack([checker_full]*3, axis=-1)[85:305, 495:570]

    t += 0.3
    ident.update([grandma_bed, vis_d0], BED_ZONE, now=t, frame_bgr=ir_with_vis0)
    t += 0.3
    roles_ir, _ = ident.update([grandma_bed, vis_d1], BED_ZONE, now=t, frame_bgr=ir_with_vis1)
    assert roles_ir[77].role == Role.VISITOR
    assert ident.force_assign_visitor(77, now=t + 0.1) is True
    assert len(ident._visitor_outfits_night) == 1
    assert len(ident._visitor_outfits_day) == 0

    # Проверяем JSON на диске и перезагрузку нового экземпляра PatientIdentifier
    disk_json = json.loads(anchor_file.read_text(encoding="utf-8"))
    assert len(disk_json["anchor_vec_day"]) == 48
    assert len(disk_json["anchor_vec_night"]) == 64
    assert len(disk_json["visitor_vecs_night"]) == 1
    assert len(disk_json["visitor_vecs_day"]) == 0

    ident_reloaded = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=1.0,
        state_path=anchor_file,
        wall_clock=lambda: clock[0] + 60.0,
    )
    assert ident_reloaded.anchor_day_ready is True
    assert ident_reloaded.anchor_night_ready is True
    assert len(ident_reloaded._visitor_outfits_night) == 1
    assert len(ident_reloaded._visitor_outfits_day) == 0


# 56. Волна 6 (Задача 7): Аудит всех сигналов «кто-то есть» — многокадровый гейт второго трека
# (confirmed_frames >= 4 и смещение >= 0.045 вне кровати/ignore) полностью блокирует 1–3-кадровые тени и складки!
def test_wave6_second_track_4_frame_motion_gate_blocks_transient_bed_phantoms() -> None:
    real_zones = make_zones_with_ignore(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
        ignore_polygons=[[[0.75, 0.10], [0.96, 0.10], [0.96, 0.65], [0.75, 0.65]]],
    )
    gate = StaticObjectGate(min_displacement=0.045)
    ident = PatientIdentifier(bed_lock_seconds=12.0, initial_bed_lock_seconds=1.0)
    fsm = FallStateMachine(sit_confirm_seconds=1.5, suppress_sitting_when_caregiver=True)

    grandma_sleep = make_pose(1, (0.34, 0.48, 0.61, 0.64), (0.53, 0.55), (0.41, 0.57))
    t = 0.0
    for _ in range(6):
        t += 0.25
        live = gate.update_and_filter([grandma_sleep], zones=real_zones, now=t)
        roles, cg = ident.update(live, real_zones, now=t)
        assert fsm.step(live, roles, cg, real_zones, now=t) is None
    assert roles[1].role == Role.PATIENT
    assert fsm.state == PatientState.IN_BED

    # Сценарий А: 3-кадровый всплеск фантомного силуэта в комнате (тень / блик ИК / спинка кровати),
    # даже если за 3 кадра он сдвинулся на 0.07 -> обязан быть отсечён (< 4 подтверждённых кадров)!
    for step_i, px in enumerate((0.68, 0.72, 0.75), start=1):
        t += 0.25
        transient_3f = make_pose(
            track_id=301,
            bbox=(px, 0.38, px + 0.09, 0.86),
            shoulder=(px + 0.045, 0.46),
            hip=(px + 0.045, 0.65),
            ankle=(px + 0.045, 0.84),
        )
        live = gate.update_and_filter([grandma_sleep, transient_3f], zones=real_zones, now=t)
        roles_a, cg_a = ident.update(live, real_zones, now=t)
        assert list(roles_a.keys()) == [1], f"Фантом просочился на кадре {step_i}: {roles_a}"
        assert cg_a is False
        assert ident.door_visit_active is False

    # Сценарий Б: 10 кадров подряд слегка дрожащего на месте предмета в комнате (disp = 0.020 < 0.045) -> отсечён!
    for j in range(10):
        t += 0.25
        dx = 0.015 if (j % 2 == 1) else 0.0
        jitter_obj = make_pose(
            track_id=302,
            bbox=(0.66 + dx, 0.40, 0.74 + dx, 0.88),
            shoulder=(0.70 + dx, 0.48),
            hip=(0.70 + dx, 0.67),
            ankle=(0.70 + dx, 0.86),
        )
        live = gate.update_and_filter([grandma_sleep, jitter_obj], zones=real_zones, now=t)
        roles_b, cg_b = ident.update(live, real_zones, now=t)
        assert list(roles_b.keys()) == [1]
        assert cg_b is False
        assert ident.door_visit_active is False

    # Сценарий В: в левом коридоре РЯДОМ с дверью, но БЕЗ пересечения door_polygon (x=0.25..0.30, y=0.86),
    # НЕ должен ложно взводиться door_visit_active! А когда реальный человек проходит >= 4 кадров с disp >= 0.045,
    # он штатно становится Role.VISITOR (co-presence без ложного флага двери).
    for step_i, wx in enumerate((0.72, 0.69, 0.66, 0.63), start=1):
        t += 0.25
        real_walker = make_pose(
            track_id=303,
            bbox=(wx, 0.36, wx + 0.10, 0.88),
            shoulder=(wx + 0.05, 0.45),
            hip=(wx + 0.05, 0.66),
            ankle=(wx + 0.05, 0.86),
        )
        live = gate.update_and_filter([grandma_sleep, real_walker], zones=real_zones, now=t)
        roles_c, cg_c = ident.update(live, real_zones, now=t)
        if step_i < 4:
            assert 303 not in roles_c
            assert cg_c is False
        else:
            assert set(roles_c.keys()) == {1, 303}
            assert roles_c[303].role == Role.VISITOR
            assert cg_c is True
            assert ident.door_visit_active is False


# 57. Волна 6.1: Оптически управляемый стабилизатор ключевых точек и бокса (OpticalKeypointStabilizer) —
# строго 0.0 RMS-дрожания на 30 кадрах ночного ИК-шума при джиттере YOLO ±0.025 и нулевой лаг при реальном движении!
def test_wave6_1_optical_keypoint_stabilizer_zero_ir_jitter_and_instant_motion_response(
    tmp_path: Path,
) -> None:
    class _FakeTensor:
        def __init__(self, arr: list | np.ndarray) -> None:
            self._arr = np.asarray(arr, dtype=np.float32)

        def int(self) -> "_FakeTensor":
            return _FakeTensor(self._arr.astype(np.int32))

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeBoxes:
        def __init__(self, xyxy: list, conf: list, ids: list | None = None) -> None:
            self.xyxy = _FakeTensor(xyxy)
            self.conf = _FakeTensor(conf)
            self.id = _FakeTensor(ids) if ids is not None else None

        def __len__(self) -> int:
            return int(len(self.conf.numpy()))

    class _FakeKeypoints:
        def __init__(self, xy: list, conf: list) -> None:
            self.xy = _FakeTensor(xy)
            self.conf = _FakeTensor(conf)

    class _FakeResult:
        def __init__(self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None) -> None:
            self.boxes = boxes
            self.keypoints = keypoints

    real_zones = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )

    # Базовая текстурированная Ч/Б ИК-сцена спальни 360x640 (кровать, подушка, складки одеяла, спящая бабушка)
    h, w = 360, 640
    yy, xx = np.ogrid[:h, :w]
    base_gray = np.clip(58 + ((xx * 3 + yy * 5) % 19), 0, 255).astype(np.float32)
    # Матрас и одеяло с мягкими складками
    bed_mask = (xx >= int(0.28 * w)) & (xx <= int(0.68 * w)) & (yy >= int(0.42 * h)) & (yy <= int(0.76 * h))
    blanket_tex = 112.0 + 14.0 * np.sin(xx / 11.0) * np.cos(yy / 9.0)
    base_gray[bed_mask] = blanket_tex[bed_mask]
    # Силуэт головы/плеч спящей бабушки на подушке
    sleeper_mask = ((xx - 0.54 * w) ** 2) / ((0.11 * w) ** 2) + ((yy - 0.56 * h) ** 2) / ((0.065 * h) ** 2) <= 1.0
    base_gray[sleeper_mask] = 168.0

    def _noisy_ir_frame(seed_idx: int, waking_up: bool = False) -> np.ndarray:
        rng_f = np.random.default_rng(70_000 + seed_idx)
        scene = base_gray.copy()
        if waking_up:
            # Реальное оптическое движение в кадре: корпус поднимается вверх, рука вскинута над головой
            scene[sleeper_mask] = 110.0
            sit_torso = (
                (xx >= int(0.42 * w))
                & (xx <= int(0.54 * w))
                & (yy >= int(0.34 * h))
                & (yy <= int(0.58 * h))
            )
            raised_arm = (
                (xx >= int(0.46 * w))
                & (xx <= int(0.51 * w))
                & (yy >= int(0.22 * h))
                & (yy <= int(0.38 * h))
            )
            scene[sit_torso] = 182.0
            scene[raised_arm] = 196.0
        ir_noise = rng_f.normal(0.0, 6.0, size=(h, w)).astype(np.float32)
        g_u8 = np.clip(np.rint(scene + ir_noise), 0, 255).astype(np.uint8)
        return np.stack([g_u8, g_u8, g_u8], axis=-1)

    class _JitteryNightYolo:
        def __init__(self) -> None:
            self.frame_idx = 0
            self.waking_up = False

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            fh, fw = source.shape[:2]
            rng_kp = np.random.default_rng(90_000 + self.frame_idx)
            if not self.waking_up:
                # Намеренный сильный ИК-джиттер YOLO до ±0.025 (на 1 кадре эталон без смещения, со 2 по 30 — тряска ±0.025)
                jx = 0.0 if self.frame_idx == 0 else float(rng_kp.uniform(-0.025, 0.025))
                jy = 0.0 if self.frame_idx == 0 else float(rng_kp.uniform(-0.025, 0.025))
                bx = [
                    (0.33 + jx * 0.8) * fw,
                    (0.49 + jy * 0.8) * fh,
                    (0.61 - jx * 0.6) * fw,
                    (0.65 - jy * 0.6) * fh,
                ]
                xy = [[0.0, 0.0] for _ in range(17)]
                cf = [0.0 for _ in range(17)]
                # Стабильные открытые точки головы/плеч/таза с плавающим от ИК-шума conf (0.43..0.85)
                c_mod = 0.0 if self.frame_idx == 0 else float(rng_kp.uniform(-0.24, 0.08))
                xy[0], cf[0] = [(0.565 + jx) * fw, (0.548 + jy) * fh], max(0.43, 0.76 + c_mod)
                xy[5], cf[5] = [(0.530 + jx) * fw, (0.552 + jy) * fh], max(0.44, 0.80 + c_mod)
                xy[6], cf[6] = [(0.530 - jx * 0.7) * fw, (0.576 - jy * 0.7) * fh], max(0.43, 0.78 + c_mod)
                xy[11], cf[11] = [(0.415 + jx * 0.9) * fw, (0.566 + jy * 0.9) * fh], max(0.42, 0.72 + c_mod)
                xy[12], cf[12] = [(0.415 - jx * 0.8) * fw, (0.590 - jy * 0.8) * fh], max(0.42, 0.70 + c_mod)

                # Под-одеяльный шум конечностей (< 0.48) и редкие одиночные ИК-вспышки 0.56 с прыгающими координатами
                xy[9], cf[9] = [(0.48 + jx * 1.5) * fw, (0.52 + jy * 1.5) * fh], 0.34 + 0.12 * ((self.frame_idx % 3) / 2.0)
                if self.frame_idx in (7, 18, 25):
                    xy[13], cf[13] = [(0.36 + jx * 2.0) * fw, (0.62 - jy * 2.0) * fh], 0.57
                else:
                    xy[13], cf[13] = [(0.36 + jx) * fw, (0.62 + jy) * fh], 0.31
                return [_FakeResult(_FakeBoxes([bx], [0.68], [77]), _FakeKeypoints([xy], [cf]))]

            # Фаза реального подъёма корпуса + взмах правой руки вверх (HELP_CALL / SITTING_UP)
            bx_sit = [0.37 * fw, 0.21 * fh, 0.57 * fw, 0.63 * fh]
            xy_sit = [[0.0, 0.0] for _ in range(17)]
            cf_sit = [0.0 for _ in range(17)]
            xy_sit[0], cf_sit[0] = [0.480 * fw, 0.335 * fh], 0.90
            xy_sit[5], cf_sit[5] = [0.465 * fw, 0.400 * fh], 0.91
            xy_sit[6], cf_sit[6] = [0.495 * fw, 0.400 * fh], 0.91
            xy_sit[8], cf_sit[8] = [0.495 * fw, 0.315 * fh], 0.87
            xy_sit[10], cf_sit[10] = [0.490 * fw, 0.235 * fh], 0.89  # Взмах запястья выше головы!
            xy_sit[11], cf_sit[11] = [0.465 * fw, 0.570 * fh], 0.86
            xy_sit[12], cf_sit[12] = [0.495 * fw, 0.570 * fh], 0.86
            return [_FakeResult(_FakeBoxes([bx_sit], [0.88], [77]), _FakeKeypoints([xy_sit], [cf_sit]))]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]

    tracker = PoseTracker(
        "yolo11m-pose.pt",
        tmp_path,
        conf=0.48,
        smooth_keypoints=True,
        roi_probe_interval=0.0,
    )
    fake_yolo = _JitteryNightYolo()
    tracker._model = fake_yolo  # type: ignore[assignment]
    tracker._roi_model = fake_yolo  # type: ignore[assignment]

    ident = PatientIdentifier(bed_lock_seconds=12.0, initial_bed_lock_seconds=0.4)
    fsm = FallStateMachine(sit_confirm_seconds=0.5, help_confirm_seconds=0.35)

    # 1. Прогоняем 30 кадров неподвижного ИК-сна с гауссовым шумом матрицы и джиттером ключевых точек ±0.025
    sleep_kps_history: list[np.ndarray] = []
    sleep_bbox_history: list[np.ndarray] = []
    hud_green_masks: list[np.ndarray] = []

    t = 100.0
    for idx in range(30):
        t += 0.083  # 12 Гц
        fake_yolo.frame_idx = idx
        ir_frame = _noisy_ir_frame(idx, waking_up=False)
        obs_list = tracker.infer(ir_frame, zones=real_zones)
        assert len(obs_list) == 1
        obs = obs_list[0]
        assert obs.is_still is True, f"Кадр {idx}: спящий трек обязан иметь is_still=True"
        assert obs._kp_stabilized is True

        # Под-одеяльные точки с conf < 0.48 (и одиночные ИК-вспышки 0.57 на kp 13) не просачиваются в UI (>= 0.48),
        # оставаясь замороженными с pub_conf <= 0.47 для внутренней анатомии filter_valid_poses
        assert obs.kp(9, 0.48) is None and 0.30 <= obs.keypoints_norm[9][2] <= 0.47
        assert obs.kp(13, 0.48) is None and 0.30 <= obs.keypoints_norm[13][2] <= 0.47

        sleep_kps_history.append(np.asarray(obs.keypoints_norm, dtype=np.float64))
        sleep_bbox_history.append(np.asarray(obs.bbox_norm, dtype=np.float64))

        roles, cg = ident.update(obs_list, real_zones, now=t, frame_bgr=ir_frame)
        roles[obs.track_id].role = Role.PATIENT
        fsm.step(obs_list, roles, cg, real_zones, now=t)

        # Проверяем, что и на серверном снимке render_annotated_frame зелёный скелет/рамка стоят абсолютно неподвижно
        hud = render_annotated_frame(ir_frame, obs_list, roles, real_zones, PatientState.IN_BED)
        green_core = (hud[:, :, 0] == 90) & (hud[:, :, 1] == 215) & (hud[:, :, 2] == 95)
        assert int(np.sum(green_core)) > 200
        hud_green_masks.append(green_core)

    kps_arr = np.stack(sleep_kps_history, axis=0)  # (30, 17, 3)
    bbox_arr = np.stack(sleep_bbox_history, axis=0)  # (30, 4)
    rms_kp_jitter = float(np.sqrt(np.mean((kps_arr[:, :, :2] - kps_arr[0:1, :, :2]) ** 2)))
    rms_conf_jitter = float(np.sqrt(np.mean((kps_arr[:, :, 2] - kps_arr[0:1, :, 2]) ** 2)))
    rms_bbox_jitter = float(np.sqrt(np.mean((bbox_arr - bbox_arr[0:1, :]) ** 2)))

    assert rms_kp_jitter == 0.0, f"Ожидалось 0.0 RMS-дрожания keypoints_norm, получено {rms_kp_jitter}"
    assert rms_conf_jitter == 0.0, f"Ожидалось 0.0 мигания conf в покое, получено {rms_conf_jitter}"
    assert rms_bbox_jitter == 0.0, f"Ожидалось 0.0 RMS-дрожания bbox_norm, получено {rms_bbox_jitter}"
    for idx in range(1, 30):
        assert np.array_equal(hud_green_masks[idx], hud_green_masks[0]), (
            f"Серверный HUD сдвинул пиксели скелета/рамки на кадре {idx}"
        )

    # 2. Дополнительно проверяем прямой вызов render_annotated_frame на СЫРЫХ (не прошедших PoseTracker) точках с джиттером
    raw_hud_masks: list[np.ndarray] = []
    for idx in range(5):
        ir_f = _noisy_ir_frame(500 + idx, waking_up=False)
        jx = 0.0 if idx == 0 else (0.014 if idx % 2 == 1 else -0.014)
        raw_obs = make_pose(77, (0.33 + jx, 0.49, 0.61 + jx, 0.65), (0.53 + jx, 0.56), (0.415 + jx, 0.58))
        roles_raw = {77: roles[obs.track_id]}
        hud_raw = render_annotated_frame(ir_f, [raw_obs], roles_raw, real_zones, PatientState.IN_BED)
        raw_core = (hud_raw[:, :, 0] == 90) & (hud_raw[:, :, 1] == 215) & (hud_raw[:, :, 2] == 95)
        assert int(np.sum(raw_core)) > 200
        raw_hud_masks.append(raw_core)
    for idx in range(1, 5):
        assert np.array_equal(raw_hud_masks[idx], raw_hud_masks[0])

    # 3. Кадры 31..36: реальный резкий подъём корпуса + взмах руки вверх -> мгновенное раскрытие без лага!
    fake_yolo.waking_up = True
    help_triggered = False
    for wake_step in range(6):
        t += 0.12
        fake_yolo.frame_idx = 30 + wake_step
        wake_frame = _noisy_ir_frame(30 + wake_step, waking_up=True)
        # Проверяем, что оптический детектор движения видит реальный подъём в ROI (>= 0.012)
        if wake_step == 0:
            cur_gs = OpticalKeypointStabilizer._to_gray_small(wake_frame)
            prev_gs = OpticalKeypointStabilizer._to_gray_small(_noisy_ir_frame(29, waking_up=False))
            opt_m = OpticalKeypointStabilizer.compute_roi_optical_motion(
                cur_gs, prev_gs, (0.37, 0.21, 0.57, 0.63), (0.33, 0.49, 0.61, 0.65)
            )
            assert opt_m >= 0.012, f"Оптическая энергия реального подъёма слишком мала: {opt_m}"

        wake_list = tracker.infer(wake_frame, zones=real_zones)
        assert len(wake_list) == 1
        w_obs = wake_list[0]
        if wake_step == 0:
            # Уже на ПЕРВОМ кадре движения защёлка покоя снята (is_still=False),
            # плечи прошли > 65% амплитуды рывка (0.552 -> <= 0.448), а запястье открылось сразу без 3-кадрового лага!
            assert w_obs.is_still is False
            assert w_obs.mid_shoulder is not None and w_obs.mid_shoulder[1] <= 0.448
            wrist_pt = w_obs.kp(10, 0.48)
            assert wrist_pt is not None and abs(wrist_pt[1] - 0.235) <= 0.015

        roles_w, cg_w = ident.update(wake_list, real_zones, now=t, frame_bgr=wake_frame)
        roles_w[w_obs.track_id].role = Role.PATIENT
        ev = fsm.step(wake_list, roles_w, cg_w, real_zones, now=t)
        if ev is not None and fsm.state == PatientState.HELP_CALL:
            help_triggered = True

    assert help_triggered is True and fsm.state == PatientState.HELP_CALL


# 58. Волна 6.1 (хотфикс строба #108): непрерывное оптическое удержание (coast) спящего трека в bed_polygon
# при провалах детекта YOLO на одеяле (чередование obs / [] вида 1x5, 0x28, 1x1, 0x17 при optical_motion_ratio < 0.012)
# и мгновенный сброс coast на первом же кадре реального движения (>= 0.012)!
def test_wave6_1_bed_sleeper_optical_coast_eliminates_intermittent_yolo_dropouts_and_resets_on_motion(
    tmp_path: Path,
) -> None:
    class _FakeTensor:
        def __init__(self, arr: list | np.ndarray) -> None:
            self._arr = np.asarray(arr, dtype=np.float32)

        def int(self) -> "_FakeTensor":
            return _FakeTensor(self._arr.astype(np.int32))

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeBoxes:
        def __init__(self, xyxy: list, conf: list, ids: list | None = None) -> None:
            self.xyxy = _FakeTensor(xyxy)
            self.conf = _FakeTensor(conf)
            self.id = _FakeTensor(ids) if ids is not None else None

        def __len__(self) -> int:
            return int(len(self.conf.numpy()))

    class _FakeKeypoints:
        def __init__(self, xy: list, conf: list) -> None:
            self.xy = _FakeTensor(xy)
            self.conf = _FakeTensor(conf)

    class _FakeResult:
        def __init__(self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None) -> None:
            self.boxes = boxes
            self.keypoints = keypoints

    real_zones = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )

    h, w = 360, 640
    yy, xx = np.ogrid[:h, :w]
    base_gray = np.clip(54 + ((xx * 5 + yy * 3) % 21), 0, 255).astype(np.float32)
    bed_mask = (xx >= int(0.28 * w)) & (xx <= int(0.68 * w)) & (yy >= int(0.42 * h)) & (yy <= int(0.76 * h))
    base_gray[bed_mask] = (108.0 + 12.0 * np.sin(xx / 10.0) * np.cos(yy / 8.0))[bed_mask]
    sleeper_mask = ((xx - 0.53 * w) ** 2) / ((0.10 * w) ** 2) + ((yy - 0.56 * h) ** 2) / ((0.06 * h) ** 2) <= 1.0
    base_gray[sleeper_mask] = 166.0

    def _make_ir_frame(seed_idx: int, empty_bed_moved: bool = False) -> np.ndarray:
        rng_f = np.random.default_rng(120_000 + seed_idx)
        scene = base_gray.copy()
        if empty_bed_moved:
            # Сдвиг одеяла/уход человека из кровати (реальное оптическое движение в спальном боксе >= 0.012)
            scene[sleeper_mask] = 86.0
            fold_shift = (xx >= int(0.36 * w)) & (xx <= int(0.58 * w)) & (yy >= int(0.50 * h)) & (yy <= int(0.64 * h))
            scene[fold_shift] = 52.0
        noise = rng_f.normal(0.0, 5.5, size=(h, w)).astype(np.float32)
        g_u8 = np.clip(np.rint(scene + noise), 0, 255).astype(np.uint8)
        return np.stack([g_u8, g_u8, g_u8], axis=-1)

    # Боевой RLE-паттерн из приёмки #108: 5 кадров видно -> 28 кадров пусто -> 1 кадр видно -> 17 пусто -> 1 видно -> 10 пусто (62 кадра!)
    raw_schedule: list[bool] = (
        [True] * 5 + [False] * 28 + [True] * 1 + [False] * 17 + [True] * 1 + [False] * 10
    )

    class _DroppingIrModel:
        def __init__(self) -> None:
            self.emit_person = True
            self.step_idx = 0

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            if not self.emit_person:
                return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]
            fh, fw = source.shape[:2]
            rng_k = np.random.default_rng(140_000 + self.step_idx)
            jx = 0.0 if self.step_idx == 0 else float(rng_k.uniform(-0.014, 0.014))
            jy = 0.0 if self.step_idx == 0 else float(rng_k.uniform(-0.014, 0.014))
            bx = [(0.34 + jx) * fw, (0.49 + jy) * fh, (0.60 + jx) * fw, (0.65 + jy) * fh]
            xy = [[0.0, 0.0] for _ in range(17)]
            cf = [0.0 for _ in range(17)]
            xy[0], cf[0] = [(0.555 + jx) * fw, (0.548 + jy) * fh], 0.78
            xy[5], cf[5] = [(0.525 + jx) * fw, (0.555 + jy) * fh], 0.80
            xy[6], cf[6] = [(0.525 + jx) * fw, (0.575 + jy) * fh], 0.79
            xy[11], cf[11] = [(0.415 + jx) * fw, (0.568 + jy) * fh], 0.74
            xy[12], cf[12] = [(0.415 + jx) * fw, (0.588 + jy) * fh], 0.73
            # Намеренно меняем сырой track_id на редких одиночных вспышках (88 -> 91 -> 94)
            raw_tid = 88 + (self.step_idx % 7)
            return [_FakeResult(_FakeBoxes([bx], [0.62], [raw_tid]), _FakeKeypoints([xy], [cf]))]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            # Даже ROI-фолбэк под плотным одеялом промахивается!
            return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]

    # Важно: используем боевой roi_probe_interval=1.6, чтобы проверить и кэш _cached_roi_obs на строке 1122, и оптический coast!
    tracker = PoseTracker(
        "yolo11m-pose.pt",
        tmp_path,
        conf=0.48,
        smooth_keypoints=True,
        roi_probe_interval=1.6,
    )
    dropping_yolo = _DroppingIrModel()
    tracker._model = dropping_yolo  # type: ignore[assignment]
    tracker._roi_model = dropping_yolo  # type: ignore[assignment]

    out_counts: list[int] = []
    out_kps: list[np.ndarray] = []
    out_bboxes: list[np.ndarray] = []

    for i, present in enumerate(raw_schedule):
        dropping_yolo.step_idx = i
        dropping_yolo.emit_person = present
        ir_f = _make_ir_frame(i, empty_bed_moved=False)
        res = tracker.infer(ir_f, zones=real_zones)
        out_counts.append(len(res))
        assert len(res) == 1, f"Строб/пропуск трека на кадре {i} (raw present={present})"
        assert res[0].is_still is True
        assert res[0]._kp_stabilized is True
        out_kps.append(np.asarray(res[0].keypoints_norm, dtype=np.float64))
        out_bboxes.append(np.asarray(res[0].bbox_norm, dtype=np.float64))

    assert out_counts == [1] * len(raw_schedule)
    kps_stack = np.stack(out_kps, axis=0)
    bbox_stack = np.stack(out_bboxes, axis=0)
    assert float(np.sqrt(np.mean((kps_stack[:, :, :2] - kps_stack[0:1, :, :2]) ** 2))) == 0.0
    assert float(np.sqrt(np.mean((bbox_stack - bbox_stack[0:1, :]) ** 2))) == 0.0

    # Как только в зоне кровати возникает реальное оптическое движение (>= 0.012) при пустом ответе YOLO —
    # оптическое удержание (coast) мгновенно сбрасывается на первом же кадре!
    dropping_yolo.step_idx = 999
    dropping_yolo.emit_person = False
    moved_empty_frame = _make_ir_frame(999, empty_bed_moved=True)
    res_after_move = tracker.infer(moved_empty_frame, zones=real_zones)
    assert res_after_move == []


# 59. Волна 6.1 (хотфикс ложного MISSING #116):
# спящий под одеялом трек с 5 анатомическими точками conf = 0.34..0.46 (плюс 1 UI-точка >= 0.48)
# и паузами детекта > 40 с проходит полную цепочку OpticalKeypointStabilizer -> PatientIdentifier -> FallStateMachine
# в течение 95 с без ложного MISSING, с 0.0 джиттера UI-точек, и мгновенно сбрасывает coast при anchor_drift >= 0.050!
def test_wave6_1_subthreshold_blanket_anatomy_and_long_optical_coast_never_trigger_false_missing() -> None:
    real_zones = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )

    h, w = 360, 640
    yy, xx = np.ogrid[:h, :w]
    base_gray = np.clip(56 + ((xx * 3 + yy * 7) % 17), 0, 255).astype(np.float32)
    bed_mask = (xx >= int(0.28 * w)) & (xx <= int(0.68 * w)) & (yy >= int(0.42 * h)) & (yy <= int(0.76 * h))
    base_gray[bed_mask] = (110.0 + 13.0 * np.sin(xx / 12.0) * np.cos(yy / 9.0))[bed_mask]
    pillow_head = ((xx - 0.54 * w) ** 2) / ((0.09 * w) ** 2) + ((yy - 0.56 * h) ** 2) / ((0.055 * h) ** 2) <= 1.0
    base_gray[pillow_head] = 164.0

    def _ir_sleep_frame(step_i: int, drifted_surface: bool = False) -> np.ndarray:
        rng_f = np.random.default_rng(220_000 + step_i)
        scene = base_gray.copy()
        if drifted_surface:
            scene[pillow_head] = 92.0
            blanket_flat = (xx >= int(0.36 * w)) & (xx <= int(0.58 * w)) & (yy >= int(0.50 * h)) & (yy <= int(0.64 * h))
            scene[blanket_flat] = 74.0
        noise = rng_f.normal(0.0, 5.0, size=(h, w)).astype(np.float32)
        g_u8 = np.clip(np.rint(scene + noise), 0, 255).astype(np.uint8)
        return np.stack([g_u8, g_u8, g_u8], axis=-1)

    def _raw_blanket_obs(step_i: int) -> PoseObservation:
        rng_k = np.random.default_rng(260_000 + step_i)
        jx = 0.0 if step_i == 0 else float(rng_k.uniform(-0.012, 0.012))
        jy = 0.0 if step_i == 0 else float(rng_k.uniform(-0.012, 0.012))
        kps = [(0.0, 0.0, 0.0)] * 17
        # 5 подпороговых анатомических точек 0.34..0.46 (< 0.54!) + 1 видимая в UI точка макушки/носа 0.58:
        # раньше зануление точек < 0.54 оставляло всего 1 точку (< min_valid_kps=4) и вызывало ложный MISSING через 60 с!
        kps[0] = (0.555 + jx, 0.548 + jy, 0.58)   # единственная точка >= 0.54 (видна в UI)
        kps[1] = (0.560 + jx, 0.544 + jy, 0.44)   # глаз 0.44 (< 0.48)
        kps[2] = (0.552 - jx, 0.551 - jy, 0.43)   # глаз 0.43 (< 0.48)
        kps[5] = (0.528 + jx, 0.556 + jy, 0.46)   # левое плечо 0.46 (< 0.48)
        kps[6] = (0.528 - jx, 0.576 - jy, 0.38)   # правое плечо 0.38 (< 0.48)
        kps[11] = (0.418 + jx, 0.568 + jy, 0.34)  # левое бедро 0.34 (< 0.48)
        return PoseObservation(
            track_id=9001,
            bbox_norm=(0.34 + jx * 0.5, 0.49 + jy * 0.5, 0.60 + jx * 0.5, 0.65 + jy * 0.5),
            conf=0.31,  # ослабленный одеялом бокс (< 0.48, идёт через is_blanket_sleeper_in_bed!)
            keypoints_norm=tuple(kps),
            liveness_verified=True,
            confirmed_frames=12,
        )

    stabilizer = OpticalKeypointStabilizer()
    ident = PatientIdentifier(bed_lock_seconds=12.0, initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(missing_confirm_seconds=60.0, night_strict_mode=True)

    t = 0.0
    dt = 0.5  # 190 шагов по 0.5 с = 95.0 секунд сна (>> 60 с missing_confirm и >> 15 с старого coast!)
    ui_nose_coords: list[tuple[float, float]] = []

    for step_i in range(190):
        t += dt
        ir_f = _ir_sleep_frame(step_i, drifted_surface=False)
        # Между t=25 с и t=70 с (45 секунд подряд!) сырой YOLO вообще молчит (проверка длинного оптического coast > 15 с),
        # а до t=25 с и после t=70 с отдаёт слабый под-одеяльный скелет с 5 точками 0.34..0.46!
        raw_in = [] if (50 <= step_i < 140) else [_raw_blanket_obs(step_i)]
        stab_list = stabilizer.stabilize(raw_in, frame_bgr=ir_f, zones=real_zones, now=t)
        assert len(stab_list) == 1, f"Потерян спящий трек на шаге {step_i} (t={t:.1f}s)"
        s_obs = stab_list[0]
        assert s_obs.is_still is True
        assert s_obs.valid_keypoint_count(0.30) == 6

        # Подпороговые точки 0.34..0.46 сохранены для PatientIdentifier/FSM, но строго скрыты от UI (< 0.48)
        for sub_idx in (1, 2, 5, 6, 11):
            assert s_obs.kp(sub_idx, 0.30) is not None
            assert s_obs.kp(sub_idx, 0.48) is None

        nose_ui = s_obs.kp(0, 0.48)
        assert nose_ui is not None
        ui_nose_coords.append(nose_ui)

        roles, cg = ident.update(stab_list, real_zones, now=t, frame_bgr=ir_f)
        assert 9001 in roles, f"PatientIdentifier.filter_valid_poses отбросил трек на t={t:.1f}s!"
        if t >= 1.5:
            assert roles[9001].role == Role.PATIENT

        ev = fsm.step(stab_list, roles, cg, real_zones, now=t)
        assert ev is None, f"Ложная тревога {ev} на t={t:.1f}s при спящей бабушке!"
        if t >= 2.0:
            assert fsm.state == PatientState.IN_BED

    nose_arr = np.asarray(ui_nose_coords, dtype=np.float64)
    assert float(np.sqrt(np.mean((nose_arr - nose_arr[0:1, :]) ** 2))) == 0.0

    # Проверка кумулятивного anchor_drift: два кадра подряд с уже изменённым рельефом одеяла (межкадровый diff == 0,
    # но отличие от эталонного патча сна anchor_roi_blur >= 0.050) обязаны сбросить coast!
    drifted_f1 = _ir_sleep_frame(500, drifted_surface=True)
    drifted_f2 = _ir_sleep_frame(501, drifted_surface=True)
    assert stabilizer.stabilize([], frame_bgr=drifted_f1, zones=real_zones, now=t + 0.5) == []
    assert stabilizer.stabilize([], frame_bgr=drifted_f2, zones=real_zones, now=t + 1.0) == []


# 60. Волна 6.1 (боевой перехват #118, 22:40–22:44):
# 1) Неподвижный вертикальный ИК-силуэт tid=31 у границы двери/кровати (bbox=[0.2388, 0.1768, 0.3299, 0.6793])
#    с шумовым джиттером рамки до ±0.032 (> min_displacement=0.045!) на 100% отсекается StaticObjectGate без пиксельного
#    движения (0 ложных visitor / border_standing_entry / caregiver), а реальный идущий человек проходит штатно!
# 2) Спящий вдоль кровати бокс tid=10030 (bbox=[0.3928, 0.5022, 0.5786, 0.7859], ar=0.655, под-одеяльное колено conf=0.36
#    за выемкой bed_polygon) + одиночный 1-кадровый ИК-всплеск во время паузы YOLO держит is_still=True и 0 пустых пакетов!
def test_wave6_1_night_ir_static_door_phantom_and_foreshortened_blanket_sleeper_zero_strobe(
    tmp_path: Path,
) -> None:
    class _FakeTensor:
        def __init__(self, arr: list | np.ndarray) -> None:
            self._arr = np.asarray(arr, dtype=np.float32)

        def int(self) -> "_FakeTensor":
            return _FakeTensor(self._arr.astype(np.int32))

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeBoxes:
        def __init__(self, xyxy: list, conf: list, ids: list | None = None) -> None:
            self.xyxy = _FakeTensor(xyxy)
            self.conf = _FakeTensor(conf)
            self.id = _FakeTensor(ids) if ids is not None else None

        def __len__(self) -> int:
            return int(len(self.conf.numpy()))

    class _FakeKeypoints:
        def __init__(self, xy: list, conf: list) -> None:
            self.xy = _FakeTensor(xy)
            self.conf = _FakeTensor(conf)

    class _FakeResult:
        def __init__(self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None) -> None:
            self.boxes = boxes
            self.keypoints = keypoints

    real_zones = ZonesConfig(
        bed_polygon=[[0.31, 0.41], [0.67, 0.42], [0.69, 0.58], [0.60, 0.65], [0.60, 0.79], [0.28, 0.79], [0.26, 0.58]],
        door_polygon=[[0.07, 0.27], [0.23, 0.17], [0.28, 0.81], [0.18, 0.94]],
    )

    h, w = 360, 640
    yy, xx = np.ogrid[:h, :w]
    base_gray = np.clip(54 + ((xx * 5 + yy * 3) % 19), 0, 255).astype(np.float32)
    # Неподвижный вертикальный дверной косяк/халат в точном боксе tid=31 [0.2388, 0.1768, 0.3299, 0.6793]
    door_jamb_mask = (xx >= int(0.24 * w)) & (xx <= int(0.325 * w)) & (yy >= int(0.18 * h)) & (yy <= int(0.68 * h))
    base_gray[door_jamb_mask] = 128.0
    # Спящая вдоль матраса бабушка в боксе [0.3928, 0.5022, 0.5786, 0.7859]
    sleeper_mask = (xx >= int(0.39 * w)) & (xx <= int(0.58 * w)) & (yy >= int(0.50 * h)) & (yy <= int(0.78 * h))
    base_gray[sleeper_mask] = (146.0 + 10.0 * np.cos(xx / 10.0) * np.cos(yy / 9.0))[sleeper_mask]

    def _make_ir_2240_frame(
        step_i: int,
        single_frame_glint: bool = False,
        walker_x: float | None = None,
    ) -> np.ndarray:
        rng_f = np.random.default_rng(310_000 + step_i)
        scene = base_gray.copy()
        if single_frame_glint:
            # Одиночный 1-кадровый локальный ИК-блик на краю одеяла (~2% окна coast_opt_ratio ~ 0.018..0.024, anchor_drift < 0.050)
            glint = (xx >= int(0.41 * w)) & (xx <= int(0.44 * w)) & (yy >= int(0.52 * h)) & (yy <= int(0.56 * h))
            scene[glint] += 26.0
        if walker_x is not None:
            wx1 = int(max(0, walker_x * w))
            wx2 = int(min(w, (walker_x + 0.09) * w))
            scene[int(0.22 * h) : int(0.84 * h), wx1:wx2] = 186.0
        noise = rng_f.normal(0.0, 5.2, size=(h, w)).astype(np.float32)
        g_u8 = np.clip(np.rint(scene + noise), 0, 255).astype(np.uint8)
        return np.stack([g_u8, g_u8, g_u8], axis=-1)

    class _Night2240Model:
        def __init__(self) -> None:
            self.step_i = 0
            self.walker_x: float | None = None

        def _sleeper_10030(self, fw: float, fh: float, jx: float, jy: float) -> tuple[list[float], list, list]:
            # Точный бокс из дампа #118: [0.3928, 0.5022, 0.5786, 0.7859] (ar = 0.655 < 0.85, угол ~ 36° вдоль кровати)
            bx = [(0.3928 + jx) * fw, (0.5022 + jy) * fh, (0.5786 + jx) * fw, (0.7859 + jy) * fh]
            xy = [[0.0, 0.0] for _ in range(17)]
            cf = [0.0 for _ in range(17)]
            xy[0], cf[0] = [(0.515 + jx) * fw, (0.535 + jy) * fh], 0.68
            xy[5], cf[5] = [(0.485 + jx) * fw, (0.560 + jy) * fh], 0.72
            xy[6], cf[6] = [(0.535 + jx) * fw, (0.565 + jy) * fh], 0.70
            xy[11], cf[11] = [(0.465 + jx) * fw, (0.685 + jy) * fh], 0.58
            xy[12], cf[12] = [(0.505 + jx) * fw, (0.690 + jy) * fh], 0.56
            # Шумное под-одеяльное колено conf=0.36 (< 0.48) за диагональной выемкой bed_polygon справа (x=0.64, y=0.69)
            xy[14], cf[14] = [(0.642 + jx) * fw, (0.692 + jy) * fh], 0.36
            return bx, xy, cf

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            fh, fw = source.shape[:2]
            rng_k = np.random.default_rng(410_000 + self.step_i)
            # Сильный джиттер рамки фантома tid=31 до ±0.032 (размах 0.064 > min_displacement=0.045!)
            pjx = float(rng_k.uniform(-0.032, 0.032))
            pjy = float(rng_k.uniform(-0.015, 0.015))
            ph_bx = [(0.2388 + pjx) * fw, (0.1768 + pjy) * fh, (0.3299 + pjx) * fw, (0.6793 + pjy) * fh]
            ph_xy = [[0.0, 0.0] for _ in range(17)]
            ph_cf = [0.0 for _ in range(17)]
            ph_xy[0], ph_cf[0] = [(0.284 + pjx) * fw, (0.235 + pjy) * fh], 0.66
            ph_xy[5], ph_cf[5] = [(0.264 + pjx) * fw, (0.310 + pjy) * fh], 0.74
            ph_xy[6], ph_cf[6] = [(0.304 + pjx) * fw, (0.312 + pjy) * fh], 0.73
            ph_xy[11], ph_cf[11] = [(0.266 + pjx) * fw, (0.495 + pjy) * fh], 0.70
            ph_xy[12], ph_cf[12] = [(0.302 + pjx) * fw, (0.498 + pjy) * fh], 0.69
            ph_xy[15], ph_cf[15] = [(0.272 + pjx) * fw, (0.660 + pjy) * fh], 0.61
            ph_xy[16], ph_cf[16] = [(0.298 + pjx) * fw, (0.662 + pjy) * fh], 0.60

            boxes_xyxy = [ph_bx]
            boxes_conf = [0.64]
            boxes_ids: list[int] | None = [31]
            kps_xy = [ph_xy]
            kps_cf = [ph_cf]

            # Кадры 0..5: полнокадровый YOLO видит спящую как tid=9;
            # Кадры 6..19: ByteTrack теряет ID (возвращает ids=None на кровати или пусто в кадрах 12..29!)
            sjx = 0.0 if self.step_i == 0 else float(rng_k.uniform(-0.010, 0.010))
            sjy = 0.0 if self.step_i == 0 else float(rng_k.uniform(-0.010, 0.010))
            if self.step_i < 6:
                s_bx, s_xy, s_cf = self._sleeper_10030(fw, fh, sjx, sjy)
                boxes_xyxy.append(s_bx)
                boxes_conf.append(0.58)
                assert boxes_ids is not None
                boxes_ids.append(9)
                kps_xy.append(s_xy)
                kps_cf.append(s_cf)
            elif 6 <= self.step_i < 12:
                # Имитация анонимной выдачи без ByteTrack ID (ids=None), рождавшей tid=10030
                s_bx, s_xy, s_cf = self._sleeper_10030(fw, fh, sjx, sjy)
                boxes_xyxy.append(s_bx)
                boxes_conf.append(0.49)
                boxes_ids = None
                kps_xy.append(s_xy)
                kps_cf.append(s_cf)

            if self.walker_x is not None:
                wx = self.walker_x
                w_bx = [wx * fw, 0.22 * fh, (wx + 0.09) * fw, 0.84 * fh]
                w_xy = [[0.0, 0.0] for _ in range(17)]
                w_cf = [0.0 for _ in range(17)]
                w_xy[0], w_cf[0] = [(wx + 0.045) * fw, 0.28 * fh], 0.88
                w_xy[5], w_cf[5] = [(wx + 0.025) * fw, 0.36 * fh], 0.90
                w_xy[6], w_cf[6] = [(wx + 0.065) * fw, 0.36 * fh], 0.90
                w_xy[11], w_cf[11] = [(wx + 0.028) * fw, 0.58 * fh], 0.88
                w_xy[12], w_cf[12] = [(wx + 0.062) * fw, 0.58 * fh], 0.88
                w_xy[15], w_cf[15] = [(wx + 0.032) * fw, 0.82 * fh], 0.85
                boxes_xyxy.append(w_bx)
                boxes_conf.append(0.84)
                if boxes_ids is not None:
                    boxes_ids.append(777)
                kps_xy.append(w_xy)
                kps_cf.append(w_cf)

            return [_FakeResult(_FakeBoxes(boxes_xyxy, boxes_conf, boxes_ids), _FakeKeypoints(kps_xy, kps_cf))]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]

    tracker = PoseTracker("yolo11m-pose.pt", tmp_path, conf=0.48, smooth_keypoints=True, roi_probe_interval=0.5)
    model_2240 = _Night2240Model()
    tracker._model = model_2240  # type: ignore[assignment]
    tracker._roi_model = model_2240  # type: ignore[assignment]
    tracker._runtime_label = "TensorRT FP16 (CUDA GPU)"

    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(missing_confirm_seconds=60.0, night_strict_mode=True)

    t = 0.0
    bed_ids_seen: set[int] = set()
    for step_i in range(45):
        t += 0.25
        model_2240.step_i = step_i
        # На шаге 20 (посреди полного молчания YOLO по кровати!) даём одиночный 1-кадровый ИК-всплеск
        ir_f = _make_ir_2240_frame(step_i, single_frame_glint=(step_i == 20))
        obs_list = tracker.infer(ir_f, zones=real_zones)

        # 1. Стоячий неподвижный фантом tid=31 (или его анонимный двойник у x~0.28) отсечён на 100% кадров!
        assert len(obs_list) == 1, f"Ожидался ровно 1 трек спящей бабушки на шаге {step_i}, получено: {obs_list}"
        bed_obs = obs_list[0]
        assert bed_obs.is_still is True, f"Спящий бокс вдоль кровати обязан иметь is_still=True (шаг {step_i})"
        assert are_legs_hanging_outside_bed(bed_obs, real_zones.bed_polygon) is False
        bed_ids_seen.add(bed_obs.track_id)

        roles, cg = ident.update(obs_list, real_zones, now=t, frame_bgr=ir_f)
        ev = fsm.step(obs_list, roles, cg, real_zones, now=t)
        assert ev is None
        assert cg is False, f"Ложный caregiver на шаге {step_i}: {roles}"
        assert ident.door_visit_active is False
        assert fsm.state != PatientState.CAREGIVER_ASSIST
        if t >= 1.5:
            assert roles[bed_obs.track_id].role == Role.PATIENT
            assert fsm.state == PatientState.IN_BED

    # Сквозной стабильный ID спящей бабушки без скачка 9 -> 10030 и 0 пустых пакетов!
    assert bed_ids_seen == {9}

    # 2. А когда через дверь РЕАЛЬНО входит и шагает по картинке живой родственник (tid=777),
    # StaticObjectGate подтверждает оптическое движение пикселей за 1 межкадровый шаг -> штатный VISITOR / CAREGIVER_ASSIST!
    for w_idx, wx in enumerate((0.12, 0.17, 0.22), start=1):
        t += 0.25
        model_2240.step_i = 100 + w_idx
        model_2240.walker_x = wx
        ir_walk = _make_ir_2240_frame(100 + w_idx, walker_x=wx)
        obs_walk = tracker.infer(ir_walk, zones=real_zones)
        roles_w, cg_w = ident.update(obs_walk, real_zones, now=t, frame_bgr=ir_walk)
        fsm.step(obs_walk, roles_w, cg_w, real_zones, now=t)
        if w_idx >= 2:
            assert 777 in roles_w
            assert roles_w[777].role == Role.VISITOR
            assert cg_w is True


# 61. Волна 7 (v6.2): Закон сохранения Бабушки в кровати (patient_in_bed_latched) —
# переворот во сне, укрывание одеялом с головой (0 точек YOLO 600 с), пауза «Я рядом» -> IN_BED,
# персистентность в patient_anchor.json (12 ч TTL) и честный сброс при реальном уходе с матраса
def test_wave7_patient_in_bed_latched_conservation_law_blanket_pause_persistence_and_real_exit(
    tmp_path: Path,
) -> None:
    anchor_file = tmp_path / "patient_anchor.json"
    fake_wall = [1_731_445_000.0]

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

    ident = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        state_path=anchor_file,
        wall_clock=lambda: fake_wall[0],
    )
    fsm = FallStateMachine(
        sit_confirm_seconds=1.2,
        leave_confirm_seconds=0.8,
        missing_confirm_seconds=12.0,
        night_strict_mode=True,
    )

    # 1. Бабушка спит на матрасе -> взводится защёлка patient_in_bed_latched и сохраняется на диск
    t = 0.0
    for _ in range(6):
        t += 0.25
        g_sleep = make_pose(1, (0.31, 0.48, 0.63, 0.66), (0.55, 0.54), (0.39, 0.57))
        fr = make_colored_room_frame([(g_sleep, (140, 140, 140))])
        roles, cg = ident.update([g_sleep], real_zones, now=t, frame_bgr=fr)
        assert (
            fsm.step(
                [g_sleep],
                roles,
                cg,
                real_zones,
                now=t,
                patient_in_bed_latched=ident.patient_in_bed_latched,
            )
            is None
        )

    assert ident.patient_in_bed_latched is True
    assert ident.latched_patient_track_id == 1
    assert ident.latched_patient_bed_bbox == (0.31, 0.48, 0.63, 0.66)
    assert fsm.state == PatientState.IN_BED
    assert anchor_file.exists()

    # Проверяем восстановление защёлки после перезапуска контейнера и истечение через > 12 ч TTL
    fake_wall[0] += 3600.0
    ident_reloaded = PatientIdentifier(
        initial_bed_lock_seconds=5.0,
        state_path=anchor_file,
        wall_clock=lambda: fake_wall[0],
    )
    assert ident_reloaded.patient_in_bed_latched is True
    assert ident_reloaded.latched_patient_bed_bbox == (0.31, 0.48, 0.63, 0.66)
    assert ident_reloaded.latched_patient_track_id == 1

    fake_wall[0] += 13.0 * 3600.0
    ident_expired = PatientIdentifier(
        initial_bed_lock_seconds=5.0,
        state_path=anchor_file,
        wall_clock=lambda: fake_wall[0],
    )
    assert ident_expired.patient_in_bed_latched is False
    fake_wall[0] = 1_731_445_100.0

    # 2. Бабушка переворачивается на другой бок, натягивает одеяло на голову (0 точек YOLO 600 с подряд!),
    # плюс оператор нажимает «Я рядом» (пауза 30 с) и пауза истекает -> выход строго в IN_BED, 0 MISSING!
    t += 0.5
    g_turn = make_pose(1, (0.30, 0.49, 0.62, 0.68), (0.53, 0.56), (0.38, 0.59))
    fr_blanket = make_colored_room_frame([(g_turn, (215, 215, 215))])
    roles_t, cg_t = ident.update([g_turn], real_zones, now=t, frame_bgr=fr_blanket)
    fsm.step([g_turn], roles_t, cg_t, real_zones, now=t, patient_in_bed_latched=ident.patient_in_bed_latched)
    assert ident.patient_in_bed_latched is True

    fsm.acknowledge_and_pause(now=t, pause_seconds=30.0)
    assert fsm.state == PatientState.PAUSED

    for step_blanket in range(120):
        t += 5.0  # 120 * 5.0 = 600 секунд (10 минут полного отсутствия скелета под одеялом!)
        roles_b, cg_b = ident.update([], real_zones, now=t, frame_bgr=fr_blanket)
        ev_b = fsm.step(
            [],
            roles_b,
            cg_b,
            real_zones,
            now=t,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )
        assert ev_b is None, f"Ложная тревога при сне под одеялом на шаге {step_blanket} (t={t}): {ev_b}"
        assert ident.patient_in_bed_latched is True
        if t > 35.0:
            assert fsm.state == PatientState.IN_BED
            assert "одеялом" in fsm.status_reason

    # 3. Бабушка реально просыпается, встаёт с матраса на пол (patient_in_bed_latched -> False) и уходит -> MISSING!
    left_events = []
    for _ in range(6):
        t += 0.25
        g_floor = make_pose(
            1,
            (0.42, 0.52, 0.58, 0.94),
            (0.50, 0.60),
            (0.50, 0.77),
            ankle=(0.50, 0.92),
        )
        fr_f = make_colored_room_frame([(g_floor, (150, 150, 150))])
        roles_f, cg_f = ident.update([g_floor], real_zones, now=t, frame_bgr=fr_f)
        ev_f = fsm.step(
            [g_floor],
            roles_f,
            cg_f,
            real_zones,
            now=t,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )
        if ev_f is not None:
            left_events.append(ev_f)

    assert ident.patient_in_bed_latched is False
    assert any(e.state == PatientState.LEFT_BED for e in left_events)

    missing_ev = None
    for _ in range(30):
        t += 0.5
        roles_m, cg_m = ident.update([], real_zones, now=t, frame_bgr=make_colored_room_frame([]))
        ev_m = fsm.step(
            [],
            roles_m,
            cg_m,
            real_zones,
            now=t,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )
        if ev_m is not None and ev_m.state == PatientState.MISSING:
            missing_ev = ev_m
            break
    assert missing_ev is not None

    # 4. Проверяем страховочную ручную очистку clear_bed_presence()
    for _ in range(5):
        t += 0.25
        g_back = make_pose(1, (0.31, 0.48, 0.63, 0.66), (0.55, 0.54), (0.39, 0.57))
        roles_r, _ = ident.update([g_back], real_zones, now=t)
    assert ident.patient_in_bed_latched is True
    ident.clear_bed_presence()
    assert ident.patient_in_bed_latched is False
    assert ident.latched_patient_bed_bbox is None


# 62. Волна 7 (v6.2): Закон сохранения Своих через дверь —
# защита стула у двери от левой ignore_polygon, оптическое удержание сидящего спиной гостя 600 с,
# переход в слепую зону комнаты без сброса визита и честный выход через дверь за 2.5 с
def test_wave7_visitor_chair_at_door_ignore_overlap_optical_occupancy_600s_and_real_door_exit() -> None:
    assert DEFAULT_DOOR_VISIT_IDLE_TIMEOUT == 14400.0

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
            # Левая штора/шкаф из боевой разметки 192.168.1.100, частично задевающая правую половину стула у двери!
            [[0.2328, 0.0222], [0.3656, 0.0444], [0.3547, 0.9956], [0.2422, 0.9956]],
        ],
    )

    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(sit_confirm_seconds=1.2, suppress_sitting_when_caregiver=True)

    rng = np.random.default_rng(702)

    def _render_scene(
        step_n: int,
        visitor_box: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        h, w = 360, 640
        yy, xx = np.ogrid[:h, :w]
        bg = 86.0 + 14.0 * np.sin(xx / 29.0) + 11.0 * np.cos(yy / 23.0)
        # Спящая бабушка на матрасе
        bg[int(0.48 * h) : int(0.67 * h), int(0.31 * w) : int(0.63 * w)] = 148.0
        # Силуэт гостя (на стуле или в двери) с ярким контрастом одежды относительно пустого стула/двери
        if visitor_box is not None:
            vx1, vy1, vx2, vy2 = visitor_box
            bg[int(vy1 * h) : int(vy2 * h), int(vx1 * w) : int(vx2 * w)] = 212.0
        noise = rng.integers(-3, 4, size=(h, w), dtype=np.int16)
        ch = np.clip(bg.astype(np.int16) + noise + (step_n % 2), 0, 255).astype(np.uint8)
        return np.stack([ch, ch, ch], axis=-1)

    # 1. До прихода гостей: бабушка спит в кровати, стул у двери пуст (снимается эталон пустой комнаты)
    t = 0.0
    g_sleep = make_pose(1, (0.31, 0.48, 0.63, 0.66), (0.55, 0.54), (0.39, 0.57))
    for i in range(6):
        t += 0.25
        fr_empty = _render_scene(i, visitor_box=None)
        roles, cg = ident.update([g_sleep], real_zones, now=t, frame_bgr=fr_empty)
        fsm.step([g_sleep], roles, cg, real_zones, now=t, patient_in_bed_latched=ident.patient_in_bed_latched)
    assert ident.door_visit_active is False

    # 2. Родственник входит через door_polygon и садится на боевой стул у двери [0.1589, 0.6671, 0.3364, 0.9970]
    chair_box = (0.1589, 0.6671, 0.3364, 0.9970)
    vis_chair_pose = make_pose(
        77,
        chair_box,
        shoulder=(0.245, 0.725),
        hip=(0.248, 0.855),
        ankle=(0.250, 0.975),
    )
    # Сидящий на стуле у двери НЕ должен резаться левой ignore_polygon!
    assert len(filter_valid_poses([vis_chair_pose], zones=real_zones)) == 1

    t += 0.25
    v_enter_0 = make_pose(77, (0.10, 0.20, 0.24, 0.92), (0.17, 0.31), (0.17, 0.58), ankle=(0.17, 0.90))
    ident.update([g_sleep, v_enter_0], real_zones, now=t, frame_bgr=_render_scene(10, v_enter_0.bbox_norm))
    t += 0.25
    v_enter_1 = make_pose(77, (0.14, 0.22, 0.27, 0.93), (0.205, 0.33), (0.205, 0.60), ankle=(0.205, 0.91))
    roles_in, cg_in = ident.update(
        [g_sleep, v_enter_1],
        real_zones,
        now=t,
        frame_bgr=_render_scene(11, v_enter_1.bbox_norm),
    )
    assert roles_in[77].role == Role.VISITOR
    assert ident.door_visit_active is True
    assert cg_in is True

    # Родственник сел на стул у двери (виден YOLO несколько кадров на t = 2.2 .. 5.2 с)
    for k in range(12):
        t += 0.25
        fr_chair = _render_scene(20 + k, visitor_box=chair_box)
        roles_ch, cg_ch = ident.update([g_sleep, vis_chair_pose], real_zones, now=t, frame_bgr=fr_chair)
        assert roles_ch[77].role == Role.VISITOR
        assert ident.door_visit_active is True
        assert ident.visitor_spot_occupied is True
        assert ident.latched_visitor_bbox == chair_box
        assert cg_ch is True

    # 3. Родственник отвернулся спиной к камере — YOLO теряет его (0 точек гостя) на 600 секунд (10 минут!),
    # но физически он продолжает сидеть на стуле -> визит и занятость стула удерживаются БЕЗ сброса!
    for s_idx in range(120):
        t += 5.0  # 600 секунд > старых 2.5 с и 90 с!
        fr_still_on_chair = _render_scene(100 + s_idx, visitor_box=chair_box)
        roles_hold, cg_hold = ident.update([g_sleep], real_zones, now=t, frame_bgr=fr_still_on_chair)
        fsm.step(
            [g_sleep],
            roles_hold,
            cg_hold,
            real_zones,
            now=t,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )
        assert ident.door_visit_active is True, f"Ложный сброс door_visit_active на стуле (шаг {s_idx}, t={t})!"
        assert ident.visitor_spot_occupied is True
        assert ident.latched_visitor_bbox == chair_box
        assert cg_hold is True

    # 4. Родственник встал со стула и отошёл в слепую зону внутри комнаты (стул опустел, верх двери спокоен):
    # visitor_spot_occupied становится False, но door_visit_active остаётся True!
    for b_idx in range(8):
        t += 0.5
        fr_blind_room = _render_scene(300 + b_idx, visitor_box=(0.82, 0.62, 0.96, 0.95))
        roles_bl, cg_bl = ident.update([g_sleep], real_zones, now=t, frame_bgr=fr_blind_room)
        assert ident.visitor_spot_occupied is False
        assert ident.door_visit_active is True
        assert cg_bl is True

    # 5. Родственник реально выходит из комнаты через дверной проём и закрывает дверь -> через >= 2.5 с визит закрыт!
    t += 0.3
    door_exit_box = (0.10, 0.16, 0.25, 0.92)
    v_leave = make_pose(77, door_exit_box, (0.175, 0.28), (0.175, 0.56), ankle=(0.175, 0.90))
    ident.update([g_sleep, v_leave], real_zones, now=t, frame_bgr=_render_scene(400, visitor_box=door_exit_box))

    for out_idx in range(8):
        t += 0.4  # суммарно 3.2 с >= 2.5 с пустой двери и пустого стула
        fr_after_exit = _render_scene(410 + out_idx, visitor_box=None)
        roles_out, cg_out = ident.update([g_sleep], real_zones, now=t, frame_bgr=fr_after_exit)
        fsm.step(
            [g_sleep],
            roles_out,
            cg_out,
            real_zones,
            now=t,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )

    assert ident.door_visit_active is False
    assert ident.visitor_spot_occupied is False
    assert ident.latched_visitor_bbox is None
    assert cg_out is False
    assert fsm.state == PatientState.IN_BED


# 63. Волна 7 (v6.2): Строгий запрет фантомного появления «Свой» без входа через дверь после прогрева (> 15 с)
def test_wave7_strict_door_portal_guard_blocks_phantom_visitor_without_door_after_warmup() -> None:
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
    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=0.8)

    t = 0.0
    g_sleep = make_pose(1, (0.31, 0.48, 0.63, 0.66), (0.55, 0.54), (0.39, 0.57))
    for _ in range(8):
        t += 0.25
        roles, cg = ident.update([g_sleep], real_zones, now=t)
    assert roles[1].role == Role.PATIENT
    assert ident.door_visit_active is False

    # Прошло > 15 с после старта движка, дверь закрыта (door_visit_active == False):
    # случайный двигающийся силуэт на полу вдали от двери НЕ может получить Role.VISITOR и зажечь caregiver_nearby!
    t = 22.0
    for step_x in (0.54, 0.56, 0.58, 0.60, 0.62, 0.64):
        t += 0.25
        phantom_mid_room = make_pose(
            505,
            (step_x, 0.48, step_x + 0.12, 0.95),
            shoulder=(step_x + 0.06, 0.56),
            hip=(step_x + 0.06, 0.74),
            ankle=(step_x + 0.06, 0.93),
        )
        roles_ph, cg_ph = ident.update([g_sleep, phantom_mid_room], real_zones, now=t)
        assert ident.door_visit_active is False
        assert cg_ph is False
        if 505 in roles_ph:
            assert roles_ph[505].role == Role.UNKNOWN
            assert roles_ph[505].reason == "blocked_no_door_entry"

    # А честный вход через door_polygon штатно выдаёт Role.VISITOR и активирует door_visit_active!
    t += 0.5
    honest_d0 = make_pose(606, (0.10, 0.22, 0.23, 0.90), (0.165, 0.33), (0.165, 0.58), ankle=(0.165, 0.88))
    ident.update([g_sleep, honest_d0], real_zones, now=t)
    t += 0.25
    honest_d1 = make_pose(606, (0.15, 0.22, 0.28, 0.90), (0.215, 0.33), (0.215, 0.58), ankle=(0.215, 0.88))
    roles_ok, cg_ok = ident.update([g_sleep, honest_d1], real_zones, now=t)
    assert roles_ok[606].role == Role.VISITOR
    assert roles_ok[606].reason == "door_entry"
    assert ident.door_visit_active is True
    assert cg_ok is True


# 64. Волна 8 (v6.3): Чистый тревожный кадр и видео без разметки зон по умолчанию (draw_zones=False vs True)
def test_wave8_render_annotated_frame_hides_zones_by_default_and_draws_when_requested() -> None:
    zones_all = ZonesConfig(
        bed_polygon=[[0.22, 0.42], [0.68, 0.42], [0.68, 0.78], [0.22, 0.78]],
        door_polygon=[[0.04, 0.08], [0.16, 0.08], [0.16, 0.94], [0.04, 0.94]],
        floor_polygon=[[0.20, 0.80], [0.92, 0.80], [0.96, 0.98], [0.18, 0.98]],
    )
    raw_frame = np.full((360, 640, 3), 42, dtype=np.uint8)

    # 1. По умолчанию (draw_zones=False) пустой кадр в спокойном IN_BED на 100% идентичен исходнику — ни единой линии зон!
    out_default = render_annotated_frame(raw_frame, [], {}, zones_all, PatientState.IN_BED)
    out_false = render_annotated_frame(raw_frame, [], {}, zones_all, PatientState.IN_BED, draw_zones=False)
    assert np.array_equal(out_default, raw_frame)
    assert np.array_equal(out_false, raw_frame)

    # 2. При явном draw_zones=True контуры кровати, двери и пола рисуются штатно
    out_true = render_annotated_frame(raw_frame, [], {}, zones_all, PatientState.IN_BED, draw_zones=True)
    assert not np.array_equal(out_true, raw_frame)
    # Проверяем изменение пикселей конкретно на верхней границе кровати (y ~ 0.42*360 = 151) и двери (x ~ 0.16*640 = 102)
    bed_top_y = int(0.42 * 360)
    door_right_x = int(0.16 * 640)
    assert not np.array_equal(out_true[bed_top_y - 1 : bed_top_y + 2, 200:360], raw_frame[bed_top_y - 1 : bed_top_y + 2, 200:360])
    assert not np.array_equal(out_true[160:280, door_right_x - 1 : door_right_x + 2], raw_frame[160:280, door_right_x - 1 : door_right_x + 2])

    # 3. В тревожном кадре LEFT_BED с людьми (draw_zones=False по умолчанию) рисуются плашка тревоги, рамки, скелеты и бейджи,
    # а свободные участки полигонов кровати, двери и пола остаются абсолютно нетронутыми!
    pat_obs = make_pose(1, (0.72, 0.30, 0.86, 0.88), (0.79, 0.40), (0.79, 0.62), ankle=(0.79, 0.86))
    vis_obs = make_pose(2, (0.48, 0.28, 0.60, 0.84), (0.54, 0.38), (0.54, 0.58), ankle=(0.54, 0.82))
    roles = {
        1: TrackIdentity(track_id=1, role=Role.PATIENT, reason="locked_bed"),
        2: TrackIdentity(track_id=2, role=Role.VISITOR, reason="door_entry", height_cm=174.0),
    }
    out_alert_clean = render_annotated_frame(
        raw_frame,
        [pat_obs, vis_obs],
        roles,
        zones_all,
        PatientState.LEFT_BED,
        status_reason="встала с кровати",
    )
    # Плашка тревоги сверху слева и силуэты людей отрисованы
    assert not np.array_equal(out_alert_clean[12:42, 12:220], raw_frame[12:42, 12:220])
    assert not np.array_equal(out_alert_clean[int(0.30 * 360) : int(0.88 * 360), int(0.72 * 640) : int(0.86 * 640)], raw_frame[int(0.30 * 360) : int(0.88 * 360), int(0.72 * 640) : int(0.86 * 640)])
    # Левая половина кровати (x=0.24..0.40, y=0.40..0.79) и дверь (x=0.03..0.17, y=0.35..0.95) без единого пикселя зон!
    assert np.array_equal(
        out_alert_clean[int(0.40 * 360) : int(0.79 * 360), int(0.24 * 640) : int(0.40 * 640)],
        raw_frame[int(0.40 * 360) : int(0.79 * 360), int(0.24 * 640) : int(0.40 * 640)],
    )
    assert np.array_equal(
        out_alert_clean[int(0.35 * 360) : int(0.95 * 360), int(0.03 * 640) : int(0.17 * 640)],
        raw_frame[int(0.35 * 360) : int(0.95 * 360), int(0.03 * 640) : int(0.17 * 640)],
    )


# 65. Волна 8 (v6.3): Снятие ложной тревоги dismiss_false_alert без 15-минутной паузы (IDLE / IN_BED / PAUSED)
def test_wave8_dismiss_false_alert_without_15m_pause_for_latched_bed_and_idle_and_paused() -> None:
    fsm = FallStateMachine(
        sit_confirm_seconds=1.2,
        leave_confirm_seconds=1.2,
        fall_confirm_seconds=1.0,
        missing_confirm_seconds=3.0,
        alert_cooldown_seconds=10.0,
        prolonged_absence_minutes=5.0,
    )
    roles = {1: TrackIdentity(track_id=1, role=Role.PATIENT, reason="locked_bed")}

    # Сценарий A: Ложное FALLEN -> dismiss_false_alert(patient_in_bed_latched=False) переводит в IDLE без паузы!
    t = 0.0
    fallen_pose = make_pose(1, (0.35, 0.76, 0.74, 0.95), (0.42, 0.84), (0.64, 0.86), ankle=(0.72, 0.88))
    ev_fall = None
    for _ in range(8):
        t += 0.25
        ev = fsm.step([fallen_pose], roles, caregiver_nearby=False, zones=BED_ZONE, now=t)
        if ev is not None:
            ev_fall = ev
    assert ev_fall is not None and ev_fall.state == PatientState.FALLEN
    assert fsm.fall_unacknowledged is True

    fsm.dismiss_false_alert(patient_in_bed_latched=False)
    assert fsm.fall_unacknowledged is False
    assert fsm.state == PatientState.IDLE
    assert fsm.status_reason == "ложная тревога снята"
    assert fsm.pause_remaining_sec(t) == 0.0
    assert fsm.get_care_diagnostics(now=t)["away_from_bed_minutes"] == 0.0

    # После снятия ложной тревоги пустые кадры НЕ вызывают вторичный ложный MISSING,
    # но охрана остаётся боевой (НЕ на 15-минутной паузе!) и сразу ловит реальную тревогу!
    for _ in range(20):
        t += 0.5
        assert fsm.step([], {}, caregiver_nearby=False, zones=BED_ZONE, now=t) is None
    assert fsm.state == PatientState.IDLE

    t += 12.0  # прошёл cooldown 10 с, но меньше 15 минут!
    real_fall_ev = None
    for _ in range(8):
        t += 0.25
        ev2 = fsm.step([fallen_pose], roles, caregiver_nearby=False, zones=BED_ZONE, now=t)
        if ev2 is not None:
            real_fall_ev = ev2
    assert real_fall_ev is not None and real_fall_ev.state == PatientState.FALLEN

    # Сценарий B: Ложное LEFT_BED/MISSING при спящей под одеялом бабушке -> dismiss_false_alert(patient_in_bed_latched=True)
    t += 20.0
    stand_away = make_pose(1, (0.76, 0.22, 0.90, 0.88), (0.83, 0.34), (0.83, 0.58), ankle=(0.83, 0.86))
    for _ in range(8):
        t += 0.25
        fsm.step([stand_away], roles, caregiver_nearby=False, zones=BED_ZONE, now=t)
    assert fsm.state == PatientState.LEFT_BED
    assert fsm.get_care_diagnostics(now=t + 120.0)["away_from_bed_minutes"] > 0.0

    fsm.dismiss_false_alert(patient_in_bed_latched=True)
    assert fsm.state == PatientState.IN_BED
    assert fsm.status_reason == "лежит в кровати (под одеялом)"
    assert fsm.fall_unacknowledged is False
    assert fsm.pause_remaining_sec(t) == 0.0
    assert fsm.get_care_diagnostics(now=t + 120.0)["away_from_bed_minutes"] == 0.0

    # Далее при пустом YOLO и активной защёлке кровати удерживается спокойный IN_BED без MISSING
    for _ in range(20):
        t += 0.5
        ev_cov = fsm.step([], {}, caregiver_nearby=False, zones=BED_ZONE, now=t, patient_in_bed_latched=True)
        assert ev_cov is None
        assert fsm.state == PatientState.IN_BED

    # Сценарий C: Если охрана УЖЕ находилась в ручной паузе PAUSED («Я рядом»), dismiss_false_alert не сбивает паузу
    fsm.acknowledge_and_pause(now=t, pause_seconds=900.0)
    assert fsm.state == PatientState.PAUSED
    fsm.dismiss_false_alert(patient_in_bed_latched=True)
    assert fsm.state == PatientState.PAUSED
    assert fsm.pause_remaining_sec(t + 10.0) == pytest.approx(890.0, abs=1e-3)


# 66. Волна 8 (v6.3): Восстановление защёлки кровати restore_bed_presence() после ложного left_bed/missing и запись на диск
def test_wave8_restore_bed_presence_recovers_latch_and_persists_anchor_json(tmp_path: Path) -> None:
    anchor_file = tmp_path / "patient_anchor.json"
    fake_wall = [1_760_000_100.0]
    ident = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.6,
        state_path=anchor_file,
        wall_clock=lambda: fake_wall[0],
    )

    # 1. Имитируем кейс, когда стоящего у кровати родственника ошибочно приняли за PATIENT,
    # а при нажатии «Ты ошибся» перевели его в VISITOR (force_assign_visitor сбрасывает защёлку)
    # и тут же восстановили присутствие бабушки под одеялом через restore_bed_presence()
    t = 0.0
    sleep_pose = make_pose(1, (0.30, 0.44, 0.62, 0.64), (0.36, 0.52), (0.54, 0.55))
    for _ in range(5):
        t += 0.25
        ident.update([sleep_pose], BED_ZONE, now=t)
    assert ident.patient_in_bed_latched is True

    # Перевод ошибочного трека в VISITOR или сброс присутствия снимает защёлку
    ident.clear_bed_presence()
    assert ident.patient_in_bed_latched is False
    assert ident.latched_patient_bed_bbox is None

    # Также имитируем зависшую метку дверного выхода единственного человека
    ident._sole_mover_at_door_ts = 12.5

    fake_wall[0] += 45.0
    ident.restore_bed_presence()

    assert ident.patient_in_bed_latched is True
    assert ident.latched_patient_bed_bbox is not None
    bx1, by1, bx2, by2 = ident.latched_patient_bed_bbox
    assert 0.20 <= bx1 < bx2 <= 0.75
    assert 0.30 <= by1 < by2 <= 0.75
    assert ident._sole_mover_at_door_ts < 0.0
    assert anchor_file.exists()

    # 2. Перезагрузка PatientIdentifier с диска обязана подхватить восстановленную защёлку кровати!
    fake_wall[0] += 10.0
    ident_reloaded = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.6,
        state_path=anchor_file,
        wall_clock=lambda: fake_wall[0],
    )
    assert ident_reloaded.patient_in_bed_latched is True
    assert ident_reloaded.latched_patient_bed_bbox is not None


# 67. Волна 12: PerspectiveScaleField, отсев фантомов пустой кровати (clutter_roi#roi0/#roi1) и ходунков (< 105 см)
def test_wave12_perspective_scale_field_and_empty_bed_clutter_rejection() -> None:
    from app.geometry import PerspectiveScaleField, is_pose_in_ignore_zones, is_valid_human_pose

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
    zones_with_floor = ZonesConfig(
        bed_polygon=bed_poly,
        door_polygon=[[0.20659, 0.19211], [0.26885, 0.82188], [0.21947, 0.99746], [0.18297, 0.99746], [0.06847, 0.26845]],
        floor_polygon=DEFAULT_FLOOR_POLYGON_STD,
    )
    zones_no_floor = ZonesConfig(bed_polygon=bed_poly, door_polygon=[], floor_polygon=[])

    psf_a = PerspectiveScaleField.from_zones(zones_with_floor)
    psf_b = PerspectiveScaleField.from_zones(zones_no_floor)
    for v_probe in (0.44, 0.58, 0.80, 0.92):
        assert psf_a.u_per_meter(v_probe) == pytest.approx(psf_b.u_per_meter(v_probe), abs=1e-5)
    assert psf_a.u_per_meter(0.44) < psf_a.u_per_meter(0.58) < psf_a.u_per_meter(0.80) < psf_a.u_per_meter(0.92)

    # Фантом пустой кровати clutter_roi#roi0 (тёмно-синий плед + подушка, conf=0.642 >= 0.48, 15 микро-точек!)
    roi0_kps = [(0.0, 0.0, 0.0)] * 17
    roi0_kps[0] = (0.595, 0.542, 0.72)
    roi0_kps[1] = (0.598, 0.539, 0.70)
    roi0_kps[2] = (0.593, 0.540, 0.68)
    roi0_kps[5] = (0.579, 0.547, 0.66)
    roi0_kps[6] = (0.580, 0.540, 0.65)   # sh_width ~ 3.1 см, head-sh ~ 7.5 см
    roi0_kps[11] = (0.539, 0.556, 0.58)  # sh-hip ~ 18.3 см
    roi0_kps[12] = (0.540, 0.549, 0.57)
    roi0_kps[13] = (0.522, 0.564, 0.52)
    roi0_kps[14] = (0.524, 0.555, 0.50)
    roi0_kps[15] = (0.511, 0.572, 0.49)  # leg_chain ~ 23.4 см
    roi0_kps[16] = (0.513, 0.562, 0.48)
    clutter_roi0 = PoseObservation(
        track_id=9001,
        bbox_norm=(0.5081, 0.5114, 0.6430, 0.6016),
        conf=0.642,
        keypoints_norm=tuple(roi0_kps),
    )
    san_roi0 = sanitize_collapsed_bed_keypoints(clutter_roi0, bed_poly, zones=zones_with_floor)
    assert san_roi0.mid_hip is None
    assert is_valid_human_pose(clutter_roi0, min_box_conf=0.48, zones=zones_with_floor) is False
    assert is_blanket_sleeper_in_bed(clutter_roi0, bed_poly, zones=zones_with_floor) is False

    # Мелкая складка одеяла clutter_roi#roi1 (w=0.0583, h=0.0633, Diag ~ 39.5 см < 44 см)
    roi1_kps = [(0.0, 0.0, 0.0)] * 17
    roi1_kps[0] = (0.462, 0.535, 0.62)
    roi1_kps[1] = (0.465, 0.532, 0.60)
    roi1_kps[5] = (0.442, 0.544, 0.58)
    roi1_kps[6] = (0.445, 0.538, 0.56)
    clutter_roi1 = PoseObservation(
        track_id=9002,
        bbox_norm=(0.4250, 0.5120, 0.4833, 0.5753),
        conf=0.290,
        keypoints_norm=tuple(roi1_kps),
    )
    assert is_valid_human_pose(clutter_roi1, min_box_conf=0.48, zones=zones_with_floor) is False
    assert is_blanket_sleeper_in_bed(clutter_roi1, bed_poly, zones=zones_with_floor) is False

    # Проверка FloorHomography: низкий предмет высотой ~82 см (ходунки/стул) не проходит как стоящий человек (< 105 см)
    hom = FloorHomography.from_zones(zones_with_floor, bed_polygon=bed_poly)
    walker_obs = make_pose(
        track_id=77,
        bbox=(0.44, 0.68, 0.52, 0.89),
        shoulder=(0.48, 0.72),
        hip=(0.48, 0.80),
        ankle=(0.48, 0.88),
        conf=0.60,
    )
    m_walker = hom.measure_pose(walker_obs, bed_polygon=bed_poly)
    assert m_walker.height_cm is None
    assert m_walker.standing_on_floor_in_front is False

    # Самозащита кровати от ошибочного ignore_polygon поверх матраса (>= 20% площади кровати, кейс 14:07/14:09)
    bad_bed_ignore = [[[0.29177, 0.46397], [0.66743, 0.46397], [0.66743, 0.83524], [0.29177, 0.83524]]]
    real_sleeper = make_pose(1, (0.282, 0.519, 0.583, 0.648), (0.535, 0.570), (0.455, 0.580), ankle=(0.310, 0.590))
    assert (
        is_pose_in_ignore_zones(
            real_sleeper,
            bad_bed_ignore,
            bed_polygon=bed_poly,
            zones=zones_with_floor,
        )
        is False
    )


# 68. Волна 12: Скриншот пользователя (user_shot_1080#det1) — санация кукольных ног 22.3 см,
# обрезка конверта кровати по окклюдеру Свой без потери эталона и скрытие полускелета в CAREGIVER_ASSIST
def test_wave12_user_screenshot_doll_legs_sanitized_bbox_clipped_by_occluder_and_skeleton_hidden(
    tmp_path: Path,
) -> None:
    from app.geometry import bbox_iou, is_valid_human_pose

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
    zones = ZonesConfig(
        bed_polygon=bed_poly,
        door_polygon=[[0.20659, 0.19211], [0.26885, 0.82188], [0.21947, 0.99746], [0.18297, 0.99746], [0.06847, 0.26845]],
        floor_polygon=DEFAULT_FLOOR_POLYGON_STD,
    )

    # Сидящая на левом краю кровати родственница «Свой»
    vis_kps = [(0.0, 0.0, 0.0)] * 17
    vis_kps[0] = (0.388, 0.458, 0.88)
    vis_kps[5] = (0.355, 0.515, 0.92)
    vis_kps[6] = (0.378, 0.502, 0.90)
    vis_kps[7] = (0.365, 0.595, 0.86)
    vis_kps[9] = (0.428, 0.625, 0.85)  # кисти родственницы на руках бабушки
    vis_kps[10] = (0.425, 0.612, 0.84)
    vis_kps[11] = (0.345, 0.652, 0.91)
    vis_kps[12] = (0.368, 0.638, 0.89)
    vis_kps[13] = (0.402, 0.722, 0.90)
    vis_kps[14] = (0.425, 0.698, 0.88)
    vis_kps[15] = (0.375, 0.865, 0.89)
    vis_kps[16] = (0.408, 0.825, 0.87)
    visitor_obs = PoseObservation(
        track_id=2,
        bbox_norm=(0.3188, 0.4107, 0.4476, 0.9006),
        conf=0.91,
        keypoints_norm=tuple(vis_kps),
    )

    # Реконструированный скелет Бабушки со скриншота: взрослый верх + кукольные ноги 22.3 см на кистях гостя
    pat_kps = [(0.0, 0.0, 0.0)] * 17
    pat_kps[0] = (0.5449, 0.5396, 0.82)
    pat_kps[5] = (0.5370, 0.6127, 0.84)
    pat_kps[6] = (0.5290, 0.5596, 0.81)
    pat_kps[7] = (0.4991, 0.6623, 0.78)
    pat_kps[8] = (0.4985, 0.5537, 0.72)
    pat_kps[9] = (0.4507, 0.6351, 0.76)
    pat_kps[10] = (0.4799, 0.5431, 0.68)
    pat_kps[11] = (0.4726, 0.6245, 0.66)
    pat_kps[12] = (0.4653, 0.5903, 0.64)
    pat_kps[13] = (0.4500, 0.6210, 0.61)
    pat_kps[14] = (0.4347, 0.5820, 0.59)
    pat_kps[15] = (0.4254, 0.6422, 0.58)
    pat_kps[16] = (0.4288, 0.6127, 0.57)
    raw_pat_obs = PoseObservation(
        track_id=1,
        bbox_norm=(0.4100, 0.5150, 0.5805, 0.6830),
        conf=0.62,
        keypoints_norm=tuple(pat_kps),
    )

    san_pat = sanitize_collapsed_bed_keypoints(
        raw_pat_obs,
        bed_poly,
        other_observations=[visitor_obs],
        zones=zones,
    )
    assert san_pat.head_center is not None
    assert san_pat.mid_shoulder is not None
    assert san_pat.mid_hip is None
    assert san_pat.lower_limbs == []
    assert is_valid_human_pose(san_pat, min_box_conf=0.48, zones=zones) is True

    # Проверка обрезки _stabilize_bed_sleeper_bbox по окклюдеру и возврата полного конверта после ухода гостя
    tracker = PoseTracker("yolo11m-pose.pt", tmp_path, smooth_keypoints=False)
    full_sleep = make_pose(1, (0.2865, 0.4259, 0.5807, 0.7741), (0.533, 0.586), (0.420, 0.595), ankle=(0.305, 0.600))
    st_full = tracker._stabilize_bed_sleeper_bbox(full_sleep, bed_poly, now=1.0)
    assert st_full.bbox_norm[0] == pytest.approx(0.2865, abs=1e-3)

    st_occluded = tracker._stabilize_bed_sleeper_bbox(
        san_pat,
        bed_poly,
        now=2.0,
        occluders=[visitor_obs],
    )
    assert st_occluded.bbox_norm[0] >= 0.425
    assert bbox_iou(st_occluded.bbox_norm, visitor_obs.bbox_norm) < 0.06
    # Внутренний конверт кровати сохранён и мгновенно раскрывается после ухода родственницы
    assert tracker._bed_envelope_bbox is not None and tracker._bed_envelope_bbox[0] == pytest.approx(0.2865, abs=1e-3)
    st_after_leave = tracker._stabilize_bed_sleeper_bbox(san_pat, bed_poly, now=3.0, occluders=[])
    assert st_after_leave.bbox_norm[0] < 0.31

    # В режиме CAREGIVER_ASSIST укрытая бабушка без таза рисуется с чистой рамкой без обрубка скелета!
    blank_canvas = np.full((360, 640, 3), 42, dtype=np.uint8)
    roles = {
        1: TrackIdentity(track_id=1, role=Role.PATIENT, reason="bed_sleep_lock"),
        2: TrackIdentity(track_id=2, role=Role.VISITOR, reason="door_entry"),
    }
    rendered_cg = render_annotated_frame(
        frame_bgr=blank_canvas,
        observations=[st_occluded, visitor_obs],
        identities=roles,
        zones=ZonesConfig(bed_polygon=bed_poly, door_polygon=[]),
        state=PatientState.CAREGIVER_ASSIST,
    )
    # В области груди/лица бабушки (0.49..0.56 x 0.53..0.63) нет ни одного зелёного пикселя костей!
    chest_patch = rendered_cg[int(0.53 * 360) : int(0.63 * 360), int(0.49 * 640) : int(0.56 * 640)]
    assert int(np.max(np.abs(chest_patch.astype(np.int16) - 42))) == 0

    # И проверка StaticObjectGate на сидящей бабушке в 14:07 с conf=0.6197 (< 0.65), согнутыми коленями и свешенными ногами
    gate = StaticObjectGate()
    sitting_1407 = make_pose(
        track_id=11,
        bbox=(0.4031, 0.4705, 0.5023, 0.8906),
        shoulder=(0.456, 0.525),
        hip=(0.458, 0.625),
        knee=(0.445, 0.715),
        ankle=(0.442, 0.855),
        conf=0.6197,
    )
    assert is_person_on_bed(sitting_1407, bed_poly, zones=zones) is True
    live_1407 = gate.update_and_filter([sitting_1407], zones=zones, now=10.0)
    assert len(live_1407) == 1 and live_1407[0].track_id == 11

    # OpticalKeypointStabilizer маскирует оптическое движение внутри бокса гостя (exclude_boxes)
    # и мгновенно поджимает замороженный bbox бабушки при перекрытии без впуска кукольных ног
    opt_stab = OpticalKeypointStabilizer()
    rng = np.random.default_rng(12)
    fr1 = rng.integers(50, 90, size=(216, 384, 3), dtype=np.uint8)
    fr2 = fr1.copy()
    # Сильное движение пикселей ТОЛЬКО внутри бокса родственницы (x=0.32..0.44, y=0.42..0.88)
    fr2[int(0.42 * 216) : int(0.88 * 216), int(0.32 * 384) : int(0.44 * 384)] = 210
    opt_stab.stabilize([st_full], frame_bgr=fr1, zones=zones, now=1.0)
    stab_pair = opt_stab.stabilize([st_occluded, visitor_obs], frame_bgr=fr2, zones=zones, now=1.25)
    stab_pat = next(o for o in stab_pair if o.track_id == 1)
    assert stab_pat.is_still is True
    assert stab_pat.bbox_norm[0] >= 0.425
    assert stab_pat.lower_limbs == []


