from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

from app.config import Settings
from app.engine import FallGuardEngine
from app.geometry import (
    PoseObservation,
    are_legs_hanging_outside_bed,
    is_person_on_bed,
    is_standing_in_front_of_bed,
)
from app.homography import FloorHomography
from app.patient_id import (
    FaceRegistry,
    PatientIdentifier,
    Role,
    compute_torso_appearance,
)
from app.state_machine import AlertEvent, AlertSeverity, FallStateMachine, PatientState
from app.zones import ZonesConfig

# Боевая разметка камеры из 19 пакетов обратной связи (zones.json)
REAL_ZONES = ZonesConfig(
    bed_polygon=[
        [0.59805, 0.79135],
        [0.60234, 0.65649],
        [0.66389, 0.64504],
        [0.69395, 0.56997],
        [0.6689, 0.41858],
        [0.30964, 0.40458],
        [0.2574, 0.57888],
        [0.27815, 0.7888],
    ],
    door_polygon=[
        [0.20659, 0.19211],
        [0.26885, 0.82188],
        [0.21947, 0.99746],
        [0.18297, 0.99746],
        [0.06847, 0.26845],
    ],
    ignore_polygons=[
        [
            [0.71599, 0.0311],
            [0.69716, 0.60407],
            [0.81019, 0.85287],
            [0.99994, 0.91388],
            [0.99725, 0.00598],
            [0.71599, 0.0012],
        ],
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
        ],
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


def _obs(
    track_id: int,
    bbox: tuple[float, float, float, float],
    kps: list[tuple[float, float, float]],
    conf: float = 0.92,
) -> PoseObservation:
    return PoseObservation(
        track_id=track_id,
        bbox_norm=bbox,
        conf=conf,
        keypoints_norm=tuple(kps),
    )


# Точные скелеты ухаживающих из всех 12 сбойных пакетов обратной связи
CAREGIVER_FB_POSES: dict[str, PoseObservation] = {
    "022720_left_bed": _obs(
        1517,
        (0.2572, 0.2993, 0.4182, 0.9872),
        [
            (0.3616, 0.4065, 0.48),
            (0.0, 0.0, 0.0),
            (0.3579, 0.3874, 0.48),
            (0.0, 0.0, 0.0),
            (0.3423, 0.3808, 0.889),
            (0.2829, 0.4595, 0.967),
            (0.3477, 0.4699, 0.993),
            (0.273, 0.5924, 0.818),
            (0.3796, 0.6025, 0.981),
            (0.3007, 0.6661, 0.632),
            (0.3959, 0.664, 0.937),
            (0.3177, 0.6856, 0.996),
            (0.3602, 0.6978, 0.998),
            (0.3408, 0.8215, 0.989),
            (0.3796, 0.8263, 0.995),
            (0.3509, 0.9455, 0.957),
            (0.3846, 0.9526, 0.972),
        ],
    ),
    "074550_sitting_up": _obs(
        12895,
        (0.4333, 0.2904, 0.595, 0.9801),
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.557, 0.3756, 0.826),
            (0.4769, 0.4321, 0.988),
            (0.5544, 0.462, 0.995),
            (0.4475, 0.5474, 0.931),
            (0.5711, 0.6046, 0.982),
            (0.4874, 0.5918, 0.752),
            (0.5735, 0.6638, 0.913),
            (0.4818, 0.6608, 0.999),
            (0.5346, 0.6757, 0.999),
            (0.481, 0.8035, 0.995),
            (0.5286, 0.8221, 0.996),
            (0.4696, 0.9393, 0.965),
            (0.5186, 0.9489, 0.971),
        ],
    ),
    "102222_left_bed": _obs(
        230,
        (0.5107, 0.3861, 0.6209, 0.9566),
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.5673, 0.4358, 0.556),
            (0.586, 0.4422, 0.769),
            (0.5531, 0.4648, 0.98),
            (0.595, 0.4881, 0.994),
            (0.5469, 0.5499, 0.669),
            (0.6023, 0.5854, 0.927),
            (0.6034, 0.6163, 0.806),
            (0.594, 0.6099, 0.597),
            (0.5423, 0.6306, 0.995),
            (0.5683, 0.6426, 0.997),
            (0.5432, 0.7568, 0.975),
            (0.5644, 0.7884, 0.987),
            (0.5388, 0.8673, 0.914),
            (0.5531, 0.9135, 0.943),
        ],
    ),
    "102227_sitting_up": _obs(
        230,
        (0.4485, 0.4914, 0.6399, 0.991),
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.6068, 0.564, 0.793),
            (0.5388, 0.5716, 0.968),
            (0.599, 0.6351, 0.988),
            (0.5073, 0.6486, 0.567),
            (0.5934, 0.7537, 0.772),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.4883, 0.7737, 0.954),
            (0.5311, 0.8015, 0.967),
            (0.4827, 0.8872, 0.614),
            (0.5421, 0.9382, 0.675),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ],
    ),
    "102309_left_bed": _obs(
        230,
        (0.4526, 0.305, 0.5811, 0.8918),
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.4943, 0.3681, 0.826),
            (0.0, 0.0, 0.0),
            (0.4989, 0.4312, 0.997),
            (0.55, 0.4007, 0.965),
            (0.478, 0.5256, 0.962),
            (0.0, 0.0, 0.0),
            (0.465, 0.4886, 0.637),
            (0.0, 0.0, 0.0),
            (0.5217, 0.6219, 0.999),
            (0.5562, 0.6007, 0.997),
            (0.5144, 0.7536, 0.998),
            (0.5361, 0.7092, 0.991),
            (0.5208, 0.8714, 0.991),
            (0.5315, 0.8087, 0.977),
        ],
    ),
    "102313_sitting_up": _obs(
        230,
        (0.4382, 0.2898, 0.5751, 0.8909),
        [
            (0.5111, 0.3469, 0.48),
            (0.5152, 0.3397, 0.588),
            (0.0, 0.0, 0.0),
            (0.5249, 0.3526, 0.944),
            (0.0, 0.0, 0.0),
            (0.5173, 0.4186, 0.997),
            (0.5626, 0.3927, 0.937),
            (0.466, 0.446, 0.989),
            (0.5293, 0.4169, 0.562),
            (0.446, 0.4261, 0.937),
            (0.0, 0.0, 0.0),
            (0.5239, 0.6128, 0.998),
            (0.5525, 0.5922, 0.994),
            (0.5137, 0.7443, 0.994),
            (0.5346, 0.6877, 0.981),
            (0.5179, 0.87, 0.968),
            (0.5341, 0.7909, 0.941),
        ],
    ),
    "102630_fallen_helper": _obs(
        254,
        (0.485, 0.4952, 0.6073, 0.8874),
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.5171, 0.5748, 0.773),
            (0.0, 0.0, 0.0),
            (0.5214, 0.6583, 0.993),
            (0.5822, 0.6197, 0.976),
            (0.5107, 0.772, 0.914),
            (0.5998, 0.6998, 0.549),
            (0.4998, 0.7192, 0.48),
            (0.0, 0.0, 0.0),
            (0.5407, 0.8298, 0.955),
            (0.5824, 0.8065, 0.93),
            (0.4865, 0.736, 0.697),
            (0.5292, 0.7063, 0.523),
            (0.505, 0.8596, 0.48),
            (0.0, 0.0, 0.0),
        ],
    ),
    "103826_sitting_up": _obs(
        347,
        (0.4061, 0.5178, 0.6039, 0.9541),
        [
            (0.5709, 0.6395, 0.82),
            (0.0, 0.0, 0.0),
            (0.5675, 0.6281, 0.911),
            (0.0, 0.0, 0.0),
            (0.5496, 0.601, 0.933),
            (0.5499, 0.5749, 0.963),
            (0.5058, 0.6108, 0.998),
            (0.5314, 0.6432, 0.66),
            (0.488, 0.7216, 0.998),
            (0.5252, 0.6892, 0.672),
            (0.5169, 0.7757, 0.991),
            (0.4707, 0.6435, 0.996),
            (0.4378, 0.6643, 0.999),
            (0.49, 0.7502, 0.984),
            (0.4553, 0.8038, 0.997),
            (0.4911, 0.8162, 0.954),
            (0.4508, 0.8983, 0.982),
        ],
    ),
    "103839_left_bed": _obs(
        347,
        (0.1525, 0.3788, 0.3279, 0.9949),
        [
            (0.2172, 0.4951, 0.969),
            (0.2247, 0.4799, 0.961),
            (0.212, 0.4789, 0.666),
            (0.2518, 0.458, 0.948),
            (0.0, 0.0, 0.0),
            (0.28, 0.5247, 0.998),
            (0.2161, 0.5163, 0.99),
            (0.3165, 0.6357, 0.995),
            (0.1944, 0.6141, 0.96),
            (0.2805, 0.7223, 0.988),
            (0.1711, 0.7162, 0.949),
            (0.2755, 0.7303, 0.999),
            (0.2351, 0.7241, 0.998),
            (0.2886, 0.8631, 0.993),
            (0.2482, 0.838, 0.988),
            (0.3103, 0.9595, 0.928),
            (0.2793, 0.9327, 0.912),
        ],
    ),
    "104920_sitting_up": _obs(
        455,
        (0.3084, 0.313, 0.4745, 0.891),
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.3606, 0.3599, 0.48),
            (0.3904, 0.3584, 0.758),
            (0.3451, 0.4067, 0.965),
            (0.4, 0.4077, 0.996),
            (0.3336, 0.4868, 0.62),
            (0.4191, 0.4885, 0.987),
            (0.3429, 0.5054, 0.626),
            (0.4271, 0.5431, 0.907),
            (0.359, 0.5825, 0.998),
            (0.3972, 0.5871, 0.999),
            (0.3777, 0.7093, 0.997),
            (0.4224, 0.7221, 0.999),
            (0.385, 0.8219, 0.992),
            (0.4348, 0.8476, 0.997),
        ],
    ),
    "110430_sitting_up": _obs(
        568,
        (0.2481, 0.4748, 0.4926, 0.9176),
        [
            (0.2978, 0.5499, 0.947),
            (0.2974, 0.5351, 0.53),
            (0.2906, 0.5417, 0.957),
            (0.0, 0.0, 0.0),
            (0.2699, 0.5456, 0.912),
            (0.2871, 0.5732, 0.871),
            (0.2625, 0.6195, 0.989),
            (0.3221, 0.6216, 0.742),
            (0.2813, 0.7213, 0.993),
            (0.3453, 0.6193, 0.859),
            (0.3242, 0.6757, 0.996),
            (0.3297, 0.7173, 0.979),
            (0.3163, 0.7553, 0.994),
            (0.4072, 0.7379, 0.977),
            (0.3965, 0.7897, 0.993),
            (0.4445, 0.8403, 0.952),
            (0.4426, 0.8843, 0.975),
        ],
    ),
    "110545_left_bed": _obs(
        568,
        (0.2472, 0.4707, 0.4973, 0.9189),
        [
            (0.2978, 0.546, 0.956),
            (0.2983, 0.5314, 0.602),
            (0.2906, 0.5371, 0.953),
            (0.0, 0.0, 0.0),
            (0.2719, 0.539, 0.892),
            (0.2909, 0.5665, 0.937),
            (0.2618, 0.6141, 0.99),
            (0.333, 0.5725, 0.889),
            (0.2739, 0.7228, 0.993),
            (0.364, 0.558, 0.941),
            (0.3208, 0.7269, 0.996),
            (0.328, 0.7205, 0.987),
            (0.3125, 0.757, 0.994),
            (0.4068, 0.7425, 0.979),
            (0.3955, 0.7865, 0.991),
            (0.446, 0.8518, 0.942),
            (0.4435, 0.8841, 0.963),
        ],
    ),
}

# Реальная бабушка, сидящая на кровати в 102630_fallen_f1ac (tid=262)
GRANDMA_102630_OBS = _obs(
    262,
    (0.4033, 0.4352, 0.512, 0.9176),
    [
        (0.4446, 0.506, 0.976),
        (0.452, 0.5003, 0.978),
        (0.4449, 0.4949, 0.485),
        (0.4743, 0.5027, 0.979),
        (0.0, 0.0, 0.0),
        (0.4941, 0.545, 0.997),
        (0.4375, 0.5212, 0.998),
        (0.492, 0.6335, 0.993),
        (0.4193, 0.5847, 0.998),
        (0.4599, 0.6873, 0.994),
        (0.4156, 0.6464, 0.998),
        (0.476, 0.6683, 1.0),
        (0.437, 0.6544, 1.0),
        (0.4746, 0.7547, 0.999),
        (0.4258, 0.74, 0.999),
        (0.4719, 0.8628, 0.99),
        (0.4295, 0.8546, 0.993),
    ],
)

# Горизонтальный сон бабушки из всех 7 ночных пакетов 233638..234247 (angle=69.7°)
GRANDMA_SLEEP_233638_OBS = _obs(
    1,
    (0.298, 0.5094, 0.6074, 0.7101),
    [
        (0.5696, 0.5432, 0.976),
        (0.5797, 0.549, 0.962),
        (0.5711, 0.5298, 0.92),
        (0.5876, 0.5738, 0.858),
        (0.5653, 0.5327, 0.6),
        (0.5712, 0.6109, 0.992),
        (0.5476, 0.5597, 0.971),
        (0.5633, 0.652, 0.935),
        (0.5097, 0.5563, 0.777),
        (0.5465, 0.6393, 0.887),
        (0.4964, 0.5518, 0.693),
        (0.491, 0.6524, 0.982),
        (0.4802, 0.6153, 0.971),
        (0.4219, 0.65, 0.96),
        (0.4137, 0.6134, 0.94),
        (0.3456, 0.6717, 0.918),
        (0.3465, 0.6478, 0.893),
    ],
)


def _jitter_obs(obs: PoseObservation, dx: float, dy: float = 0.0, track_id: int | None = None) -> PoseObservation:
    bx1, by1, bx2, by2 = obs.bbox_norm
    kps = [
        (round(kx + dx, 5), round(ky + dy, 5), kc) if kc > 0.0 else (0.0, 0.0, 0.0)
        for kx, ky, kc in obs.keypoints_norm
    ]
    return PoseObservation(
        track_id=obs.track_id if track_id is None else track_id,
        bbox_norm=(round(bx1 + dx, 5), round(by1 + dy, 5), round(bx2 + dx, 5), round(by2 + dy, 5)),
        conf=obs.conf,
        keypoints_norm=tuple(kps),
    )


def _build_engine(tmp_path: Path) -> FallGuardEngine:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        rtsp_url="",
        tg_bot_token="123456:TEST_WAVE10_TOKEN",
        tg_chat_ids=("100001",),
        data_dir=data_dir,
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
        bed_lock_seconds=12.0,
        initial_bed_lock_seconds=1.5,
        sit_confirm_seconds=1.6,
        leave_confirm_seconds=1.8,
        fall_confirm_seconds=1.6,
        missing_confirm_seconds=10.0,
        alert_cooldown_seconds=30.0,
        repeat_fall_alert_seconds=15.0,
        max_fall_repeats=3,
        suppress_sitting_when_caregiver=True,
        face_match_threshold=0.38,
        tg_bot_username="NoFallenAi",
        frame_rotate=0,
        frame_flip_h=False,
        low_light_enhance=False,
    )
    engine = FallGuardEngine(settings)

    def _forbid_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Сетевой вызов в тесте запрещён: {request.method} {request.url}")

    engine.notifier._transport = httpx.MockTransport(_forbid_network)
    return engine


def test_wave10_all_12_caregiver_feedback_poses_standing_in_front_and_never_on_bed() -> None:
    """
    Волна 10 (#161 + #162): все 12 сбойных поз ухаживающих (стоящих в полный рост,
    наклонившихся к подушке/над стулом 102227 и 103826, стоящих у двери 103839 и сидящих
    на стуле 102630/110430/110545) распознаются строго ВНЕ кровати и без свешивания ног.
    При этом настоящая бабушка (102630 tid=262 и сон 233638..234247) остаётся на кровати.
    """
    hom = FloorHomography.from_zones(REAL_ZONES, bed_polygon=REAL_ZONES.bed_polygon)
    assert hom.calibrated is True
    assert hom.is_valid is True

    tall_standing_cases = {
        "022720_left_bed",
        "074550_sitting_up",
        "102222_left_bed",
        "102309_left_bed",
        "102313_sitting_up",
        "104920_sitting_up",
    }

    for case_name, obs in CAREGIVER_FB_POSES.items():
        st_front = is_standing_in_front_of_bed(obs, REAL_ZONES.bed_polygon, zones=REAL_ZONES)
        on_bed = is_person_on_bed(obs, REAL_ZONES.bed_polygon, zones=REAL_ZONES)
        legs_out = are_legs_hanging_outside_bed(obs, REAL_ZONES.bed_polygon, zones=REAL_ZONES)
        assert st_front is True, f"{case_name}: ожидался is_standing_in_front_of_bed=True"
        assert on_bed is False, f"{case_name}: ожидался is_person_on_bed=False"
        assert legs_out is False, f"{case_name}: ожидался are_legs_hanging_outside_bed=False"

        if case_name in tall_standing_cases:
            m = hom.measure_pose(obs, bed_polygon=REAL_ZONES.bed_polygon)
            assert m.height_cm is not None and m.height_cm >= 175.0, f"{case_name}: рост={m.height_cm}"
            assert m.standing_on_floor_in_front is True, f"{case_name}: ожидался standing_on_floor_in_front=True"

    # Настоящая бабушка в 102630 (tid=262) и во сне 233638..234247 — строго на кровати!
    assert is_standing_in_front_of_bed(GRANDMA_102630_OBS, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False
    assert is_person_on_bed(GRANDMA_102630_OBS, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    assert is_standing_in_front_of_bed(GRANDMA_SLEEP_233638_OBS, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False
    assert is_person_on_bed(GRANDMA_SLEEP_233638_OBS, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True
    assert are_legs_hanging_outside_bed(GRANDMA_SLEEP_233638_OBS, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False


def test_wave10_face_registry_back_turned_spatial_ownership_and_strict_outside_bed_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Волна 10 (#161 Причина #1 + #162 п.1):
    1) Гейт затылка блокирует матч на 022720, 074550, 102222, 102309, 102630(tid=254), 104920;
    2) Обратная проекция 0°/±90° и пространственное владение отвергают чужие лица на подушке и у соседнего трека;
    3) Вне кровати Role.PATIENT выдаётся ТОЛЬКО при уверенном анфасе (оба глаза и нос >= 0.45),
       ширине лица >= 48 px, best_patient >= 0.54 (0.68 у двери) и марже >= +0.08 над visitor!
    """
    for back_case in (
        "022720_left_bed",
        "074550_sitting_up",
        "102222_left_bed",
        "102309_left_bed",
        "102630_fallen_helper",
        "104920_sitting_up",
    ):
        assert FaceRegistry._is_back_turned_posture(CAREGIVER_FB_POSES[back_case]) is True, back_case

    assert FaceRegistry._is_back_turned_posture(GRANDMA_102630_OBS) is False
    assert FaceRegistry._is_back_turned_posture(GRANDMA_SLEEP_233638_OBS) is False

    # Проверка точной обратной проекции координат центра лица из повёрнутых кропов
    frame_shape = (500, 1000)   # (H, W)
    crop_shape = (200, 200)     # (Hc, Wc)
    crop_origin = (200, 100)    # (x1, y1); центр (u=100, v=100) -> глобально (0.30, 0.40)
    assert FaceRegistry._map_face_center_to_frame_norm(
        (80, 80, 40, 40), None, crop_shape, crop_origin, frame_shape
    ) == pytest.approx((0.30, 0.40), abs=1e-4)
    # Точка в верхнем левом углу исходного кропа (u=20, v=40) -> глобально (0.22, 0.28):
    # при повороте +90° (CW): u_rot = Hc - v = 160, v_rot = u = 20
    fc_cw = FaceRegistry._map_face_center_to_frame_norm(
        (150, 10, 20, 20), cv2.ROTATE_90_CLOCKWISE, crop_shape, crop_origin, frame_shape
    )
    assert fc_cw == pytest.approx((0.22, 0.28), abs=2e-3)
    # при повороте -90° (CCW): u_rot = v = 40, v_rot = Wc - u = 180
    fc_ccw = FaceRegistry._map_face_center_to_frame_norm(
        (30, 170, 20, 20), cv2.ROTATE_90_COUNTERCLOCKWISE, crop_shape, crop_origin, frame_shape
    )
    assert fc_ccw == pytest.approx((0.22, 0.28), abs=2e-3)

    # Перекрёстное владение: лицо бабушки tid=262 (0.453, 0.501) отвергается для помощницы tid=254
    helper_254 = CAREGIVER_FB_POSES["102630_fallen_helper"]
    assert (
        FaceRegistry._is_face_spatially_owned(
            fcx=0.453,
            fcy=0.501,
            obs=helper_254,
            allow_rot=False,
            other_observations=[GRANDMA_102630_OBS],
            zones=REAL_ZONES,
            latched_patient_bed_bbox=None,
        )
        is False
    )
    # Лицо на подушке внутри latched_patient_bed_bbox отвергается для стоящего на полу трека
    stander_out = CAREGIVER_FB_POSES["110545_left_bed"]
    assert (
        FaceRegistry._is_face_spatially_owned(
            fcx=0.56,
            fcy=0.54,
            obs=stander_out,
            allow_rot=False,
            other_observations=[],
            zones=REAL_ZONES,
            latched_patient_bed_bbox=(0.298, 0.509, 0.607, 0.710),
        )
        is False
    )

    # Проверяем строгие пороги match_person_crop на синтетическом банке эмбеддингов
    reg = FaceRegistry(faces_dir=tmp_path / "faces", models_dir=tmp_path / "models", threshold=0.38)
    pat_vec = np.zeros(128, dtype=np.float32)
    pat_vec[0] = 1.0
    vis_vec = np.zeros(128, dtype=np.float32)
    vis_vec[1] = 1.0
    reg._models_ready = True
    reg._embeddings[Role.PATIENT.value] = {"pat_clean.jpg": pat_vec}
    reg._embeddings[Role.VISITOR.value] = {"vis_clean.jpg": vis_vec}

    state_box: dict[str, Any] = {
        "rect_w": 38,
        "sim_pat": 0.49,
        "sim_vis": 0.44,
    }

    def _fake_detect_rot(
        crop_bgr: np.ndarray,
        allow_rotations: bool = True,
    ) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, int | None]:
        ch, cw = crop_bgr.shape[:2]
        rw = int(state_box["rect_w"])
        sp = float(state_box["sim_pat"])
        sv = float(state_box["sim_vis"])
        rem_sq = max(0.0, 1.0 - sp * sp - sv * sv)
        feat = np.zeros(128, dtype=np.float32)
        feat[0] = sp
        feat[1] = sv
        feat[2] = float(np.sqrt(rem_sq))
        rect = (int((cw - rw) * 0.5), int((ch - rw) * 0.5), rw, rw)
        return feat, rect, crop_bgr, None

    monkeypatch.setattr(reg, "_detect_face_with_rot_code", _fake_detect_rot)
    frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)

    # 1) На затылке 074550 и 102630_helper гейт срабатывает ДО детекции лиц -> Role.UNKNOWN
    assert reg.match_person_crop(frame, CAREGIVER_FB_POSES["074550_sitting_up"], zones=REAL_ZONES) == (
        Role.UNKNOWN,
        0.0,
    )

    # 2) Вне кровати на стуле 110545: мелкое лицо 38 px и sim_pat=0.49 -> отвергается (НЕ Role.PATIENT!)
    res_small, _ = reg.match_person_crop(frame, CAREGIVER_FB_POSES["110545_left_bed"], zones=REAL_ZONES)
    assert res_small == Role.UNKNOWN

    frontal_outside_free = _obs(
        570,
        (0.38, 0.28, 0.52, 0.88),
        [
            (0.450, 0.340, 0.95),  # нос >= 0.45 (строго выше bed_polygon и правее door_polygon!)
            (0.458, 0.332, 0.92),  # левый глаз >= 0.45
            (0.442, 0.332, 0.92),  # правый глаз >= 0.45
            (0.468, 0.336, 0.85),
            (0.432, 0.336, 0.85),
            (0.480, 0.420, 0.96),
            (0.420, 0.420, 0.96),
            (0.490, 0.520, 0.90),
            (0.410, 0.520, 0.90),
            (0.485, 0.600, 0.90),
            (0.415, 0.600, 0.90),
            (0.470, 0.620, 0.96),
            (0.430, 0.620, 0.96),
            (0.470, 0.750, 0.95),
            (0.430, 0.750, 0.95),
            (0.470, 0.865, 0.95),
            (0.430, 0.865, 0.95),
        ],
    )

    # 3) Даже при крупном лице 64 px и высоком sim_pat=0.62 вне кровати боковой профиль без левого глаза (<0.45) отвергается!
    profile_outside_free = _obs(
        569,
        frontal_outside_free.bbox_norm,
        [
            (0.450, 0.340, 0.95),  # нос виден
            (0.458, 0.332, 0.20),  # левый глаз < 0.45 (боковой профиль!)
            (0.442, 0.332, 0.92),  # правый глаз виден
            *frontal_outside_free.keypoints_norm[3:],
        ],
    )
    state_box.update({"rect_w": 64, "sim_pat": 0.62, "sim_vis": 0.20})
    res_prof, _ = reg.match_person_crop(frame, profile_outside_free, zones=REAL_ZONES)
    assert res_prof == Role.UNKNOWN

    # 4) Строгий анфас вне кровати (оба глаза и нос >= 0.45, face_w=64 >= 48), но малая маржа (0.58 vs 0.52 = +0.06 < +0.08) -> отвергается!
    state_box.update({"rect_w": 64, "sim_pat": 0.58, "sim_vis": 0.52})
    res_margin, _ = reg.match_person_crop(frame, frontal_outside_free, zones=REAL_ZONES)
    assert res_margin == Role.UNKNOWN

    # 4б) В зоне двери (103839_left_bed) и при наложении головы сидящего на стуле 110545 на bed_polygon даже sim_pat=0.62 отвергается!
    state_box.update({"rect_w": 64, "sim_pat": 0.62, "sim_vis": 0.20})
    res_door_low, _ = reg.match_person_crop(frame, CAREGIVER_FB_POSES["103839_left_bed"], zones=REAL_ZONES)
    assert res_door_low == Role.UNKNOWN
    res_chair_bed_bg, _ = reg.match_person_crop(frame, CAREGIVER_FB_POSES["110545_left_bed"], zones=REAL_ZONES)
    assert res_chair_bed_bg == Role.UNKNOWN

    # 5) Строгий анфас крупным планом вне кровати и вне полигона матраса/двери
    # (sim_pat=0.62 >= 0.54, маржа +0.42 >= +0.08, face_w=64 >= 48) -> разрешён!
    role_ok_out, score_ok_out = reg.match_person_crop(frame, frontal_outside_free, zones=REAL_ZONES)
    assert role_ok_out == Role.PATIENT
    assert score_ok_out == pytest.approx(0.62, abs=1e-2)

    # 6) На кровати (GRANDMA_SLEEP_233638_OBS): достаточно best_patient=0.46 >= 0.42 и маржи +0.16 >= +0.04
    state_box.update({"rect_w": 36, "sim_pat": 0.46, "sim_vis": 0.30})
    role_ok_bed, score_ok_bed = reg.match_person_crop(frame, GRANDMA_SLEEP_233638_OBS, zones=REAL_ZONES)
    assert role_ok_bed == Role.PATIENT
    assert score_ok_bed == pytest.approx(0.46, abs=1e-2)


def test_wave10_case_102630_co_present_caregiver_keeps_grandma_patient_and_suppresses_fall_alert() -> None:
    """
    Волна 10 (#161 Причина #1 + Причина #3 — инцидент 102630_fallen_f1ac):
    При одновременном присутствии сидящей на кровати бабушки (tid=262) и помощницы на стуле
    спиной к камере (tid=254) бабушка сохраняет Role.PATIENT, помощница — Role.VISITOR,
    caregiver_nearby=True переводит FSM в CAREGIVER_ASSIST ДО проверки падения/сползания (0 FALLEN!).
    """
    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(
        sit_confirm_seconds=1.4,
        leave_confirm_seconds=1.6,
        fall_confirm_seconds=1.4,
        suppress_sitting_when_caregiver=True,
    )

    t = 0.0
    for _ in range(6):
        t += 0.25
        roles, cg = ident.update([GRANDMA_SLEEP_233638_OBS], REAL_ZONES, now=t)
        assert fsm.step([GRANDMA_SLEEP_233638_OBS], roles, cg, REAL_ZONES, now=t) is None
    assert roles[1].role == Role.PATIENT

    # Входит помощница через дверь (tid=254)
    d0 = _obs(
        254,
        (0.11, 0.26, 0.23, 0.92),
        [
            (0.17, 0.32, 0.9),
            (0.16, 0.31, 0.9),
            (0.18, 0.31, 0.9),
            (0.15, 0.32, 0.8),
            (0.19, 0.32, 0.8),
            (0.14, 0.40, 0.95),
            (0.20, 0.40, 0.95),
            (0.13, 0.52, 0.9),
            (0.21, 0.52, 0.9),
            (0.14, 0.62, 0.9),
            (0.20, 0.62, 0.9),
            (0.15, 0.64, 0.95),
            (0.19, 0.64, 0.95),
            (0.15, 0.77, 0.95),
            (0.19, 0.77, 0.95),
            (0.15, 0.90, 0.95),
            (0.19, 0.90, 0.95),
        ],
    )
    for dx in (0.0, 0.05, 0.10):
        t += 0.25
        roles, cg = ident.update([GRANDMA_SLEEP_233638_OBS, _jitter_obs(d0, dx)], REAL_ZONES, now=t)
    assert roles[254].role == Role.VISITOR
    assert ident.door_visit_active is True

    # Сцена 102630: бабушка села с новым ID tid=262, помощница сидит перед ней на стуле спиной tid=254
    helper_254 = CAREGIVER_FB_POSES["102630_fallen_helper"]
    for idx in range(25):
        t += 0.3
        batch = [
            _jitter_obs(helper_254, 0.003 * (idx % 2)),
            _jitter_obs(GRANDMA_102630_OBS, 0.002 * (idx % 2)),
        ]
        roles, cg = ident.update(batch, REAL_ZONES, now=t)
        ev = fsm.step(batch, roles, cg, REAL_ZONES, now=t, patient_in_bed_latched=ident.patient_in_bed_latched)
        assert ev is None, f"Ложная тревога при уходе помощницы на шаге {idx}: {ev}"
        assert cg is True
        assert fsm.state == PatientState.CAREGIVER_ASSIST
        if 262 in roles and roles[262].role != Role.UNKNOWN:
            assert roles[262].role == Role.PATIENT
        assert roles[254].role == Role.VISITOR

    # Даже если при активном caregiver_nearby поза пациента выглядит низко сползшей — тревога подавлена!
    slumped_pat = _obs(
        262,
        (0.35, 0.74, 0.72, 0.95),
        [
            (0.40, 0.82, 0.9),
            (0.41, 0.81, 0.9),
            (0.39, 0.81, 0.9),
            (0.42, 0.82, 0.8),
            (0.38, 0.82, 0.8),
            (0.44, 0.84, 0.95),
            (0.44, 0.86, 0.95),
            (0.50, 0.86, 0.9),
            (0.50, 0.88, 0.9),
            (0.54, 0.88, 0.9),
            (0.54, 0.90, 0.9),
            (0.60, 0.86, 0.95),
            (0.60, 0.88, 0.95),
            (0.66, 0.88, 0.95),
            (0.66, 0.90, 0.95),
            (0.70, 0.89, 0.95),
            (0.70, 0.91, 0.95),
        ],
    )
    for _ in range(10):
        t += 0.3
        ev_suppressed = fsm.step(
            [slumped_pat, helper_254],
            {262: roles[262], 254: roles[254]},
            caregiver_nearby=True,
            zones=REAL_ZONES,
            now=t,
        )
        assert ev_suppressed is None
        assert fsm.state == PatientState.CAREGIVER_ASSIST

    # Но как только помощница ушла (caregiver_nearby=False), реальное падение бабушки ловится штатно!
    fall_ev = None
    for _ in range(8):
        t += 0.3
        ev_alone = fsm.step(
            [slumped_pat],
            {262: roles[262]},
            caregiver_nearby=False,
            zones=REAL_ZONES,
            now=t,
        )
        if ev_alone is not None:
            fall_ev = ev_alone
    assert fall_ev is not None and fall_ev.state == PatientState.FALLEN


def test_wave10_covered_grandma_and_companion_103826_103839_104920_110430_110545_never_alert() -> None:
    """
    Волна 10 (#162 п.2 и п.3 — цепочка 10:38:26 -> 10:38:39 -> 10:49:20 -> 11:04:30 -> 11:05:45):
    Бабушка спит под одеялом (patient_in_bed_latched=True, 0 скелетов на кровати).
    Родственница наклоняется над стулом (103826), подходит с тарелкой к двери (103839),
    НЕ выходит, а возвращается к кровати (104920) и садится читать книгу на стуле (110430, 110545)
    со сменой track_id (347 -> 455 -> 568).
    Инварианты:
    - Родственница всегда Role.VISITOR (никогда не Role.PATIENT и не сбрасывает защёлку кровати);
    - door_visit_active НЕ закрывается, пока в комнате вне кровати виден живой человек;
    - _visitor_exiting_at_door сбрасывается при возврате от двери в комнату;
    - caregiver_nearby=True, 0 ложных sitting_up / left_bed / missing!
    """
    ident = PatientIdentifier(
        bed_lock_seconds=10.0,
        initial_bed_lock_seconds=0.8,
        door_visit_idle_timeout_sec=8.0,
    )
    fsm = FallStateMachine(
        sit_confirm_seconds=1.5,
        leave_confirm_seconds=1.5,
        missing_confirm_seconds=5.0,
        suppress_sitting_when_caregiver=True,
    )

    # 1. Бабушка уснула на кровати и защёлкнута (patient_in_bed_latched=True)
    t = 0.0
    for _ in range(6):
        t += 0.25
        roles, cg = ident.update([GRANDMA_SLEEP_233638_OBS], REAL_ZONES, now=t)
        assert (
            fsm.step(
                [GRANDMA_SLEEP_233638_OBS],
                roles,
                cg,
                REAL_ZONES,
                now=t,
                patient_in_bed_latched=ident.patient_in_bed_latched,
            )
            is None
        )
    assert ident.patient_in_bed_latched is True

    # 2. Бабушка укрылась с головой (скелет пропал), а через 25 с родственница проходит по комнате:
    # 103826 (наклон 65°) -> 103839 (у двери с тарелкой!) -> 104920 (у изголовья) -> 110430 / 110545 (на стуле)
    t = 25.0
    sequence = [
        ("103826_sitting_up", CAREGIVER_FB_POSES["103826_sitting_up"]),
        ("103839_left_bed", CAREGIVER_FB_POSES["103839_left_bed"]),
        ("104920_sitting_up", CAREGIVER_FB_POSES["104920_sitting_up"]),
        ("110430_sitting_up", CAREGIVER_FB_POSES["110430_sitting_up"]),
        ("110545_left_bed", CAREGIVER_FB_POSES["110545_left_bed"]),
    ]

    for stage_name, base_pose in sequence:
        # Имитируем живые микродвижения трека на 10 кадрах (включая паузы > 3.5 с на стуле после подхода к двери!)
        for step_i in range(10):
            t += 0.45
            dx = 0.025 * (step_i % 3) if step_i < 4 else 0.004 * (step_i % 2)
            cur_pose = _jitter_obs(base_pose, dx)
            roles_cur, cg_cur = ident.update([cur_pose], REAL_ZONES, now=t)
            ev = fsm.step(
                [cur_pose],
                roles_cur,
                cg_cur,
                REAL_ZONES,
                now=t,
                patient_in_bed_latched=ident.patient_in_bed_latched,
            )
            assert ev is None, f"Ложная тревога на стадии {stage_name} (шаг {step_i}): {ev}"
            assert ident.patient_in_bed_latched is True, f"Сброшена защёлка кровати на {stage_name}!"
            if base_pose.track_id in roles_cur:
                tr_cur = roles_cur[base_pose.track_id]
                assert tr_cur.role == Role.VISITOR, f"{stage_name}: неверная роль {tr_cur.role} ({tr_cur.reason})"
                assert ident.door_visit_active is True, f"{stage_name}: закрылся door_visit_active при человеке в комнате!"
                assert cg_cur is True

        if stage_name in ("104920_sitting_up", "110430_sitting_up", "110545_left_bed"):
            # После возврата от двери вглубь комнаты флаг выхода у двери обязан быть сброшен!
            assert ident._visitor_exiting_at_door is False, stage_name

    # 3. Теперь родственница реально встала, вышла через дверной проём и покинула кадр на >= 3.2 с
    t += 0.4
    exit_door_pose = _jitter_obs(CAREGIVER_FB_POSES["103839_left_bed"], -0.02, track_id=568)
    ident.update([exit_door_pose], REAL_ZONES, now=t)
    for _ in range(9):
        t += 0.4
        roles_empty, cg_empty = ident.update([], REAL_ZONES, now=t)
        ev_empty = fsm.step(
            [],
            roles_empty,
            cg_empty,
            REAL_ZONES,
            now=t,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )
        assert ev_empty is None
        assert fsm.state in (PatientState.CAREGIVER_ASSIST, PatientState.IN_BED)

    assert ident.door_visit_active is False
    assert ident.patient_in_bed_latched is True
    assert fsm.state == PatientState.IN_BED


def test_wave10_horizontal_sleep_233638_to_234247_never_triggers_legs_hanging_or_sitting_up() -> None:
    """
    Волна 10 (#161 Причина #4 — 7 ночных пакетов 233638..234247):
    Горизонтально спящая на боку бабушка (angle=69.7°, bh=0.2007) с лодыжкой у края узкого
    полигона кровати никогда не даёт ложного свешивания ног, sitting_up или left_bed
    даже в ночном строгом режиме (23:40).
    """
    ident = PatientIdentifier(bed_lock_seconds=10.0, initial_bed_lock_seconds=0.8)
    fsm = FallStateMachine(
        sit_confirm_seconds=1.5,
        leave_confirm_seconds=1.6,
        night_strict_mode=True,
        night_start_hour=22,
        night_end_hour=7,
    )

    t = 0.0
    for step_i in range(80):
        t += 0.5
        pose = _jitter_obs(GRANDMA_SLEEP_233638_OBS, 0.002 * (step_i % 2))
        assert are_legs_hanging_outside_bed(pose, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False
        roles, cg = ident.update([pose], REAL_ZONES, now=t)
        ev = fsm.step(
            [pose],
            roles,
            cg,
            REAL_ZONES,
            now=t,
            local_hour=23,
            patient_in_bed_latched=ident.patient_in_bed_latched,
        )
        assert ev is None, f"Ложная ночная тревога во время горизонтального сна на шаге {step_i}: {ev}"
        if step_i >= 2:
            assert fsm.state == PatientState.IN_BED


def test_wave10_mode_anchor_protection_and_engine_self_healing_on_sitting_up_and_fallen(
    tmp_path: Path,
) -> None:
    """
    Волна 10 (#161 Причины #1 и #4):
    1) Защита mode_anchor: чистый спальный эталон сохраняется в _clean_bed_anchor_day и не перезаписывается
       по face_patient вне кровати; метод rollback_or_clear_polluted_anchor откатывает загрязнённый якорь.
    2) Полное самоисцеление FallGuardEngine.report_false_alert на ложных sitting_up и fallen:
       - переводит внекроватного виновника в Role.VISITOR;
       - откатывает загрязнённый якорь одежды;
       - продвигает реальную бабушку на кровати в Role.PATIENT (кейс 102630 fallen) либо восстанавливает
         защёлку под одеялом (кейс 074550/110430 sitting_up);
       - снимает тревогу без 15-минутной паузы!
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

    # Формируем цветной дневной кадр с контрастно разной одеждой бабушки (синяя) и гостя (красная)
    frame_bed = np.full((720, 1280, 3), (90, 95, 100), dtype=np.uint8)
    cv2.rectangle(frame_bed, (380, 360), (780, 515), (195, 75, 45), thickness=-1)  # синяя кофта бабушки на кровати
    cv2.rectangle(frame_bed, (310, 340), (630, 660), (35, 45, 210), thickness=-1)  # красный свитер гостя на стуле

    # 1. Бабушка спит в кровати -> формируется чистый эталон _clean_bed_anchor_day
    t = 10.0
    for i in range(8):
        t += 0.3
        engine.identifier.update(
            [_jitter_obs(GRANDMA_SLEEP_233638_OBS, 0.002 * (i % 2))],
            zones=zones,
            now=t,
            frame_bgr=frame_bed,
        )
    clean_anchor = engine.identifier._anchor_vec_day
    assert clean_anchor is not None
    clean_anchor_copy = clean_anchor.copy()
    assert engine.identifier._clean_bed_anchor_day is not None

    # Имитируем загрязнение якоря чужим торсом и проверяем откат через report_false_alert на ложном SITTING_UP
    chair_vis = CAREGIVER_FB_POSES["110430_sitting_up"]
    for i in range(5):
        t += 0.3
        engine.identifier.update(
            [_jitter_obs(chair_vis, 0.02 * i)],
            zones=zones,
            now=t,
            frame_bgr=frame_bed,
        )
    assert engine.identifier.force_assign_patient(568) is True
    polluted_vec = compute_torso_appearance(frame_bed, chair_vis)
    assert polluted_vec is not None
    engine.identifier._set_mode_anchor(polluted_vec, night_ir=False, now=t)
    engine.identifier._patient_in_bed_latched = False
    engine.fsm._state = PatientState.SITTING_UP

    with engine._lock:
        engine._latest_raw_frame = frame_bed
        engine._latest_observations = [chair_vis]
        engine._latest_identities = dict(engine.identifier._tracks)
        engine._sync_tracks_summary_locked()

    now_wall = time.time()
    ev_sit = AlertEvent(
        state=PatientState.SITTING_UP,
        severity=AlertSeverity.WARNING,
        title="Пытается встать с кровати",
        description="Ложный sitting_up на родственнице на стуле.",
        timestamp=t,
        patient_track_id=568,
    )
    clip_sit = engine.archive.save_event_snapshot(ev_sit, b"\xff\xd8\xff\xe0SIT_JPG", unix_ts=now_wall - 1.0)
    sit_id = str(clip_sit["id"])
    engine.feedback.register_recent_alert(
        alert_id=sit_id,
        event=ev_sit,
        annotated_jpeg=b"\xff\xd8\xff\xe0SIT_JPG",
        raw_jpeg=b"\xff\xd8\xff\xe0RAW_JPG",
        context={
            "event_wall_ts": now_wall - 1.0,
            "telemetry_snapshot": engine.get_telemetry_snapshot(),
            "baseline_in_bed": dict(engine.fsm.baseline_in_bed),
            "zones": zones.to_dict(),
            "was_bed_latched": True,
        },
    )

    res_sit = engine.report_false_alert(sit_id, reporter_name="Тест SITTING_UP")
    assert res_sit["ok"] is True
    assert res_sit["reassigned_visitor_track_id"] == 568
    assert engine.identifier._tracks[568].role == Role.VISITOR
    assert engine.identifier.patient_in_bed_latched is True
    assert engine.fsm.state == PatientState.IN_BED
    assert engine.fsm.pause_remaining_sec(time.monotonic()) == 0.0
    # Загрязнённый якорь откатился к чистому спальному эталону бабушки!
    restored_anchor = engine.identifier._anchor_vec_day
    assert restored_anchor is not None
    assert float(np.dot(restored_anchor, clean_anchor_copy)) > 0.999

    # 2. Кейс Б: ложный FALLEN (аналог 102630) — помощница вне кровати tid=254 ошибочно PATIENT,
    # а сидящая на кровати бабушка tid=262 разжалована в VISITOR!
    helper_254 = CAREGIVER_FB_POSES["102630_fallen_helper"]
    for i in range(5):
        t += 0.3
        engine.identifier.update(
            [_jitter_obs(helper_254, 0.02 * i), _jitter_obs(GRANDMA_102630_OBS, 0.002 * (i % 2))],
            zones=zones,
            now=t,
            frame_bgr=frame_bed,
        )
    # Искусственно инвертируем роли (как произошло в старом баге 102630)
    engine.identifier._tracks[254].role = Role.PATIENT
    engine.identifier._tracks[254].reason = "face_patient"
    engine.identifier._tracks[262].role = Role.VISITOR
    engine.identifier._tracks[262].reason = "id_switch_demoted_visitor"
    engine.identifier._last_patient_track_id = 254
    engine.fsm._state = PatientState.FALLEN
    engine.fsm._fall_unacknowledged = True

    with engine._lock:
        engine._latest_raw_frame = frame_bed
        engine._latest_observations = [helper_254, GRANDMA_102630_OBS]
        engine._latest_identities = dict(engine.identifier._tracks)
        engine._sync_tracks_summary_locked()

    ev_fall = AlertEvent(
        state=PatientState.FALLEN,
        severity=AlertSeverity.CRITICAL,
        title="Падение / сползание вне кровати!",
        description="Ложный fallen на помощнице tid=254.",
        timestamp=t,
        patient_track_id=254,
    )
    clip_fall = engine.archive.save_event_snapshot(ev_fall, b"\xff\xd8\xff\xe0FALL_JPG", unix_ts=now_wall)
    fall_id = str(clip_fall["id"])
    engine.feedback.register_recent_alert(
        alert_id=fall_id,
        event=ev_fall,
        annotated_jpeg=b"\xff\xd8\xff\xe0FALL_JPG",
        raw_jpeg=b"\xff\xd8\xff\xe0RAW_JPG",
        context={
            "event_wall_ts": now_wall,
            "telemetry_snapshot": engine.get_telemetry_snapshot(),
            "baseline_in_bed": dict(engine.fsm.baseline_in_bed),
            "zones": zones.to_dict(),
            "was_bed_latched": False,
        },
    )

    res_fall = engine.report_false_alert(fall_id, reporter_name="Тест FALLEN")
    assert res_fall["ok"] is True
    assert res_fall["reassigned_visitor_track_id"] == 254
    assert engine.identifier._tracks[254].role == Role.VISITOR
    # Сидящая на кровати бабушка tid=262 автоматически возвращена в Role.PATIENT!
    assert engine.identifier._tracks[262].role == Role.PATIENT
    assert engine.fsm.fall_unacknowledged is False
    assert engine.fsm.state != PatientState.PAUSED
    assert engine.fsm.pause_remaining_sec(time.monotonic()) == 0.0
