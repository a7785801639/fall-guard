from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from app.detector import PoseTracker
from app.geometry import (
    PoseObservation,
    is_blanket_sleeper_in_bed,
    is_person_on_bed,
)
from app.zones import ZonesConfig
from tests.test_wave10_feedback_fixes import REAL_ZONES
from tests.test_wave13_standup_and_midroom_visitor_fixes import SEATED_GRANDMA_20261119


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


def _make_sleeper_kps(
    head: tuple[float, float] = (0.555, 0.555),
    sh_l: tuple[float, float] = (0.525, 0.570),
    sh_r: tuple[float, float] = (0.565, 0.576),
    hip_l: tuple[float, float] | None = (0.420, 0.596),
    hip_r: tuple[float, float] | None = (0.440, 0.612),
) -> tuple[tuple[float, float, float], ...]:
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[0] = (head[0], head[1], 0.88)
    kps[1] = (head[0] + 0.005, head[1] - 0.004, 0.85)
    kps[4] = (head[0] - 0.006, head[1] + 0.002, 0.82)
    kps[5] = (sh_l[0], sh_l[1], 0.90)
    kps[6] = (sh_r[0], sh_r[1], 0.89)
    if hip_l is not None:
        kps[11] = (hip_l[0], hip_l[1], 0.84)
    if hip_r is not None:
        kps[12] = (hip_r[0], hip_r[1], 0.83)
    return tuple(kps)


def _obs_to_full_frame_arrays(
    obs: PoseObservation,
    w: float,
    h: float,
) -> tuple[list[float], list[list[float]], list[float]]:
    bx = [
        obs.bbox_norm[0] * w,
        obs.bbox_norm[1] * h,
        obs.bbox_norm[2] * w,
        obs.bbox_norm[3] * h,
    ]
    xy = [[pt[0] * w if pt[2] > 0 else 0.0, pt[1] * h if pt[2] > 0 else 0.0] for pt in obs.keypoints_norm]
    cf = [pt[2] for pt in obs.keypoints_norm]
    return bx, xy, cf


def _obs_to_roi_arrays(
    obs: PoseObservation,
    w: float,
    h: float,
    rx1: float,
    ry1: float,
) -> tuple[list[float], list[list[float]], list[float]]:
    bx = [
        obs.bbox_norm[0] * w - rx1,
        obs.bbox_norm[1] * h - ry1,
        obs.bbox_norm[2] * w - rx1,
        obs.bbox_norm[3] * h - ry1,
    ]
    xy = [
        [
            (pt[0] * w - rx1) if pt[2] > 0 else 0.0,
            (pt[1] * h - ry1) if pt[2] > 0 else 0.0,
        ]
        for pt in obs.keypoints_norm
    ]
    cf = [pt[2] for pt in obs.keypoints_norm]
    return bx, xy, cf


# 1. Регрессия 560e: жёсткий clamp _bed_envelope_bbox в AABB bed_polygon (+0.02 X / +0.03 Y),
#    кап размера (<= 0.75 * bed_w, <= 0.72 * bed_h) и отсутствие монотонного min/max храповика при is_shrunk.
def test_wave15_560e_bed_envelope_clamped_and_no_shrunk_ratchet(tmp_path: Path) -> None:
    bed_poly = REAL_ZONES.bed_polygon
    b_xs = [float(p[0]) for p in bed_poly]
    b_ys = [float(p[1]) for p in bed_poly]
    b_min_x, b_max_x = min(b_xs), max(b_xs)
    b_min_y, b_max_y = min(b_ys), max(b_ys)
    bed_w = b_max_x - b_min_x
    bed_h = b_max_y - b_min_y

    tracker = PoseTracker("yolo11m-pose.pt", tmp_path, smooth_keypoints=False)

    # А. Прямой гигантский бокс из инцидента 560e (вылезает на коврик и ходунки до y2=0.970)
    oversized_560e = PoseObservation(
        track_id=9001,
        bbox_norm=(0.231, 0.440, 0.694, 0.970),
        conf=0.35,
        keypoints_norm=_make_sleeper_kps(head=(0.48, 0.56), sh_l=(0.45, 0.58), sh_r=(0.51, 0.58), hip_l=None, hip_r=None),
    )
    st_560e = tracker._stabilize_bed_sleeper_bbox(oversized_560e, bed_poly, now=1.0)
    assert st_560e.bbox_norm[0] >= (b_min_x - 0.02) - 1e-6
    assert st_560e.bbox_norm[2] <= (b_max_x + 0.02) + 1e-6
    assert st_560e.bbox_norm[1] >= (b_min_y - 0.03) - 1e-6
    assert st_560e.bbox_norm[3] <= (b_max_y + 0.03) + 1e-6
    assert st_560e.bbox_width <= (0.75 * bed_w) + 1e-6
    assert st_560e.bbox_height <= (0.72 * bed_h) + 1e-6

    # Б. Отсутствие монотонного min/max храповика при серии сжатых (is_shrunk) боксов:
    #    1) внутри env -> env остаётся неподвижен; 2) при раскачке по краям -> размер НЕ растёт и не выходит за AABB!
    tracker.reset_smoother()
    base_sleep = PoseObservation(
        track_id=1,
        bbox_norm=(0.34, 0.52, 0.59, 0.69),
        conf=0.58,
        keypoints_norm=_make_sleeper_kps(),
    )
    st0 = tracker._stabilize_bed_sleeper_bbox(base_sleep, bed_poly, now=10.0)
    env0 = tracker._bed_envelope_bbox
    assert env0 is not None
    init_w = env0[2] - env0[0]
    init_h = env0[3] - env0[1]

    # Сжатый бокс строго внутри env не должен сдвигать или раздувать env
    inside_shrunk = PoseObservation(
        track_id=1,
        bbox_norm=(0.50, 0.54, 0.58, 0.62),
        conf=0.28,
        keypoints_norm=_make_sleeper_kps(hip_l=None, hip_r=None),
    )
    for idx in range(20):
        st_in = tracker._stabilize_bed_sleeper_bbox(inside_shrunk, bed_poly, now=11.0 + idx * 0.2)
        assert st_in.bbox_norm == pytest.approx(st0.bbox_norm, abs=1e-9)

    # Раскачка сжатых боксов влево/вправо/вниз (включая выброс y2=0.96) не раздувает ширину/высоту env
    jitter_boxes = [
        (0.24, 0.50, 0.36, 0.60),
        (0.56, 0.51, 0.69, 0.61),
        (0.38, 0.42, 0.50, 0.52),
        (0.38, 0.72, 0.50, 0.96),
    ]
    for step in range(40):
        jb = jitter_boxes[step % len(jitter_boxes)]
        cx = 0.5 * (jb[0] + jb[2])
        cy = min(0.68, 0.5 * (jb[1] + jb[3]))
        j_obs = PoseObservation(
            track_id=1,
            bbox_norm=jb,
            conf=0.25,
            keypoints_norm=_make_sleeper_kps(
                head=(min(0.62, max(0.33, cx)), min(0.64, max(0.47, cy))),
                sh_l=(min(0.60, max(0.31, cx - 0.02)), min(0.66, max(0.49, cy + 0.02))),
                sh_r=(min(0.64, max(0.35, cx + 0.02)), min(0.66, max(0.49, cy + 0.02))),
                hip_l=None,
                hip_r=None,
            ),
        )
        st_j = tracker._stabilize_bed_sleeper_bbox(j_obs, bed_poly, now=20.0 + step * 0.2)
        assert st_j.bbox_norm[0] >= (b_min_x - 0.02) - 1e-6
        assert st_j.bbox_norm[2] <= (b_max_x + 0.02) + 1e-6
        assert st_j.bbox_norm[1] >= (b_min_y - 0.03) - 1e-6
        assert st_j.bbox_norm[3] <= (b_max_y + 0.03) + 1e-6
        assert st_j.bbox_width <= init_w + 1e-6
        assert st_j.bbox_height <= init_h + 1e-6

    # В. Сохранение tiny_head_obs (test_fallguard.py:2785-2824): тёплый и холодный старт держат >= 0.23 x >= 0.15
    tracker.reset_smoother()
    st_warm = tracker._stabilize_bed_sleeper_bbox(base_sleep, bed_poly, now=50.0)
    tiny_head_obs = PoseObservation(
        track_id=1,
        bbox_norm=(0.520, 0.525, 0.595, 0.600),
        conf=0.28,
        keypoints_norm=_make_sleeper_kps(
            head=(0.555, 0.555),
            sh_l=(0.527, 0.570),
            sh_r=(0.583, 0.575),
            hip_l=None,
            hip_r=None,
        ),
    )
    st_tiny = tracker._stabilize_bed_sleeper_bbox(tiny_head_obs, bed_poly, now=50.2)
    assert st_tiny.bbox_width >= 0.23
    assert st_tiny.bbox_height >= 0.15
    assert abs(st_tiny.bbox_norm[0] - st_warm.bbox_norm[0]) < 0.02

    tracker.reset_smoother()
    st_cold = tracker._stabilize_bed_sleeper_bbox(tiny_head_obs, bed_poly, now=55.0)
    assert st_cold.bbox_width >= 0.23
    assert st_cold.bbox_height >= 0.15
    assert st_cold.bbox_norm[3] <= (b_max_y + 0.03) + 1e-6


# 2. Регрессия b632: реальный выход трека #18413 с кровати (в том числе после посадки на край)
#    немедленно сбрасывает _last_bed_track_id / _cached_roi_obs, не позволяя ID телепортироваться на пустой матрас.
def test_wave15_b632_bed_exit_resets_pinning_and_prevents_id_teleport(tmp_path: Path) -> None:
    W, H = 1920.0, 1080.0
    seated_src = SEATED_GRANDMA_20261119["091209_sitting_up"]
    seated_18413 = PoseObservation(
        track_id=18413,
        bbox_norm=seated_src.bbox_norm,
        conf=0.86,
        keypoints_norm=seated_src.keypoints_norm,
    )
    assert is_person_on_bed(seated_18413, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    # Поза бабушки #18413, ушедшей с кровати к креслу-туалету справа (bb=[0.59, 0.48, 0.77, 0.87])
    kps_off_bed = [(0.0, 0.0, 0.0)] * 17
    kps_off_bed[0] = (0.680, 0.520, 0.92)
    kps_off_bed[1] = (0.674, 0.514, 0.90)
    kps_off_bed[2] = (0.686, 0.514, 0.90)
    kps_off_bed[5] = (0.652, 0.575, 0.94)
    kps_off_bed[6] = (0.708, 0.578, 0.94)
    kps_off_bed[11] = (0.658, 0.700, 0.91)
    kps_off_bed[12] = (0.704, 0.702, 0.91)
    kps_off_bed[13] = (0.660, 0.785, 0.88)
    kps_off_bed[14] = (0.702, 0.788, 0.88)
    kps_off_bed[15] = (0.662, 0.855, 0.86)
    kps_off_bed[16] = (0.700, 0.856, 0.86)
    off_bed_18413 = PoseObservation(
        track_id=18413,
        bbox_norm=(0.59, 0.48, 0.77, 0.87),
        conf=0.82,
        keypoints_norm=tuple(kps_off_bed),
    )
    assert is_person_on_bed(off_bed_18413, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is False

    # Спальная поза #18413 на кровати и последующая случайная детекция #18460 на пустом матрасе
    sleep_18413 = PoseObservation(
        track_id=18413,
        bbox_norm=(0.34, 0.50, 0.56, 0.74),
        conf=0.62,
        keypoints_norm=_make_sleeper_kps(
            head=(0.525, 0.555),
            sh_l=(0.500, 0.575),
            sh_r=(0.535, 0.585),
            hip_l=(0.405, 0.625),
            hip_r=(0.425, 0.640),
        ),
    )
    spurious_mattress_18460 = PoseObservation(
        track_id=18460,
        bbox_norm=(0.34, 0.49, 0.56, 0.75),
        conf=0.55,
        keypoints_norm=_make_sleeper_kps(
            head=(0.520, 0.552),
            sh_l=(0.495, 0.574),
            sh_r=(0.532, 0.584),
            hip_l=(0.402, 0.622),
            hip_r=(0.422, 0.638),
        ),
    )

    class _FakeB632Model:
        def __init__(self) -> None:
            self.phase = "sleeping"

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            h, w = source.shape[:2]
            if self.phase == "sleeping":
                bx, xy, cf = _obs_to_full_frame_arrays(sleep_18413, w, h)
                return [_FakeResult(_FakeBoxes([bx], [sleep_18413.conf], [18413]), _FakeKeypoints([xy], [cf]))]
            if self.phase == "sitting":
                bx, xy, cf = _obs_to_full_frame_arrays(seated_18413, w, h)
                return [_FakeResult(_FakeBoxes([bx], [seated_18413.conf], [18413]), _FakeKeypoints([xy], [cf]))]
            if self.phase == "exited_to_chair":
                bx, xy, cf = _obs_to_full_frame_arrays(off_bed_18413, w, h)
                return [_FakeResult(_FakeBoxes([bx], [off_bed_18413.conf], [18413]), _FakeKeypoints([xy], [cf]))]
            if self.phase == "empty_mattress_spurious":
                bx, xy, cf = _obs_to_full_frame_arrays(spurious_mattress_18460, w, h)
                return [
                    _FakeResult(
                        _FakeBoxes([bx], [spurious_mattress_18460.conf], [18460]),
                        _FakeKeypoints([xy], [cf]),
                    )
                ]
            return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]

    tracker = PoseTracker("yolo11m-pose.pt", tmp_path, conf=0.48, smooth_keypoints=False, roi_probe_interval=0.0)
    fake = _FakeB632Model()
    tracker._model = fake  # type: ignore[assignment]
    tracker._roi_model = fake  # type: ignore[assignment]
    tracker._runtime_label = "ONNX Runtime (CPU AVX2, passive)"
    rng = np.random.default_rng(632)
    frame = rng.integers(60, 140, size=(int(H), int(W), 3), dtype=np.uint8)

    # Шаг 1: бабушка #18413 спит в кровати -> пиннинг закреплён за 18413
    obs1 = tracker.infer(frame, zones=REAL_ZONES)
    assert [o.track_id for o in obs1] == [18413]
    assert tracker._last_bed_track_id == 18413
    assert tracker._last_bed_seen_ts > 0.0
    assert tracker._cached_roi_obs is not None

    # Шаг 2: бабушка #18413 села на край кровати -> _last_bed_track_id ещё 18413, кэш не клонируется в #9001
    fake.phase = "sitting"
    obs2 = tracker.infer(frame, zones=REAL_ZONES)
    assert [o.track_id for o in obs2] == [18413]
    assert tracker._last_bed_track_id == 18413

    # Шаг 3: бабушка #18413 сошла с кровати к креслу -> пиннинг и кэш ROI мгновенно сброшены!
    fake.phase = "exited_to_chair"
    obs3 = tracker.infer(frame, zones=REAL_ZONES)
    assert [o.track_id for o in obs3] == [18413]
    assert tracker._last_bed_track_id == 9001
    assert tracker._last_bed_bbox is None
    assert tracker._last_bed_seen_ts == 0.0
    assert tracker._cached_roi_obs is None

    # Шаг 4: случайная детекция #18460 на пустом матрасе (в пределах 120 с) НЕ получает старый ID #18413!
    fake.phase = "empty_mattress_spurious"
    obs4 = tracker.infer(frame, zones=REAL_ZONES)
    assert [o.track_id for o in obs4] == [18460]
    assert all(o.track_id != 18413 for o in obs4)

    # Шаг 5 (прямой выход без промежуточного сидения + мгновенный фантом на матрасе): тоже сбрасывает пиннинг
    tracker.reset_smoother()
    fake.phase = "sleeping"
    assert [o.track_id for o in tracker.infer(frame, zones=REAL_ZONES)] == [18413]
    fake.phase = "exited_to_chair"
    assert [o.track_id for o in tracker.infer(frame, zones=REAL_ZONES)] == [18413]
    assert tracker._last_bed_seen_ts == 0.0
    assert tracker._cached_roi_obs is None
    fake.phase = "empty_mattress_spurious"
    assert [o.track_id for o in tracker.infer(frame, zones=REAL_ZONES)] == [18460]


# 3. Регрессия b07d + защита co-presence (test_fallguard.py:1863) и одиночного сна:
#    если в existing_obs уже есть человек is_person_on_bed, Bed ROI требует conf >= eff_conf (0.48),
#    отсекая фантом одеяла #9001 (conf=0.17), но пропуская реального второго спящего (conf=0.61).
def test_wave15_b07d_roi_conf_gate_when_person_on_bed_and_copresence_preserved(tmp_path: Path) -> None:
    W, H = 1920.0, 1080.0
    # Окно 0 для REAL_ZONES.bed_polygon (min_x=0.2574, min_y=0.40458)
    windows = PoseTracker._bed_roi_windows(REAL_ZONES, int(W), int(H))
    rx1, ry1, _, _ = windows[0]

    seated_src = SEATED_GRANDMA_20261119["091720_left_bed"]
    seated_1834 = PoseObservation(
        track_id=1834,
        bbox_norm=seated_src.bbox_norm,
        conf=0.88,
        keypoints_norm=seated_src.keypoints_norm,
    )
    assert is_person_on_bed(seated_1834, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    # Фантом пустого одеяла #9001 с conf=0.17 справа от сидящей #1834 (сам по себе проходит is_blanket_sleeper_in_bed!)
    blanket_kps = _make_sleeper_kps(
        head=(0.560, 0.540),
        sh_l=(0.530, 0.565),
        sh_r=(0.585, 0.568),
        hip_l=None,
        hip_r=None,
    )
    blanket_phantom_low = PoseObservation(
        track_id=9001,
        bbox_norm=(0.455, 0.495, 0.615, 0.665),
        conf=0.17,
        keypoints_norm=blanket_kps,
    )
    assert is_blanket_sleeper_in_bed(blanket_phantom_low, REAL_ZONES.bed_polygon, zones=REAL_ZONES) is True

    # Реальная спящая бабушка рядом с сидящей на кровати гостьей (conf=0.61 >= 0.48)
    real_sleeper_high = PoseObservation(
        track_id=9001,
        bbox_norm=(0.455, 0.495, 0.615, 0.665),
        conf=0.61,
        keypoints_norm=blanket_kps,
    )

    class _FakeB07dModel:
        def __init__(self) -> None:
            self.full_mode = "seated_1834"
            self.roi_conf_val = 0.17
            self.predict_confs: list[float] = []

        def track(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            h, w = source.shape[:2]
            if self.full_mode == "seated_1834":
                bx, xy, cf = _obs_to_full_frame_arrays(seated_1834, w, h)
                return [_FakeResult(_FakeBoxes([bx], [seated_1834.conf], [1834]), _FakeKeypoints([xy], [cf]))]
            return [_FakeResult(_FakeBoxes([], []), _FakeKeypoints([], []))]

        def predict(self, source: np.ndarray, **kwargs: object) -> list[_FakeResult]:
            self.predict_confs.append(float(kwargs.get("conf", -1.0)))
            # Намеренно игнорируем kwargs['conf'] и возвращаем кандидата с self.roi_conf_val,
            # чтобы проверить жёсткую фильтрацию в самом _infer_single_roi_window!
            cand = PoseObservation(
                track_id=9001,
                bbox_norm=blanket_phantom_low.bbox_norm,
                conf=self.roi_conf_val,
                keypoints_norm=blanket_phantom_low.keypoints_norm,
            )
            bx, xy, cf = _obs_to_roi_arrays(cand, W, H, float(rx1), float(ry1))
            return [_FakeResult(_FakeBoxes([bx], [self.roi_conf_val]), _FakeKeypoints([xy], [cf]))]

    tracker = PoseTracker("yolo11m-pose.pt", tmp_path, conf=0.48, smooth_keypoints=False, roi_probe_interval=0.0)
    fake = _FakeB07dModel()
    tracker._model = fake  # type: ignore[assignment]
    tracker._roi_model = fake  # type: ignore[assignment]
    tracker._runtime_label = "ONNX Runtime (CPU AVX2, passive)"
    frame = np.zeros((int(H), int(W), 3), dtype=np.uint8)

    # А. Инцидент b07d: на кровати уже сидит #1834 -> ROI запрашивается с conf=0.48 и ОТСЕКАЕТ одеяло conf=0.17 (и 0.30)!
    fake.full_mode = "seated_1834"
    fake.roi_conf_val = 0.17
    obs_b07d = tracker.infer(frame, zones=REAL_ZONES)
    assert fake.predict_confs and all(c == pytest.approx(0.48) for c in fake.predict_confs)
    assert [o.track_id for o in obs_b07d] == [1834]

    fake.roi_conf_val = 0.30
    obs_b07d_30 = tracker.infer(frame, zones=REAL_ZONES)
    assert [o.track_id for o in obs_b07d_30] == [1834]

    # Прежний низкоуверенный кэш ROI (conf=0.17) не просачивается рядом с сидящей #1834
    # ни при пустом ответе ROI-пробы, ни между интервалами проб (roi_probe_interval > 0)
    rng = np.random.default_rng(7)
    tex_frame = rng.integers(50, 150, size=(int(H), int(W), 3), dtype=np.uint8)
    now_mono = time.monotonic()
    tracker._cached_roi_obs = blanket_phantom_low
    tracker._cached_roi_ts = now_mono
    fake.roi_conf_val = 0.17
    obs_no_cache_leak = tracker.infer(tex_frame, zones=REAL_ZONES)
    assert [o.track_id for o in obs_no_cache_leak] == [1834]
    assert tracker._cached_roi_obs is None

    tracker._roi_probe_interval = 5.0
    tracker._last_roi_probe_ts = time.monotonic()
    tracker._cached_roi_obs = blanket_phantom_low
    tracker._cached_roi_ts = time.monotonic()
    obs_no_interprobe_leak = tracker.infer(tex_frame, zones=REAL_ZONES)
    assert [o.track_id for o in obs_no_interprobe_leak] == [1834]
    tracker._roi_probe_interval = 0.0

    # Б. Co-presence со спящей бабушкой (conf=0.61 >= 0.48) при сидящем на кровати человеке -> ОБА трека сохранены!
    fake.roi_conf_val = real_sleeper_high.conf
    obs_both = tracker.infer(frame, zones=REAL_ZONES)
    assert {o.track_id for o in obs_both} == {1834, 9001}

    # В. Одиночный глубокий сон под одеялом (на полном кадре никого is_person_on_bed нет) -> conf=0.07 и conf=0.17 ПРОХОДИТ!
    tracker.reset_smoother()
    fake.predict_confs.clear()
    fake.full_mode = "empty"
    fake.roi_conf_val = 0.17
    obs_alone = tracker.infer(frame, zones=REAL_ZONES)
    assert fake.predict_confs and all(c == pytest.approx(0.07) for c in fake.predict_confs)
    assert [o.track_id for o in obs_alone] == [9001]
