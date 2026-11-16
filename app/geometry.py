from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .homography import FloorHomography

# Индексы точек COCO-17 (YOLO-pose)
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16


@dataclass(frozen=True)
class PoseObservation:
    """Наблюдение одного человека в кадре с трекингом и нормализованными координатами [0..1]."""

    track_id: int
    bbox_norm: tuple[float, float, float, float]  # (x1, y1, x2, y2) в [0..1]
    conf: float
    keypoints_norm: tuple[tuple[float, float, float], ...]  # 17 × (x_norm, y_norm, kp_conf)
    liveness_verified: bool = False
    spawn_center_hint: tuple[float, float] | None = None
    confirmed_frames: int = 0
    outside_bed_disp: float = 0.0
    is_still: bool = False
    _kp_stabilized: bool = False

    def kp(self, idx: int, min_conf: float = 0.30) -> tuple[float, float] | None:
        if idx < 0 or idx >= len(self.keypoints_norm):
            return None
        x, y, c = self.keypoints_norm[idx]
        if c < min_conf or (x <= 0.0 and y <= 0.0):
            return None
        return (x, y)

    def midpoint(self, idx_a: int, idx_b: int, min_conf: float = 0.30) -> tuple[float, float] | None:
        pa = self.kp(idx_a, min_conf)
        pb = self.kp(idx_b, min_conf)
        if pa and pb:
            return ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5)
        return pa or pb

    @property
    def mid_shoulder(self) -> tuple[float, float] | None:
        return self.midpoint(KP_L_SHOULDER, KP_R_SHOULDER)

    @property
    def mid_hip(self) -> tuple[float, float] | None:
        return self.midpoint(KP_L_HIP, KP_R_HIP)

    @property
    def head_center(self) -> tuple[float, float] | None:
        pts = [self.kp(i, 0.30) for i in (KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR)]
        valid = [p for p in pts if p is not None]
        if not valid:
            return None
        return (
            sum(p[0] for p in valid) / len(valid),
            sum(p[1] for p in valid) / len(valid),
        )

    def has_confident_head(self, min_conf: float = 0.35) -> bool:
        """Есть ли хотя бы одна уверенная ключевая точка головы (нос, глаза, уши)."""
        return any(
            self.kp(i, min_conf) is not None
            for i in (KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR)
        )

    @property
    def upper_body_point(self) -> tuple[float, float]:
        """Опорная точка верхней части корпуса (плечи -> голова -> центр тела)."""
        return self.mid_shoulder or self.head_center or self.body_center

    @property
    def body_center(self) -> tuple[float, float]:
        mh = self.mid_hip
        ms = self.mid_shoulder
        if mh and ms:
            return ((mh[0] + ms[0]) * 0.5, (mh[1] + ms[1]) * 0.5)
        if mh:
            return mh
        if ms:
            return ms
        x1, y1, x2, y2 = self.bbox_norm
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def lower_limbs(self) -> list[tuple[float, float]]:
        pts = [
            self.kp(i, 0.32)
            for i in (KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE)
        ]
        return [p for p in pts if p is not None]

    @property
    def bbox_width(self) -> float:
        return max(1e-4, self.bbox_norm[2] - self.bbox_norm[0])

    @property
    def bbox_height(self) -> float:
        return max(1e-4, self.bbox_norm[3] - self.bbox_norm[1])

    @property
    def aspect_ratio_wh(self) -> float:
        """Отношение ширины бокса к высоте (>1.12 характерно для горизонтального положения)."""
        return self.bbox_width / self.bbox_height

    def torso_length_2d(self, aspect_xy: float = 16.0 / 9.0) -> float | None:
        """Длина 2D-проекции вектора 'таз -> плечи' с поправкой на соотношение сторон кадра."""
        ms = self.mid_shoulder
        mh = self.mid_hip
        if not ms or not mh:
            return None
        dx = (ms[0] - mh[0]) * aspect_xy
        dy = mh[1] - ms[1]
        norm = math.hypot(dx, dy)
        return norm if norm >= 1e-4 else None

    def torso_angle_from_vertical_deg(self, aspect_xy: float = 16.0 / 9.0) -> float | None:
        """
        Угол линии 'таз -> плечи' от вертикали вверх (в градусах):
        0° — сидит или стоит строго вертикально (либо лежит вдоль оптической оси камеры);
        90° — лежит горизонтально поперёк кадра;
        >90° — голова/плечи ниже уровня таза.
        """
        ms = self.mid_shoulder
        mh = self.mid_hip
        if not ms or not mh:
            return None
        dx = (ms[0] - mh[0]) * aspect_xy
        dy = mh[1] - ms[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-4:
            return None
        cos_val = max(-1.0, min(1.0, dy / norm))
        return math.degrees(math.acos(cos_val))

    def torso_vertical_rise(self) -> float:
        """Насколько плечи выше таза в долях высоты кадра (>0 — плечи выше таза)."""
        ms = self.mid_shoulder
        mh = self.mid_hip
        if not ms or not mh:
            return 0.0
        return mh[1] - ms[1]

    def is_near_border(self, margin: float = 0.06) -> bool:
        x1, y1, x2, y2 = self.bbox_norm
        return x1 <= margin or y1 <= margin or x2 >= (1.0 - margin) or y2 >= (1.0 - margin)

    def valid_keypoint_count(self, min_conf: float = 0.30) -> int:
        """Количество достоверно видимых ключевых точек скелета."""
        cnt = 0
        for idx in range(min(17, len(self.keypoints_norm))):
            x, y, c = self.keypoints_norm[idx]
            if c >= min_conf and (x > 0.0 or y > 0.0):
                cnt += 1
        return cnt

    def has_core_anatomy(self, min_conf: float = 0.30) -> bool:
        """
        Проверяет наличие базовой человеческой анатомии (плечи, таз или голова + конечность).
        Отсекает бесформенные пятна (скомканные одеяла, подушки, засветку ИК).
        """
        if self.midpoint(KP_L_SHOULDER, KP_R_SHOULDER, min_conf) is not None:
            return True
        if self.midpoint(KP_L_HIP, KP_R_HIP, min_conf) is not None:
            return True
        head_pts = sum(
            1 for i in (KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR) if self.kp(i, min_conf) is not None
        )
        limb_pts = sum(
            1
            for i in (KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST, KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE)
            if self.kp(i, min_conf) is not None
        )
        return head_pts >= 2 and limb_pts >= 2


def point_in_polygon(pt: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    """Проверка вхождения нормализованной точки (x, y) в полигон [(x1, y1), ...]."""
    if len(polygon) < 3:
        return False
    arr = np.array(polygon, dtype=np.float32) * 1000.0
    px, py = float(pt[0]) * 1000.0, float(pt[1]) * 1000.0
    return cv2.pointPolygonTest(arr, (px, py), False) >= 0.0


def polygon_distance_norm(pt: tuple[float, float], polygon: Sequence[Sequence[float]]) -> float:
    """Знаковое расстояние до границы полигона в нормализованных единицах (>0 внутри, <0 снаружи)."""
    if len(polygon) < 3:
        return -1.0
    arr = np.array(polygon, dtype=np.float32) * 1000.0
    px, py = float(pt[0]) * 1000.0, float(pt[1]) * 1000.0
    return float(cv2.pointPolygonTest(arr, (px, py), True)) / 1000.0


def polygon_y_bounds(polygon: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Возвращает (min_y, max_y) полигона в нормализованных координатах [0..1]."""
    if len(polygon) < 3:
        return (0.0, 1.0)
    ys = [float(p[1]) for p in polygon]
    return (min(ys), max(ys))


def bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Вычисляет IoU двух нормализованных прямоугольников (x1, y1, x2, y2)."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1e-6, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


def bbox_polygon_overlap_ratio(
    bbox_norm: tuple[float, float, float, float],
    polygon: Sequence[Sequence[float]],
    grid_n: int = 8,
) -> float:
    """Доля площади прямоугольника bbox_norm, попадающая внутрь произвольного полигона [0..1]."""
    if len(polygon) < 3:
        return 0.0
    x1, y1, x2, y2 = bbox_norm
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1e-5 or bh <= 1e-5:
        return 0.0
    arr = np.array(polygon, dtype=np.float32) * 1000.0
    # Быстрый отсев по AABB полигона
    px_min = float(np.min(arr[:, 0])) / 1000.0
    px_max = float(np.max(arr[:, 0])) / 1000.0
    py_min = float(np.min(arr[:, 1])) / 1000.0
    py_max = float(np.max(arr[:, 1])) / 1000.0
    if x2 < px_min or x1 > px_max or y2 < py_min or y1 > py_max:
        return 0.0

    inside = 0
    total = grid_n * grid_n
    for iy in range(grid_n):
        sy = (y1 + bh * ((iy + 0.5) / grid_n)) * 1000.0
        for ix in range(grid_n):
            sx = (x1 + bw * ((ix + 0.5) / grid_n)) * 1000.0
            if cv2.pointPolygonTest(arr, (sx, sy), False) >= 0.0:
                inside += 1
    return inside / float(total)


def is_pose_touching_door(
    obs: PoseObservation,
    door_polygon: Sequence[Sequence[float]] | None,
    min_overlap: float = 0.20,
) -> bool:
    """
    Проверяет, пересекает ли человек полигон двери (door_polygon) ключевыми опорными точками
    или площадью бокса >= min_overlap (по умолчанию 20%).
    """
    if not door_polygon or len(door_polygon) < 3:
        return False
    if point_in_polygon(obs.body_center, door_polygon):
        return True
    if point_in_polygon(obs.upper_body_point, door_polygon):
        return True
    if obs.mid_hip is not None and point_in_polygon(obs.mid_hip, door_polygon):
        return True
    if obs.head_center is not None and point_in_polygon(obs.head_center, door_polygon):
        return True
    return bbox_polygon_overlap_ratio(obs.bbox_norm, door_polygon) >= min_overlap


def is_confident_foreground_seated_person(
    obs: PoseObservation,
    bed_polygon: Sequence[Sequence[float]] | None = None,
    zones: Any = None,
) -> bool:
    """
    Распознаёт реального человека, сидящего на стуле ближнего переднего плана у нижнего края кадра
    (например, ухаживающий родственник на стуле слева у двери/кровати: y1 >= 0.54, y2 >= 0.92,
    компактная сидячая высота <= 0.46, высокая уверенность YOLO >= 0.78 и оба плеча + оба бедра).
    Отличается от настенной куртки/вешалки (y1 <= 0.42, y2 <= 0.90, bh >= 0.50), дверного косяка
    и стоячих фантомов в комнате (y1 <= 0.48, bh >= 0.47).
    """
    if bed_polygon and len(bed_polygon) >= 3 and is_person_on_bed(obs, bed_polygon, zones=zones):
        return False
    _, y1, _, y2 = obs.bbox_norm
    if y1 < 0.54 or y2 < 0.92:
        return False
    if not (0.22 <= obs.bbox_height <= 0.46):
        return False
    if obs.conf < 0.78:
        return False
    if obs.kp(KP_L_SHOULDER, 0.65) is None or obs.kp(KP_R_SHOULDER, 0.65) is None:
        return False
    if obs.kp(KP_L_HIP, 0.60) is None or obs.kp(KP_R_HIP, 0.60) is None:
        return False
    has_upper_detail = (
        obs.has_confident_head(0.30)
        or any(obs.kp(i, 0.40) is not None for i in (KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST))
        or (
            obs.conf >= 0.85
            and obs.kp(KP_L_SHOULDER, 0.80) is not None
            and obs.kp(KP_R_SHOULDER, 0.80) is not None
            and obs.kp(KP_L_HIP, 0.70) is not None
            and obs.kp(KP_R_HIP, 0.70) is not None
        )
    )
    return bool(has_upper_detail)


def is_pose_in_ignore_zones(
    obs: PoseObservation,
    ignore_polygons: Sequence[Sequence[Sequence[float]]] | None,
    overlap_threshold: float = 0.55,
    door_polygon: Sequence[Sequence[float]] | None = None,
) -> bool:
    """
    Проверяет, попадает ли скелет в любую из зон игнорирования предметов (ignore_polygons):
    отсекается только если поза НЕ касается door_polygon И при этом её body_center лежит внутри
    игнор-полигона либо >= overlap_threshold (по умолчанию 55%) площади bbox_norm перекрыто игнор-зоной.
    Защищает стул для родственника у двери и ближнего переднего плана от поглощения высокой настенной ignore_polygon.
    """
    if not ignore_polygons:
        return False
    if is_pose_touching_door(obs, door_polygon, min_overlap=0.10):
        return False
    fg_seated = is_confident_foreground_seated_person(obs)
    bc = obs.body_center
    eff_overlap = max(float(overlap_threshold), 0.55)
    for poly in ignore_polygons:
        if len(poly) < 3:
            continue
        if fg_seated:
            poly_min_y = min(float(pt[1]) for pt in poly)
            if poly_min_y <= 0.25:
                continue
        if point_in_polygon(bc, poly):
            return True
        if bbox_polygon_overlap_ratio(obs.bbox_norm, poly) >= eff_overlap:
            return True
    return False


def is_standing_in_front_of_bed(
    obs: PoseObservation,
    bed_polygon: Sequence[Sequence[float]],
    zones: Any = None,
) -> bool:
    """
    Определяет человека (включая ребёнка или наклонившегося над подушкой ухаживающего),
    стоящего на полу вплотную ПЕРЕД бортом кровати:
    через проективную 3D-геометрию FloorHomography отделяет плоскость пола Z = 0 см от поверхности матраса Z = 55 см,
    устраняя 2D-параллакс таза при верхне-боковом ракурсе стационарной камеры.
    """
    if len(bed_polygon) < 3:
        return False
    angle = obs.torso_angle_from_vertical_deg()
    if angle is None or angle >= 72.0 or obs.aspect_ratio_wh >= 0.70:
        return False
    if angle >= 48.0:
        mh_bend = obs.mid_hip
        has_vertical_standing_leg = mh_bend is not None and any(
            (ak := obs.kp(idx, 0.45)) is not None and (ak[1] - mh_bend[1]) >= 0.16
            for idx in (KP_L_ANKLE, KP_R_ANKLE)
        )
        if not (obs.aspect_ratio_wh < 0.55 and obs.bbox_height >= 0.42 and has_vertical_standing_leg):
            return False

    bed_min_y, bed_max_y = polygon_y_bounds(bed_polygon)
    hom = FloorHomography.from_zones(zones, bed_polygon=bed_polygon)
    m = hom.measure_pose(obs, bed_polygon=bed_polygon) if hom.is_valid else None
    if m is not None:
        if angle < 28.0 and obs.aspect_ratio_wh < 0.56 and m.standing_on_floor_in_front:
            return True
        # Полноростовой взрослый на калиброванном полу перед кроватью (включая лёгкий сгиб колена или наклон к подушке)
        if (
            hom.calibrated
            and obs.bbox_norm[3] >= (bed_max_y + 0.06)
            and obs.bbox_height >= 0.42
        ):
            if (
                angle < 34.0
                and m.height_cm is not None
                and m.height_cm >= 168.0
                and (
                    obs.bbox_norm[1] <= (bed_min_y - 0.015)
                    or (m.floor_xy_cm is not None and m.floor_xy_cm[1] >= 58.0)
                )
            ):
                return True
            if (
                hom.calibrated
                and m.floor_xy_cm is not None
                and m.floor_xy_cm[1] >= 75.0
                and obs.bbox_norm[3] >= (bed_max_y + 0.10)
            ):
                return True

    # Наклонившийся над кроватью ухаживающий, стоящий ногами на полу (оба видимых колена глубоко ниже борта матраса)
    if obs.bbox_height >= 0.46 and obs.bbox_norm[3] >= (bed_max_y + 0.08):
        vis_knees = [
            pt[1]
            for idx in (KP_L_KNEE, KP_R_KNEE)
            if (pt := obs.kp(idx, 0.35)) is not None
        ]
        if (
            vis_knees
            and max(vis_knees) >= (bed_max_y + 0.05)
            and min(vis_knees) >= (bed_max_y + 0.02)
        ):
            return True

    # Геометрический фолбэк для нестандартных синтетических полигонов без калиброванного пола
    if angle >= 28.0 or obs.aspect_ratio_wh >= 0.55:
        return False
    ankles_on_floor = False
    for idx in (KP_L_ANKLE, KP_R_ANKLE):
        ak = obs.kp(idx, 0.35)
        if ak is not None and ak[1] >= (bed_max_y + 0.03) and not point_in_polygon(ak, bed_polygon):
            ankles_on_floor = True
            break
    if not (obs.bbox_norm[3] >= (bed_max_y + 0.045) or ankles_on_floor):
        return False

    mh = obs.mid_hip
    if mh is not None:
        dist_in = polygon_distance_norm(mh, bed_polygon)
        if mh[1] < (bed_max_y - 0.08) and dist_in > 0.08:
            return False
    return True


def is_person_on_bed(
    obs: PoseObservation,
    bed_polygon: Sequence[Sequence[float]],
    zones: Any = None,
) -> bool:
    """Считаем человека находящимся на кровати, если таз (или голова+плечи под одеялом) внутри полигона кровати."""
    if len(bed_polygon) < 3:
        return False
    if is_standing_in_front_of_bed(obs, bed_polygon, zones=zones):
        return False
    mh = obs.mid_hip
    hc = obs.head_center
    ms = obs.mid_shoulder
    if mh is not None:
        if point_in_polygon(mh, bed_polygon):
            return True
        if (
            hc is not None
            and ms is not None
            and point_in_polygon(hc, bed_polygon)
            and point_in_polygon(ms, bed_polygon)
            and obs.bbox_height <= 0.30
            and not obs.lower_limbs
        ):
            return True
        return False
    if ms is not None and point_in_polygon(ms, bed_polygon):
        return True
    if hc is not None and point_in_polygon(hc, bed_polygon):
        return True
    return point_in_polygon(obs.body_center, bed_polygon)


def sanitize_collapsed_bed_keypoints(
    obs: PoseObservation,
    bed_polygon: Sequence[Sequence[float]],
) -> PoseObservation:
    """
    Санирует смятые/сваленные кучей на подушку нижние ключевые точки (таз, колени, лодыжки 11..16)
    у человека внутри bed_polygon: когда под одеялом видно только лицо/плечи, YOLO-pose может
    свалить все 17 точек в крошечный головной комок (давая ложный вертикальный угол ~0..5° и
    визуальный мусор микро-скелета на лице).
    Обнуляет такие фантомные точки 11..16 в (0.0, 0.0, 0.0) до расчёта углов и сглаживания,
    не затрагивая реальный горизонтальный скелет спящего вдоль кровати человека.
    """
    if len(bed_polygon) < 3 or len(obs.keypoints_norm) < 17:
        return obs

    hc = obs.head_center
    ms = obs.mid_shoulder
    if hc is None or not point_in_polygon(hc, bed_polygon) or obs.bbox_height > 0.22:
        return obs
    if ms is not None and not point_in_polygon(ms, bed_polygon):
        return obs

    mh = obs.mid_hip
    raw_angle = obs.torso_angle_from_vertical_deg()
    collapsed_box = obs.bbox_width <= 0.145 and obs.bbox_height <= 0.135
    torso_len = (
        math.hypot(ms[0] - mh[0], ms[1] - mh[1])
        if (ms is not None and mh is not None)
        else None
    )
    collapsed_torso = (
        torso_len is not None
        and (
            torso_len < 0.042
            or (torso_len < 0.068 and (raw_angle is None or raw_angle < 42.0))
        )
    )

    pts = list(obs.keypoints_norm)
    changed = False

    for idx in (KP_L_HIP, KP_R_HIP, KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE):
        kx, ky, kc = pts[idx]
        if kc <= 0.0 and kx == 0.0 and ky == 0.0:
            continue
        if collapsed_box or collapsed_torso:
            pts[idx] = (0.0, 0.0, 0.0)
            changed = True
            continue

        is_hip = idx in (KP_L_HIP, KP_R_HIP)
        thr_head = 0.075 if is_hip else 0.088
        near_head = math.hypot(kx - hc[0], ky - hc[1]) < thr_head
        near_sh = (
            ms is not None
            and (obs.bbox_width <= 0.155 or raw_angle is None or raw_angle < 42.0)
            and math.hypot(kx - ms[0], ky - ms[1]) < (0.062 if is_hip else 0.082)
        )
        if near_head or near_sh:
            pts[idx] = (0.0, 0.0, 0.0)
            changed = True

    if changed and obs.bbox_height < 0.22:
        l_hip_gone = pts[KP_L_HIP][2] < 0.30
        r_hip_gone = pts[KP_R_HIP][2] < 0.30
        if l_hip_gone and r_hip_gone:
            for idx in (KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE):
                if pts[idx] != (0.0, 0.0, 0.0):
                    pts[idx] = (0.0, 0.0, 0.0)

    if not changed:
        return obs
    return PoseObservation(
        track_id=obs.track_id,
        bbox_norm=obs.bbox_norm,
        conf=obs.conf,
        keypoints_norm=tuple(pts),
        liveness_verified=obs.liveness_verified,
        spawn_center_hint=obs.spawn_center_hint,
        confirmed_frames=obs.confirmed_frames,
        outside_bed_disp=obs.outside_bed_disp,
        is_still=obs.is_still,
        _kp_stabilized=obs._kp_stabilized,
    )


def is_blanket_sleeper_in_bed(
    obs: PoseObservation,
    bed_polygon: Sequence[Sequence[float]],
    min_bed_box_conf: float = 0.072,
) -> bool:
    """
    Распознаёт человека, спящего под плотным/пёстрым одеялом строго внутри bed_polygon
    (когда на подушке видна только голова и край плеч, бёдра скрыты одеялом, а box_conf
    может проседать до 0.072..0.47 из-за пёстрого рисунка одеяла и контрового света окна).
    Исключает висящую на стене одежду и бесформенные складки пустого одеяла.
    """
    if len(bed_polygon) < 3 or obs.conf < min_bed_box_conf:
        return False
    hc = obs.head_center
    ms = obs.mid_shoulder
    mh = obs.mid_hip
    if ms is None:
        return False
    if hc is None:
        # Сон отвернувшись лицом в подушку/к стене: требуем полный горизонтальный скелет корпуса и конечностей на матрасе
        angle_body = obs.torso_angle_from_vertical_deg()
        return bool(
            obs.conf >= max(min_bed_box_conf, 0.24)
            and obs.aspect_ratio_wh >= 0.70
            and mh is not None
            and point_in_polygon(ms, bed_polygon)
            and point_in_polygon(mh, bed_polygon)
            and obs.kp(KP_L_SHOULDER, 0.50) is not None
            and obs.kp(KP_R_SHOULDER, 0.50) is not None
            and obs.kp(KP_L_HIP, 0.50) is not None
            and obs.kp(KP_R_HIP, 0.50) is not None
            and obs.valid_keypoint_count(0.35) >= 8
            and angle_body is not None
            and angle_body >= 55.0
        )
    if not (point_in_polygon(hc, bed_polygon) and point_in_polygon(ms, bed_polygon)):
        return False

    confident_head_pts = sum(
        1 for i in (KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR) if obs.kp(i, 0.42) is not None
    )
    if obs.conf < 0.17:
        # Глубокое укрывание пёстрым одеялом по подбородок (0.072 <= box_conf < 0.17):
        # строгая проверка выраженной головы (глаз/ухо >= 0.45) + ОБОИХ плеч на подушке внутри bed_polygon
        has_eye_or_ear = any(
            obs.kp(i, 0.45) is not None for i in (KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR)
        )
        both_shoulders_visible = (
            obs.kp(KP_L_SHOULDER, 0.32) is not None
            and obs.kp(KP_R_SHOULDER, 0.32) is not None
            and (obs.kp(KP_L_SHOULDER, 0.48) is not None or obs.kp(KP_R_SHOULDER, 0.48) is not None)
        )
        if not (
            confident_head_pts >= 2
            and has_eye_or_ear
            and both_shoulders_visible
            and obs.bbox_height <= 0.32
            and obs.aspect_ratio_wh >= 0.40
            and obs.valid_keypoint_count(0.30) >= 5
        ):
            return False
    elif confident_head_pts >= 2:
        if obs.kp(KP_L_SHOULDER, 0.45) is None and obs.kp(KP_R_SHOULDER, 0.45) is None:
            return False
        if obs.valid_keypoint_count(0.30) < 4:
            return False
    elif confident_head_pts == 1:
        # Сон на боку или с прикрытым предплечьем лицом (открыто ухо/висок >= 0.55 при обоих плечах и >= 6 точках)
        strong_ear_or_head = any(
            obs.kp(i, 0.55) is not None for i in (KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR)
        )
        both_shoulders = (
            obs.kp(KP_L_SHOULDER, 0.55) is not None and obs.kp(KP_R_SHOULDER, 0.55) is not None
        )
        if not (strong_ear_or_head and both_shoulders and obs.valid_keypoint_count(0.45) >= 6):
            return False
    else:
        return False

    angle = obs.torso_angle_from_vertical_deg()
    if angle is not None and angle < 28.0 and obs.bbox_height >= 0.34 and obs.aspect_ratio_wh < 0.72:
        return False
    if angle is not None and angle < 20.0 and obs.bbox_height >= 0.20 and obs.aspect_ratio_wh < 0.35:
        return False
    return True


def are_legs_hanging_outside_bed(
    obs: PoseObservation,
    bed_polygon: Sequence[Sequence[float]],
    min_outside_margin: float = 0.025,
    min_limb_conf: float = 0.48,
    zones: Any = None,
) -> bool:
    """
    Детекция свешивания ног с кровати:
    таз (или плечи при тазе на самом бортике) ещё в пределах кровати,
    а хотя бы одно уверенно видимое колено или стопа (>= min_limb_conf) вышли наружу за контур кровати дальше margin.
    Подпороговые под-одеяльные точки (< 0.48) и глубоко горизонтальная поза сна вдоль матраса игнорируются,
    чтобы не сбивать детектор сна у выемки bed_polygon.
    Если и таз, и плечи уже вышли за контур кровати — человек полностью покинул кровать (LEFT_BED).
    """
    if len(bed_polygon) < 3 or is_standing_in_front_of_bed(obs, bed_polygon, zones=zones):
        return False
    _, bed_max_y = polygon_y_bounds(bed_polygon)
    angle = obs.torso_angle_from_vertical_deg()
    ms = obs.mid_shoulder
    mh = obs.mid_hip
    if (
        angle is not None
        and angle >= 60.0
        and ms is not None
        and mh is not None
        and point_in_polygon(ms, bed_polygon)
        and point_in_polygon(mh, bed_polygon)
        and obs.bbox_norm[3] <= (bed_max_y + 0.03)
    ):
        return False
    anchor = obs.mid_hip or obs.body_center
    dist_hip = polygon_distance_norm(anchor, bed_polygon)
    if dist_hip < -0.025:
        return False
    if dist_hip < 0.0:
        if ms is None or not point_in_polygon(ms, bed_polygon):
            return False
    eff_conf = max(0.32, float(min_limb_conf))
    for kp_idx in (KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE):
        limb_pt = obs.kp(kp_idx, eff_conf)
        if limb_pt is not None and polygon_distance_norm(limb_pt, bed_polygon) < -min_outside_margin:
            return True
    return False


def is_hand_raised_for_help(
    obs: PoseObservation,
    baseline_shoulder_y: float | None = None,
) -> tuple[bool, str]:
    """
    Биомеханический детектор намеренного жеста вызова помощи поднятой вверх рукой
    (для пациентов после инсульта с афазией/дизартрией) с защитой от руки под подушкой во сне.
    """
    nose = obs.kp(KP_NOSE, 0.40)
    head_pts = [obs.kp(i, 0.45) for i in (KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR)]
    valid_head_ys = [pt[1] for pt in head_pts if pt is not None]
    min_head_y = min(valid_head_ys) if valid_head_ys else None
    min_lift = max(0.105, obs.bbox_height * 0.30)

    raised_sides: list[str] = []
    arms = (
        (KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST, "левая рука"),
        (KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST, "правая рука"),
    )
    for kp_sh, kp_el, kp_wr, side_ru in arms:
        wrist = obs.kp(kp_wr, 0.50)
        shoulder = obs.kp(kp_sh, 0.48)
        if wrist is None or shoulder is None:
            continue

        lift = shoulder[1] - wrist[1]
        if lift < min_lift:
            continue
        if abs(wrist[0] - shoulder[0]) > lift * 1.05:
            continue

        elbow = obs.kp(kp_el, 0.40)
        if elbow is not None:
            if wrist[1] > elbow[1] + 0.025 or elbow[1] > shoulder[1] + 0.03:
                continue

        if nose is not None and wrist[1] > nose[1] - 0.03:
            continue
        if min_head_y is not None and wrist[1] > min_head_y - 0.015:
            continue
        if baseline_shoulder_y is not None and baseline_shoulder_y > 0.0:
            if wrist[1] > baseline_shoulder_y - 0.10:
                continue

        raised_sides.append(side_ru)

    if len(raised_sides) == 2:
        return True, "Подняты обе руки вверх над головой"
    if len(raised_sides) == 1:
        return True, f"Поднята {raised_sides[0]} вверх над головой"
    return False, ""


def is_valid_human_pose(
    obs: PoseObservation,
    min_kp_conf: float = 0.30,
    min_valid_kps: int = 4,
    min_box_conf: float = 0.48,
    zones: Any = None,
) -> bool:
    """
    Анатомический фильтр против фантомных сработок YOLO на одеялах, подушках, куртках и вешалках:
    1) порог уверенности бокса >= min_box_conf (по умолчанию 0.48);
    2) игнорирование полигонов вещей (zones.ignore_polygons);
    3) минимум min_valid_kps достоверных ключевых точек и опорная анатомия корпуса;
    4) отсев «скелетов-вешалок» без уверенной головы (kpts 0..4 < 0.35) при узком/вертикальном силуэте вне кровати.
    """
    if obs.bbox_width < 0.02 or obs.bbox_height < 0.03:
        return False

    if zones is not None:
        ignore_polys = getattr(zones, "ignore_polygons", None)
        door_poly = getattr(zones, "door_polygon", None)
        if ignore_polys and is_pose_in_ignore_zones(obs, ignore_polys, door_polygon=door_poly):
            return False

    bed_poly = getattr(zones, "bed_polygon", ()) if zones is not None else ()
    has_bed = len(bed_poly) >= 3
    if obs.conf < min_box_conf:
        rich_seated_or_angled_person = (
            obs.conf >= 0.34
            and obs.valid_keypoint_count(0.65) >= 10
            and sum(
                1 for i in (KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR) if obs.kp(i, 0.65) is not None
            ) >= 2
            and obs.kp(KP_L_SHOULDER, 0.65) is not None
            and obs.kp(KP_R_SHOULDER, 0.65) is not None
            and (obs.kp(KP_L_HIP, 0.65) is not None or obs.kp(KP_R_HIP, 0.65) is not None)
        )
        if not (rich_seated_or_angled_person or (has_bed and is_blanket_sleeper_in_bed(obs, bed_poly))):
            return False

    if obs.valid_keypoint_count(min_kp_conf) < min_valid_kps:
        return False
    if not obs.has_core_anatomy(min_kp_conf):
        return False

    on_bed = is_person_on_bed(obs, bed_poly) if has_bed else False
    angle = obs.torso_angle_from_vertical_deg()
    is_standing_coat_posture = (
        angle is not None
        and angle < 28.0
        and obs.bbox_height >= 0.36
        and obs.aspect_ratio_wh < 0.68
    )
    sleeping_in_bed = on_bed and not is_standing_coat_posture
    has_arm_joint = any(
        obs.kp(i, 0.40) is not None for i in (KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST)
    )
    back_body_joints = sum(
        1
        for i in (KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST, KP_L_HIP, KP_R_HIP)
        if obs.kp(i, 0.40) is not None
    )
    back_turned_human = (
        obs.conf >= 0.52
        and obs.kp(KP_L_SHOULDER, 0.50) is not None
        and obs.kp(KP_R_SHOULDER, 0.50) is not None
        and has_arm_joint
        and back_body_joints >= 2
        and (obs.bbox_norm[3] >= 0.65 or obs.aspect_ratio_wh >= 0.45)
    )

    if not sleeping_in_bed and not obs.has_confident_head(0.35) and not back_turned_human:
        is_hanging_silhouette = obs.aspect_ratio_wh < 0.75 or (angle is not None and angle < 38.0)
        if is_hanging_silhouette:
            return False

    return True


def filter_valid_poses(
    observations: Sequence[PoseObservation],
    min_kp_conf: float = 0.30,
    min_valid_kps: int = 4,
    min_box_conf: float = 0.48,
    zones: Any = None,
) -> list[PoseObservation]:
    """Отбрасывает детекции без анатомии скелета, слабые боксы, скелеты-вешалки и зоны ignore_polygons."""
    return [
        obs
        for obs in observations
        if is_valid_human_pose(
            obs,
            min_kp_conf=min_kp_conf,
            min_valid_kps=min_valid_kps,
            min_box_conf=min_box_conf,
            zones=zones,
        )
    ]


def extract_torso_luma_grid(
    frame_bgr: np.ndarray | None,
    obs: PoseObservation,
    grid_w: int = 16,
    grid_h: int = 20,
) -> np.ndarray | None:
    """
    Строит сглаженную сетку яркости торса (zero-mean), устойчивую к высокочастотному ИК-шуму
    и скачкам автоэкспозиции камеры. Используется для проверки реального движения пикселей.
    """
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None
    h, w = frame_bgr.shape[:2]
    if h < 32 or w < 32:
        return None
    bx1, by1, bx2, by2 = obs.bbox_norm
    bw = bx2 - bx1
    bh = by2 - by1
    x1 = int(max(0, (bx1 + bw * 0.15) * w))
    x2 = int(min(w, (bx2 - bw * 0.15) * w))
    y1 = int(max(0, (by1 + bh * 0.12) * h))
    y2 = int(min(h, (by1 + bh * 0.72) * h))
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # INTER_AREA усредняет блоки пикселей и подавляет белый ИК-шум матрицы в ~6–8 раз
    small = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return small - float(np.mean(small))


@dataclass
class _LivenessState:
    first_ts: float
    last_ts: float
    spawn_center: tuple[float, float]
    spawn_shoulder: tuple[float, float] | None
    spawn_angle: float | None
    last_center: tuple[float, float] = (0.5, 0.5)
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    max_displacement: float = 0.0
    outside_bed_spawn: tuple[float, float] | None = None
    max_outside_bed_disp: float = 0.0
    frames_seen: int = 1
    spawned_in_door: bool = False
    was_on_bed: bool = False
    has_moved: bool = False
    prev_luma_grid: np.ndarray | None = None
    pixel_motion_hits: int = 0
    optical_frames_checked: int = 0


class StaticObjectGate:
    """
    Темпоральный фильтр живости (Temporal Liveness / Static Object Gate):
    Висящая на стене куртка/халат, вешалка, штора, стул, дверной косяк или спинка кровати никогда не ходят по комнате.
    Любой трек вне кровати (или стоящий вертикально angle < 28°), который с момента появления
    ни разу реально не сдвинулся с места (а при наличии реального видеокадра frame_bgr — и не подтвердил
    оптическое движение пикселей в ROI >= 0.012) — отсекается как неподвижная вещь / ИК-фантом.
    Дополнительно считает число подтверждённых кадров (frames_seen) и честное пространственное
    смещение центра строго ВНЕ полигона кровати и ВНЕ ignore_polygons (max_outside_bed_disp).
    """

    def __init__(
        self,
        min_displacement: float = 0.045,
        ttl_seconds: float = 12.0,
    ) -> None:
        self._min_disp = min_displacement
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._tracks: dict[int, _LivenessState] = {}
        self._prev_gray_small: np.ndarray | None = None
        self._prev_gray_ts: float = -1e9
        self._prev_scene_textured: bool | None = None
        self._last_frame_id: int | None = None

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._prev_gray_small = None
            self._prev_gray_ts = -1e9
            self._prev_scene_textured = None
            self._last_frame_id = None

    def is_track_moved(self, track_id: int) -> bool:
        with self._lock:
            st = self._tracks.get(track_id)
            return bool(st and st.has_moved)

    def get_spawn_center(self, track_id: int) -> tuple[float, float] | None:
        with self._lock:
            st = self._tracks.get(track_id)
            return st.spawn_center if st is not None else None

    def get_frames_seen(self, track_id: int) -> int:
        with self._lock:
            st = self._tracks.get(track_id)
            return int(st.frames_seen) if st is not None else 0

    def get_outside_bed_displacement(self, track_id: int) -> float:
        with self._lock:
            st = self._tracks.get(track_id)
            return float(st.max_outside_bed_disp) if st is not None else 0.0

    def is_spawned_in_door(self, track_id: int) -> bool:
        with self._lock:
            st = self._tracks.get(track_id)
            return bool(st and st.spawned_in_door)

    def update_and_filter(
        self,
        observations: Sequence[PoseObservation],
        zones: Any = None,
        now: float | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> list[PoseObservation]:
        ts = now if now is not None else time.monotonic()
        bed_poly = getattr(zones, "bed_polygon", ()) if zones is not None else ()
        door_poly = getattr(zones, "door_polygon", ()) if zones is not None else ()
        ignore_polys = getattr(zones, "ignore_polygons", None) if zones is not None else None
        has_bed = len(bed_poly) >= 3
        has_door = len(door_poly) >= 3

        with self._lock:
            verified_ttl = max(self._ttl, 120.0)
            stale = [
                tid
                for tid, st in self._tracks.items()
                if (ts - st.last_ts)
                > (
                    verified_ttl
                    if (st.has_moved and not (st.was_on_bed and not st.spawned_in_door))
                    else self._ttl
                )
            ]
            for tid in stale:
                del self._tracks[tid]

            cur_gray_small = OpticalKeypointStabilizer._to_gray_small(frame_bgr)
            same_frame_obj = frame_bgr is not None and id(frame_bgr) == self._last_frame_id
            prev_gray_valid = (
                self._prev_gray_small is not None
                and cur_gray_small is not None
                and self._prev_gray_small.shape == cur_gray_small.shape
                and 0.0 <= (ts - self._prev_gray_ts) <= 3.5
            )
            scene_textured = (
                int(np.ptp(cur_gray_small)) >= 8 if cur_gray_small is not None else False
            )
            if not same_frame_obj:
                scene_cut = (
                    self._prev_scene_textured is not None
                    and cur_gray_small is not None
                    and scene_textured != self._prev_scene_textured
                )
                if (
                    not scene_cut
                    and prev_gray_valid
                    and cur_gray_small is not None
                    and self._prev_gray_small is not None
                ):
                    cur_i16 = cur_gray_small.astype(np.int16)
                    prev_i16 = self._prev_gray_small.astype(np.int16)
                    mean_luma_jump = abs(float(np.mean(cur_i16)) - float(np.mean(prev_i16)))
                    global_diff = float(np.mean(np.abs(cur_i16 - prev_i16) > 24))
                    if mean_luma_jump > 12.0 or global_diff > 0.45:
                        scene_cut = True
                if scene_cut:
                    self._prev_gray_small = None
                    prev_gray_valid = False

            has_opt_pair = (
                scene_textured
                and prev_gray_valid
                and not same_frame_obj
                and self._prev_gray_small is not None
            )

            current_ids = {obs.track_id for obs in observations}
            live_obs: list[PoseObservation] = []
            for obs in observations:
                center = obs.body_center
                angle = obs.torso_angle_from_vertical_deg()
                on_bed = is_person_on_bed(obs, bed_poly, zones=zones) if has_bed else False
                in_door = has_door and is_pose_touching_door(obs, door_poly, min_overlap=0.20)
                in_ignore = (
                    is_pose_in_ignore_zones(obs, ignore_polys, door_polygon=door_poly)
                    if ignore_polys
                    else False
                )
                outside_bed_and_ignore = (not on_bed) and (not in_ignore)

                # Стоячий вертикальный силуэт во весь рост (характерен для висящей куртки/халата/косяка на стене)
                is_standing_coat_posture = (
                    angle is not None
                    and angle < 28.0
                    and obs.bbox_height >= 0.36
                    and obs.aspect_ratio_wh < 0.68
                )
                # Сидящий на матрасе человек с отчётливо видимыми коленями (не плоская куртка/вешалка)
                seated_with_knees_on_bed = (
                    on_bed
                    and obs.conf >= 0.65
                    and (obs.kp(KP_L_KNEE, 0.50) is not None or obs.kp(KP_R_KNEE, 0.50) is not None)
                )
                sleeping_or_sitting_in_bed = on_bed and (
                    not is_standing_coat_posture or seated_with_knees_on_bed
                )
                fg_seated_human = outside_bed_and_ignore and is_confident_foreground_seated_person(
                    obs,
                    bed_poly,
                    zones=zones,
                )

                luma_grid = extract_torso_luma_grid(frame_bgr, obs)
                st = self._tracks.get(obs.track_id)

                if st is None:
                    inherited_moved = bool(obs.liveness_verified)
                    inh_frames = max(1, int(obs.confirmed_frames))
                    inh_out_spawn: tuple[float, float] | None = center if outside_bed_and_ignore else None
                    inh_out_disp = float(obs.outside_bed_disp)
                    inh_in_door = bool(in_door)
                    if obs.spawn_center_hint is not None and has_door and point_in_polygon(obs.spawn_center_hint, door_poly):
                        inh_in_door = True

                    for prev_tid, prev_st in self._tracks.items():
                        if prev_tid in current_ids or not prev_st.has_moved:
                            continue
                        # Матрасный трек сна НИКОГДА не передаёт статус движения соседним фантомам/складкам/спинке кровати!
                        if prev_st.was_on_bed and not prev_st.spawned_in_door:
                            continue
                        if not (
                            prev_st.spawned_in_door
                            or (prev_st.frames_seen >= 4 and prev_st.max_outside_bed_disp >= 0.045)
                        ):
                            continue
                        if not (0.0 <= (ts - prev_st.last_ts) <= max(self._ttl, 120.0)):
                            continue
                        if (
                            bbox_iou(obs.bbox_norm, prev_st.last_bbox) >= 0.35
                            or math.hypot(center[0] - prev_st.last_center[0], center[1] - prev_st.last_center[1]) <= 0.12
                        ):
                            inherited_moved = True
                            if prev_st.spawned_in_door:
                                inh_in_door = True
                            inh_frames = max(
                                inh_frames,
                                prev_st.frames_seen,
                                4 if prev_st.spawned_in_door else 0,
                            )
                            inh_out_disp = max(
                                inh_out_disp,
                                prev_st.max_outside_bed_disp,
                                0.05 if prev_st.spawned_in_door else 0.0,
                            )
                            if prev_st.outside_bed_spawn is not None:
                                inh_out_spawn = prev_st.outside_bed_spawn
                            break

                    initial_spawn = obs.spawn_center_hint if obs.spawn_center_hint is not None else center
                    if outside_bed_and_ignore and inh_out_spawn is not None and (not scene_textured or inherited_moved):
                        init_out_d = math.hypot(center[0] - inh_out_spawn[0], center[1] - inh_out_spawn[1])
                        inh_out_disp = max(inh_out_disp, init_out_d)
                    st = _LivenessState(
                        first_ts=ts,
                        last_ts=ts,
                        spawn_center=initial_spawn,
                        spawn_shoulder=obs.mid_shoulder,
                        spawn_angle=angle,
                        last_center=center,
                        last_bbox=obs.bbox_norm,
                        outside_bed_spawn=inh_out_spawn,
                        max_outside_bed_disp=inh_out_disp,
                        frames_seen=inh_frames,
                        spawned_in_door=inh_in_door,
                        was_on_bed=sleeping_or_sitting_in_bed,
                        has_moved=inherited_moved,
                        prev_luma_grid=luma_grid,
                        pixel_motion_hits=2 if inherited_moved else 0,
                    )
                    self._tracks[obs.track_id] = st
                else:
                    prev_bbox = st.last_bbox
                    if ts > (st.last_ts + 1e-6):
                        st.frames_seen += 1
                    st.frames_seen = max(st.frames_seen, int(obs.confirmed_frames))
                    st.last_ts = ts
                    st.last_center = center
                    st.last_bbox = obs.bbox_norm
                    if in_door and not st.was_on_bed and st.frames_seen <= 6:
                        st.spawned_in_door = True
                    if obs.liveness_verified:
                        st.has_moved = True
                        st.pixel_motion_hits = max(st.pixel_motion_hits, 2)
                    if sleeping_or_sitting_in_bed:
                        st.was_on_bed = True

                    roi_opt = 0.0
                    luma_hit = False
                    if luma_grid is not None and st.prev_luma_grid is not None:
                        mean_abs_diff = float(np.mean(np.abs(luma_grid - st.prev_luma_grid)))
                        if mean_abs_diff >= 12.0:
                            luma_hit = True
                        st.prev_luma_grid = 0.7 * st.prev_luma_grid + 0.3 * luma_grid
                    elif luma_grid is not None and st.prev_luma_grid is None:
                        st.prev_luma_grid = luma_grid

                    if has_opt_pair:
                        # Строго безпараллаксная оценка движения в фиксированных координатах кадра:
                        # исключает ложный luma_hit от качания самой рамки YOLO по контрастному краю косяка/стены!
                        roi_opt = OpticalKeypointStabilizer.compute_roi_optical_motion(
                            cur_gray_small,
                            self._prev_gray_small,
                            obs.bbox_norm,
                            prev_bbox,
                        )
                        st.optical_frames_checked += 1
                        if roi_opt >= 0.030:
                            st.pixel_motion_hits += 2
                        elif roi_opt >= 0.012:
                            st.pixel_motion_hits += 1
                        elif roi_opt < 0.008 and not st.has_moved:
                            st.pixel_motion_hits = max(0, st.pixel_motion_hits - 1)
                    elif luma_hit and not same_frame_obj:
                        st.pixel_motion_hits += 1

                    optical_gate_active = (st.optical_frames_checked > 0) or (
                        scene_textured and not same_frame_obj
                    )
                    # На текстурированном видеокадре при нулевом оптическом движении внутри бокса
                    # НЕ позволяем ИК-шуму рамки YOLO накручивать ложное смещение центра!
                    pure_ir_bbox_jitter = (
                        has_opt_pair
                        and roi_opt < 0.012
                        and st.pixel_motion_hits == 0
                        and not st.has_moved
                    )
                    if not pure_ir_bbox_jitter:
                        disp = math.hypot(center[0] - st.spawn_center[0], center[1] - st.spawn_center[1])
                        if disp > st.max_displacement:
                            st.max_displacement = disp

                        if outside_bed_and_ignore:
                            if st.outside_bed_spawn is None:
                                st.outside_bed_spawn = center
                            else:
                                out_d = math.hypot(
                                    center[0] - st.outside_bed_spawn[0],
                                    center[1] - st.outside_bed_spawn[1],
                                )
                                if out_d > st.max_outside_bed_disp:
                                    st.max_outside_bed_disp = out_d
                        if obs.outside_bed_disp > st.max_outside_bed_disp:
                            st.max_outside_bed_disp = float(obs.outside_bed_disp)
                    elif outside_bed_and_ignore and st.outside_bed_spawn is None:
                        st.outside_bed_spawn = center

                    eff_min_disp = min(self._min_disp, 0.025) if in_door else self._min_disp
                    if st.max_displacement >= eff_min_disp and (
                        not optical_gate_active or st.pixel_motion_hits >= 2
                    ):
                        st.has_moved = True

                    # Проверка согласованной артикуляции позы + реального изменения пикселей (защита от ИК-шума)
                    if not st.has_moved:
                        pose_changed = False
                        if st.spawn_angle is not None and angle is not None and abs(angle - st.spawn_angle) >= 20.0:
                            pose_changed = True
                        if st.spawn_shoulder is not None and obs.mid_shoulder is not None:
                            s_disp = math.hypot(
                                obs.mid_shoulder[0] - st.spawn_shoulder[0],
                                obs.mid_shoulder[1] - st.spawn_shoulder[1],
                            )
                            if s_disp >= 0.055:
                                pose_changed = True

                        if optical_gate_active:
                            if pose_changed and st.pixel_motion_hits >= 2:
                                st.has_moved = True
                        elif frame_bgr is None and pose_changed:
                            st.has_moved = True

                # Пропускаем трек ТОЛЬКО если он доказал реальное движение, находится в кровати (не стоячей вешалкой)
                # или уверенно сидит на стуле ближнего переднего плана (оба плеча + таз, y1 >= 0.54, y2 >= 0.92).
                if st.has_moved or st.was_on_bed or sleeping_or_sitting_in_bed or fg_seated_human:
                    out_obs = PoseObservation(
                        track_id=obs.track_id,
                        bbox_norm=obs.bbox_norm,
                        conf=obs.conf,
                        keypoints_norm=obs.keypoints_norm,
                        liveness_verified=bool(st.has_moved or obs.liveness_verified or fg_seated_human),
                        spawn_center_hint=st.spawn_center,
                        confirmed_frames=st.frames_seen,
                        outside_bed_disp=round(st.max_outside_bed_disp, 5),
                        is_still=obs.is_still,
                        _kp_stabilized=obs._kp_stabilized,
                    )
                    live_obs.append(out_obs)
                elif not has_bed and not is_standing_coat_posture and (angle is None or angle >= 28.0):
                    live_obs.append(
                        PoseObservation(
                            track_id=obs.track_id,
                            bbox_norm=obs.bbox_norm,
                            conf=obs.conf,
                            keypoints_norm=obs.keypoints_norm,
                            liveness_verified=obs.liveness_verified,
                            spawn_center_hint=st.spawn_center,
                            confirmed_frames=st.frames_seen,
                            outside_bed_disp=round(st.max_outside_bed_disp, 5),
                            is_still=obs.is_still,
                            _kp_stabilized=obs._kp_stabilized,
                        )
                    )

            if cur_gray_small is not None and not same_frame_obj:
                self._prev_gray_small = cur_gray_small
                self._prev_gray_ts = ts
                self._prev_scene_textured = scene_textured
                self._last_frame_id = id(frame_bgr)
            elif same_frame_obj and ts > self._prev_gray_ts:
                self._prev_gray_ts = ts

            return live_obs


@dataclass
class _TrackSmoothState:
    last_ts: float
    bbox: list[float]
    kp_xy: list[list[float]]
    kp_conf: list[float]
    kp_valid: list[bool]
    kp_coast: list[int]


class KeypointSmoother:
    """
    Адаптивный экспоненциальный фильтр ключевых точек и рамок по track_id.
    Сильно подавляет микродрожание YOLO11-pose в плохом освещении и остром ракурсе,
    но мгновенно пропускает быстрые движения (падение / рывок) без задержки реакции.
    """

    def __init__(
        self,
        alpha_min: float = 0.24,
        alpha_max: float = 0.94,
        jitter_threshold: float = 0.008,
        fast_threshold: float = 0.045,
        min_kp_conf: float = 0.22,
        max_coast_frames: int = 2,
        ttl_seconds: float = 5.0,
    ) -> None:
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._jitter_thr = jitter_threshold
        self._fast_thr = max(jitter_threshold + 1e-4, fast_threshold)
        self._min_kp_conf = min_kp_conf
        self._max_coast = max_coast_frames
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._states: dict[int, _TrackSmoothState] = {}

    def _adaptive_alpha(self, displacement: float) -> float:
        if displacement <= self._jitter_thr:
            return self._alpha_min
        if displacement >= self._fast_thr:
            return self._alpha_max
        ratio = (displacement - self._jitter_thr) / (self._fast_thr - self._jitter_thr)
        return self._alpha_min + ratio * (self._alpha_max - self._alpha_min)

    def reset(self) -> None:
        with self._lock:
            self._states.clear()

    def smooth_one(self, obs: PoseObservation, now: float | None = None) -> PoseObservation:
        return self.smooth([obs], now=now)[0]

    def smooth(
        self,
        observations: Sequence[PoseObservation],
        now: float | None = None,
    ) -> list[PoseObservation]:
        ts = now if now is not None else time.monotonic()
        with self._lock:
            stale = [tid for tid, st in self._states.items() if (ts - st.last_ts) > self._ttl]
            for tid in stale:
                del self._states[tid]

            result: list[PoseObservation] = []
            for obs in observations:
                st = self._states.get(obs.track_id)
                bx1, by1, bx2, by2 = obs.bbox_norm
                raw_cx = (bx1 + bx2) * 0.5
                raw_cy = (by1 + by2) * 0.5

                if st is not None:
                    prev_cx = (st.bbox[0] + st.bbox[2]) * 0.5
                    prev_cy = (st.bbox[1] + st.bbox[3]) * 0.5
                    if math.hypot(raw_cx - prev_cx, raw_cy - prev_cy) > 0.28:
                        st = None

                if st is None:
                    kp_xy: list[list[float]] = []
                    kp_conf: list[float] = []
                    kp_valid: list[bool] = []
                    kp_coast: list[int] = []
                    for k in range(17):
                        if k < len(obs.keypoints_norm):
                            kx, ky, kc = obs.keypoints_norm[k]
                        else:
                            kx, ky, kc = 0.0, 0.0, 0.0
                        valid = kc >= self._min_kp_conf and (kx > 0.0 or ky > 0.0)
                        kp_xy.append([kx, ky])
                        kp_conf.append(kc)
                        kp_valid.append(valid)
                        kp_coast.append(0)
                    self._states[obs.track_id] = _TrackSmoothState(
                        last_ts=ts,
                        bbox=[bx1, by1, bx2, by2],
                        kp_xy=kp_xy,
                        kp_conf=kp_conf,
                        kp_valid=kp_valid,
                        kp_coast=kp_coast,
                    )
                    result.append(obs)
                    continue

                st.last_ts = ts
                box_disp = max(
                    math.hypot(bx1 - st.bbox[0], by1 - st.bbox[1]),
                    math.hypot(bx2 - st.bbox[2], by2 - st.bbox[3]),
                )
                a_box = self._adaptive_alpha(box_disp)
                for idx_b, raw_val in enumerate((bx1, by1, bx2, by2)):
                    st.bbox[idx_b] = max(0.0, min(1.0, st.bbox[idx_b] + a_box * (raw_val - st.bbox[idx_b])))

                smoothed_kps: list[tuple[float, float, float]] = []
                for k in range(17):
                    if k < len(obs.keypoints_norm):
                        kx, ky, kc = obs.keypoints_norm[k]
                    else:
                        kx, ky, kc = 0.0, 0.0, 0.0

                    is_valid = kc >= self._min_kp_conf and (kx > 0.0 or ky > 0.0)
                    if is_valid:
                        if st.kp_valid[k]:
                            disp = math.hypot(kx - st.kp_xy[k][0], ky - st.kp_xy[k][1])
                            alpha = self._adaptive_alpha(disp)
                            nx = max(0.0, min(1.0, st.kp_xy[k][0] + alpha * (kx - st.kp_xy[k][0])))
                            ny = max(0.0, min(1.0, st.kp_xy[k][1] + alpha * (ky - st.kp_xy[k][1])))
                        else:
                            nx, ny = kx, ky
                        st.kp_xy[k][0] = nx
                        st.kp_xy[k][1] = ny
                        st.kp_conf[k] = kc
                        st.kp_valid[k] = True
                        st.kp_coast[k] = 0
                        smoothed_kps.append((nx, ny, kc))
                    elif (
                        st.kp_valid[k]
                        and st.kp_conf[k] >= 0.42
                        and st.kp_coast[k] < self._max_coast
                    ):
                        st.kp_coast[k] += 1
                        coasted_conf = max(0.31, st.kp_conf[k] * 0.82)
                        st.kp_conf[k] = coasted_conf
                        smoothed_kps.append((st.kp_xy[k][0], st.kp_xy[k][1], coasted_conf))
                    else:
                        st.kp_valid[k] = False
                        st.kp_coast[k] = 0
                        st.kp_conf[k] = 0.0
                        smoothed_kps.append((0.0, 0.0, 0.0))

                result.append(
                    PoseObservation(
                        track_id=obs.track_id,
                        bbox_norm=(st.bbox[0], st.bbox[1], st.bbox[2], st.bbox[3]),
                        conf=obs.conf,
                        keypoints_norm=tuple(smoothed_kps),
                        liveness_verified=obs.liveness_verified,
                        spawn_center_hint=obs.spawn_center_hint,
                        confirmed_frames=obs.confirmed_frames,
                        outside_bed_disp=obs.outside_bed_disp,
                        is_still=obs.is_still,
                        _kp_stabilized=obs._kp_stabilized,
                    )
                )
            return result


@dataclass
class _OpticalTrackState:
    last_ts: float
    last_detected_ts: float
    on_bed_reclining: bool
    last_conf: float
    liveness_verified: bool
    spawn_center_hint: tuple[float, float] | None
    confirmed_frames: int
    outside_bed_disp: float
    bbox: list[float]
    kp_xy: list[list[float]]
    kp_pub_conf: list[float]
    kp_ui_visible: list[bool]
    kp_anat_active: list[bool]
    kp_drop_streak: list[int]
    kp_birth_streak: list[int]
    kp_cand_xy: list[list[float]]
    anchor_roi_blur: np.ndarray | None = None
    anchor_frame_mean: float = 0.0
    coast_motion_streak: int = 0


class OpticalKeypointStabilizer:
    """
    Оптически управляемый стабилизатор ключевых точек и бокса для ночного ИК-режима (Волна 6.1):
    - Считает реальную оптическую энергию движения внутри ROI бокса (48x36 после INTER_AREA + 3x3 GaussianBlur,
      доля пикселей |I_t - I_{t-1}| > 10) по фиксированным координатам кадра без параллакса самого бокса.
    - При optical_motion_ratio < 0.012 (человек спит неподвижно, шумит только ИК-матрица):
      намертво замораживает keypoints_norm и bbox_norm (0.0 сдвига, deadzone = 0.018), удерживает гистерезис
      видимости в UI (enter_conf = 0.54, keep_conf = 0.40, >= 3 стабильных кадра для новой точки на кровати),
      но СОХРАНЯЕТ замороженными анатомические точки 0.30 <= kc < 0.54 с pub_conf <= 0.47 (невидимо в UI < 0.48,
      но 100% валидно для filter_valid_poses(min_kp_conf=0.30, min_valid_kps=4) в PatientIdentifier и FSM!).
    - Оптически привязанный coast (bed_coast_ttl = 3600 с): удерживает спящий трек при пропусках YOLO,
      пока одновременно межкадровое движение < 0.012 и кумулятивное отличие от anchor_roi_blur < 0.050.
    - При реальном движении (optical_motion_ratio >= 0.012, anchor_drift >= 0.050 или прорыв >= 0.045 с conf >= 0.60):
      мгновенно сбрасывает coast и раскрывает скоростной коэффициент alpha -> 0.65..0.92 с нулевой задержкой
      для SITTING_UP, HELP_CALL, LEFT_BED, MISSING и FALLEN (obs.is_still = False).
    """

    _SMALL_W = 384
    _SMALL_H = 216
    _ROI_W = 48
    _ROI_H = 36

    def __init__(
        self,
        motion_ratio_threshold: float = 0.012,
        deadzone: float = 0.018,
        enter_conf: float = 0.54,
        keep_conf: float = 0.40,
        min_publish_conf: float = 0.48,
        min_anatomy_conf: float = 0.30,
        bed_enter_frames: int = 3,
        fast_breakout_disp: float = 0.045,
        fast_breakout_conf: float = 0.60,
        ttl_seconds: float = 8.0,
        bed_coast_ttl: float = 3600.0,
        anchor_drift_threshold: float = 0.050,
    ) -> None:
        self._motion_thr = float(motion_ratio_threshold)
        self._deadzone = float(deadzone)
        self._enter_conf = float(enter_conf)
        self._keep_conf = float(keep_conf)
        self._min_pub_conf = float(min_publish_conf)
        self._min_anat_conf = float(min_anatomy_conf)
        self._bed_enter_frames = max(2, int(bed_enter_frames))
        self._fast_disp = float(fast_breakout_disp)
        self._fast_conf = float(fast_breakout_conf)
        self._bed_coast_ttl = max(0.0, float(bed_coast_ttl))
        self._anchor_drift_thr = float(anchor_drift_threshold)
        self._ttl = (
            max(float(ttl_seconds), self._bed_coast_ttl + 1.0)
            if self._bed_coast_ttl > 0.0
            else float(ttl_seconds)
        )
        self._lock = threading.RLock()
        self._states: dict[int, _OpticalTrackState] = {}
        self._prev_gray_small: np.ndarray | None = None
        self._prev_gray_ts: float = -1e9
        self._prev_scene_textured: bool | None = None
        self._last_frame_id: int | None = None

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._prev_gray_small = None
            self._prev_gray_ts = -1e9
            self._prev_scene_textured = None
            self._last_frame_id = None

    @classmethod
    def _to_gray_small(cls, frame_bgr: np.ndarray | None) -> np.ndarray | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if frame_bgr.ndim == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr
        if gray.shape[1] == cls._SMALL_W and gray.shape[0] == cls._SMALL_H:
            return gray.copy()
        return cv2.resize(gray, (cls._SMALL_W, cls._SMALL_H), interpolation=cv2.INTER_AREA)

    @classmethod
    def _extract_blurred_roi(
        cls,
        gray_small: np.ndarray | None,
        box_norm: tuple[float, float, float, float] | Sequence[float],
    ) -> np.ndarray | None:
        if gray_small is None or gray_small.size == 0:
            return None
        sh, sw = gray_small.shape[:2]
        ux1_f = max(0.0, float(box_norm[0]) - 0.02)
        uy1_f = max(0.0, float(box_norm[1]) - 0.02)
        ux2_f = min(1.0, float(box_norm[2]) + 0.02)
        uy2_f = min(1.0, float(box_norm[3]) + 0.02)

        x1 = int(round(ux1_f * sw))
        y1 = int(round(uy1_f * sh))
        x2 = int(round(ux2_f * sw))
        y2 = int(round(uy2_f * sh))
        if (x2 - x1) < 12:
            cx = (x1 + x2) // 2
            x1 = max(0, cx - 6)
            x2 = min(sw, x1 + 12)
        if (y2 - y1) < 12:
            cy = (y1 + y2) // 2
            y1 = max(0, cy - 6)
            y2 = min(sh, y1 + 12)
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return None
        crop = gray_small[y1:y2, x1:x2]
        roi = cv2.resize(crop, (cls._ROI_W, cls._ROI_H), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(roi, (3, 3), 0)

    @classmethod
    def compute_anchor_drift(
        cls,
        cur_gray_small: np.ndarray | None,
        anchor_roi_blur: np.ndarray | None,
        anchor_frame_mean: float,
        box_norm: tuple[float, float, float, float] | Sequence[float],
    ) -> float:
        """Кумулятивное оптическое отличие текущего патча кровати от эталона сна (с компенсацией общей экспозиции)."""
        if cur_gray_small is None or anchor_roi_blur is None:
            return 0.0
        cur_roi_blur = cls._extract_blurred_roi(cur_gray_small, box_norm)
        if cur_roi_blur is None or cur_roi_blur.shape != anchor_roi_blur.shape:
            return 0.0
        frame_shift = max(-10.0, min(10.0, float(np.mean(cur_gray_small)) - float(anchor_frame_mean)))
        adj_diff = np.abs(
            (cur_roi_blur.astype(np.float32) - frame_shift) - anchor_roi_blur.astype(np.float32)
        )
        return float(np.mean(adj_diff > 12.0))

    @classmethod
    def compute_roi_optical_motion(
        cls,
        cur_gray_small: np.ndarray | None,
        prev_gray_small: np.ndarray | None,
        box_a: tuple[float, float, float, float] | Sequence[float],
        box_b: tuple[float, float, float, float] | Sequence[float],
    ) -> float:
        """
        Вычисляет долю пикселей с |I_t - I_{t-1}| > 10 после даунсэмпла 48x36 и 3x3 blur
        по фиксированному объединённому окну кадра (без паразитного сдвига самого окна от джиттера YOLO).
        """
        if cur_gray_small is None or prev_gray_small is None:
            return 0.0
        if cur_gray_small.shape != prev_gray_small.shape:
            return 0.0

        sh, sw = cur_gray_small.shape[:2]
        ux1_f = max(0.0, min(float(box_a[0]), float(box_b[0])) - 0.02)
        uy1_f = max(0.0, min(float(box_a[1]), float(box_b[1])) - 0.02)
        ux2_f = min(1.0, max(float(box_a[2]), float(box_b[2])) + 0.02)
        uy2_f = min(1.0, max(float(box_a[3]), float(box_b[3])) + 0.02)

        x1 = int(round(ux1_f * sw))
        y1 = int(round(uy1_f * sh))
        x2 = int(round(ux2_f * sw))
        y2 = int(round(uy2_f * sh))
        if (x2 - x1) < 12:
            cx = (x1 + x2) // 2
            x1 = max(0, cx - 6)
            x2 = min(sw, x1 + 12)
        if (y2 - y1) < 12:
            cy = (y1 + y2) // 2
            y1 = max(0, cy - 6)
            y2 = min(sh, y1 + 12)
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return 0.0

        cur_crop = cur_gray_small[y1:y2, x1:x2]
        prev_crop = prev_gray_small[y1:y2, x1:x2]
        cur_roi = cv2.resize(cur_crop, (cls._ROI_W, cls._ROI_H), interpolation=cv2.INTER_AREA)
        prev_roi = cv2.resize(prev_crop, (cls._ROI_W, cls._ROI_H), interpolation=cv2.INTER_AREA)
        cur_blur = cv2.GaussianBlur(cur_roi, (3, 3), 0)
        prev_blur = cv2.GaussianBlur(prev_roi, (3, 3), 0)
        diff = np.abs(cur_blur.astype(np.int16) - prev_blur.astype(np.int16))
        return float(np.mean(diff > 10))

    def is_box_optically_still(
        self,
        frame_bgr: np.ndarray | None,
        box_norm: tuple[float, float, float, float] | Sequence[float],
    ) -> bool:
        """Проверяет, что внутри заданного бокса нет реального оптического движения (>= 0.012) относительно прошлого кадра."""
        with self._lock:
            if frame_bgr is None or self._prev_gray_small is None:
                return True
            if id(frame_bgr) == self._last_frame_id:
                return True
            cur_gray_small = self._to_gray_small(frame_bgr)
            if cur_gray_small is None or cur_gray_small.shape != self._prev_gray_small.shape:
                return True
            if int(np.ptp(cur_gray_small)) < 8:
                return True
            cur_i16 = cur_gray_small.astype(np.int16)
            prev_i16 = self._prev_gray_small.astype(np.int16)
            if abs(float(np.mean(cur_i16)) - float(np.mean(prev_i16))) > 12.0:
                return False
            if float(np.mean(np.abs(cur_i16 - prev_i16) > 24)) > 0.45:
                return False
            return (
                self.compute_roi_optical_motion(
                    cur_gray_small, self._prev_gray_small, box_norm, box_norm
                )
                < self._motion_thr
            )

    @staticmethod
    def _is_reclining_on_bed(obs: PoseObservation, zones: Any) -> bool:
        bed_poly = getattr(zones, "bed_polygon", None) if zones is not None else None
        if not bed_poly or len(bed_poly) < 3:
            return False
        if not is_person_on_bed(obs, bed_poly, zones=zones):
            return False
        if are_legs_hanging_outside_bed(obs, bed_poly):
            return False
        bed_min_y, bed_max_y = polygon_y_bounds(bed_poly)
        contained_on_mattress = (
            bbox_polygon_overlap_ratio(obs.bbox_norm, bed_poly) >= 0.68
            and obs.bbox_norm[1] >= (bed_min_y - 0.03)
            and obs.bbox_norm[3] <= (bed_max_y + 0.025)
            and obs.bbox_height <= 0.34
        )
        angle = obs.torso_angle_from_vertical_deg()
        if not contained_on_mattress:
            if angle is not None and angle < 42.0 and obs.bbox_height >= 0.30 and obs.aspect_ratio_wh < 0.85:
                return False
        elif angle is not None and angle < 25.0 and obs.torso_vertical_rise() >= 0.14:
            return False
        return True

    def stabilize(
        self,
        observations: Sequence[PoseObservation],
        frame_bgr: np.ndarray | None = None,
        zones: Any = None,
        now: float | None = None,
    ) -> list[PoseObservation]:
        ts = float(now) if now is not None else time.monotonic()
        with self._lock:
            stale = [tid for tid, st in self._states.items() if (ts - st.last_ts) > self._ttl]
            for tid in stale:
                del self._states[tid]

            cur_gray_small = self._to_gray_small(frame_bgr)
            same_frame_obj = frame_bgr is not None and id(frame_bgr) == self._last_frame_id
            prev_gray_valid = (
                self._prev_gray_small is not None
                and cur_gray_small is not None
                and self._prev_gray_small.shape == cur_gray_small.shape
                and 0.0 <= (ts - self._prev_gray_ts) <= 3.5
            )
            scene_textured = (
                int(np.ptp(cur_gray_small)) >= 8 if cur_gray_small is not None else False
            )
            if not same_frame_obj:
                scene_cut = (
                    self._prev_scene_textured is not None
                    and cur_gray_small is not None
                    and scene_textured != self._prev_scene_textured
                )
                if not scene_cut and prev_gray_valid and cur_gray_small is not None and self._prev_gray_small is not None:
                    cur_i16 = cur_gray_small.astype(np.int16)
                    prev_i16 = self._prev_gray_small.astype(np.int16)
                    mean_luma_jump = abs(float(np.mean(cur_i16)) - float(np.mean(prev_i16)))
                    global_diff = float(np.mean(np.abs(cur_i16 - prev_i16) > 24))
                    if mean_luma_jump > 12.0 or global_diff > 0.45:
                        scene_cut = True
                if scene_cut:
                    self._states.clear()
                    prev_gray_valid = False

            incoming_tids = {obs.track_id for obs in observations}
            stabilized_out: list[PoseObservation] = []
            for obs in observations:
                bx1, by1, bx2, by2 = obs.bbox_norm
                raw_cx = (bx1 + bx2) * 0.5
                raw_cy = (by1 + by2) * 0.5
                on_bed_reclining = self._is_reclining_on_bed(obs, zones)

                # Поддержка экстремально глубокого укрывания одеялом (conf бокса < 0.20, видны только точки головы/плеч 0..6)
                confident_54_count = sum(
                    1
                    for k in range(min(17, len(obs.keypoints_norm)))
                    if obs.keypoints_norm[k][2] >= self._enter_conf
                    and (obs.keypoints_norm[k][0] > 0.0 or obs.keypoints_norm[k][1] > 0.0)
                )
                deep_blanket_head_mode = on_bed_reclining and obs.conf < 0.22 and confident_54_count < 4

                st = self._states.get(obs.track_id)
                if st is not None:
                    prev_cx = (st.bbox[0] + st.bbox[2]) * 0.5
                    prev_cy = (st.bbox[1] + st.bbox[3]) * 0.5
                    if math.hypot(raw_cx - prev_cx, raw_cy - prev_cy) > 0.26:
                        st = None
                elif on_bed_reclining:
                    # Бесшовное переназначение замороженного состояния при смене track_id у спящего трека в кровати
                    adopt_tid: int | None = None
                    for cand_tid, cand_st in self._states.items():
                        if cand_tid in incoming_tids or not cand_st.on_bed_reclining:
                            continue
                        cand_cx = (cand_st.bbox[0] + cand_st.bbox[2]) * 0.5
                        cand_cy = (cand_st.bbox[1] + cand_st.bbox[3]) * 0.5
                        if (
                            bbox_iou(obs.bbox_norm, (cand_st.bbox[0], cand_st.bbox[1], cand_st.bbox[2], cand_st.bbox[3])) >= 0.25
                            or math.hypot(raw_cx - cand_cx, raw_cy - cand_cy) <= 0.18
                        ):
                            adopt_tid = cand_tid
                            break
                    if adopt_tid is not None:
                        st = self._states.pop(adopt_tid)
                        st.coast_motion_streak = 0
                        if st.anchor_roi_blur is None and cur_gray_small is not None and scene_textured:
                            st.anchor_roi_blur = self._extract_blurred_roi(cur_gray_small, (bx1, by1, bx2, by2))
                            st.anchor_frame_mean = float(np.mean(cur_gray_small))
                        self._states[obs.track_id] = st

                # Инициализация нового трека:
                # - уверенные точки >= eff_enter публикуются в UI (>= 0.48);
                # - анатомические точки 0.30 <= kc < eff_enter сохраняются с pub_conf <= 0.47 для PatientIdentifier/FSM
                #   (невидимы в UI/HUD < 0.48, но исключают голодание filter_valid_poses под одеялом!).
                if st is None:
                    kp_xy: list[list[float]] = []
                    kp_pub_conf: list[float] = []
                    kp_ui_vis: list[bool] = []
                    kp_anat: list[bool] = []
                    kp_drop: list[int] = []
                    kp_birth: list[int] = []
                    kp_cand: list[list[float]] = []
                    init_pts: list[tuple[float, float, float]] = []
                    sub_ui_cap = round(self._min_pub_conf - 0.01, 4)

                    for k in range(17):
                        if k < len(obs.keypoints_norm):
                            kx, ky, kc = obs.keypoints_norm[k]
                        else:
                            kx, ky, kc = 0.0, 0.0, 0.0
                        kx = max(0.0, min(1.0, float(kx)))
                        ky = max(0.0, min(1.0, float(ky)))
                        kc = float(kc)
                        non_zero = kx > 0.0 or ky > 0.0

                        eff_enter = (
                            0.32
                            if (deep_blanket_head_mode and k <= 6)
                            else (self._enter_conf if on_bed_reclining else self._min_pub_conf)
                        )
                        if non_zero and kc >= eff_enter:
                            pub_c = round(kc, 4)
                            kp_xy.append([kx, ky])
                            kp_pub_conf.append(pub_c)
                            kp_ui_vis.append(True)
                            kp_anat.append(True)
                            kp_birth.append(self._bed_enter_frames)
                            init_pts.append((kx, ky, pub_c))
                        elif non_zero and kc >= self._min_anat_conf:
                            pub_c = round(min(sub_ui_cap, kc), 4)
                            kp_xy.append([kx, ky])
                            kp_pub_conf.append(pub_c)
                            kp_ui_vis.append(False)
                            kp_anat.append(True)
                            kp_birth.append(1 if kc >= self._enter_conf else 0)
                            init_pts.append((kx, ky, pub_c))
                        else:
                            kp_xy.append([kx, ky])
                            kp_pub_conf.append(0.0)
                            kp_ui_vis.append(False)
                            kp_anat.append(False)
                            kp_birth.append(0)
                            init_pts.append((0.0, 0.0, 0.0))
                        kp_drop.append(0)
                        kp_cand.append([kx, ky])

                    init_still = bool(on_bed_reclining)
                    init_anchor_roi = (
                        self._extract_blurred_roi(cur_gray_small, (bx1, by1, bx2, by2))
                        if (on_bed_reclining and cur_gray_small is not None and scene_textured)
                        else None
                    )
                    init_frame_mean = float(np.mean(cur_gray_small)) if cur_gray_small is not None else 0.0
                    self._states[obs.track_id] = _OpticalTrackState(
                        last_ts=ts,
                        last_detected_ts=ts,
                        on_bed_reclining=on_bed_reclining,
                        last_conf=obs.conf,
                        liveness_verified=obs.liveness_verified,
                        spawn_center_hint=obs.spawn_center_hint,
                        confirmed_frames=obs.confirmed_frames,
                        outside_bed_disp=obs.outside_bed_disp,
                        bbox=[bx1, by1, bx2, by2],
                        kp_xy=kp_xy,
                        kp_pub_conf=kp_pub_conf,
                        kp_ui_visible=kp_ui_vis,
                        kp_anat_active=kp_anat,
                        kp_drop_streak=kp_drop,
                        kp_birth_streak=kp_birth,
                        kp_cand_xy=kp_cand,
                        anchor_roi_blur=init_anchor_roi,
                        anchor_frame_mean=init_frame_mean,
                    )
                    stabilized_out.append(
                        PoseObservation(
                            track_id=obs.track_id,
                            bbox_norm=(bx1, by1, bx2, by2),
                            conf=obs.conf,
                            keypoints_norm=tuple(init_pts),
                            liveness_verified=obs.liveness_verified,
                            spawn_center_hint=obs.spawn_center_hint,
                            confirmed_frames=obs.confirmed_frames,
                            outside_bed_disp=obs.outside_bed_disp,
                            is_still=init_still,
                            _kp_stabilized=True,
                        )
                    )
                    continue

                st.last_ts = ts
                st.last_detected_ts = ts
                st.on_bed_reclining = on_bed_reclining
                st.last_conf = obs.conf
                st.liveness_verified = obs.liveness_verified
                st.spawn_center_hint = obs.spawn_center_hint
                st.confirmed_frames = obs.confirmed_frames
                st.outside_bed_disp = obs.outside_bed_disp
                st.coast_motion_streak = 0
                sub_ui_cap = round(self._min_pub_conf - 0.01, 4)
                opt_ratio = (
                    0.0
                    if same_frame_obj
                    else self.compute_roi_optical_motion(
                        cur_gray_small,
                        self._prev_gray_small if prev_gray_valid else None,
                        obs.bbox_norm,
                        st.bbox,
                    )
                )

                # Оцениваем геометрический сдвиг уверенных ключевых точек относительно замороженного состояния
                max_active_disp = 0.0
                has_fast_breakout = False
                for k in range(min(17, len(obs.keypoints_norm))):
                    kx, ky, kc = obs.keypoints_norm[k]
                    if kc <= 0.0 or (kx <= 0.0 and ky <= 0.0):
                        continue
                    if st.kp_ui_visible[k] or st.kp_anat_active[k]:
                        d_k = math.hypot(float(kx) - st.kp_xy[k][0], float(ky) - st.kp_xy[k][1])
                        if (st.kp_ui_visible[k] or float(kc) >= 0.35) and d_k > max_active_disp:
                            max_active_disp = d_k
                        if d_k >= self._fast_disp and float(kc) >= self._fast_conf:
                            has_fast_breakout = True
                    elif float(kc) >= self._fast_conf and k in (
                        KP_L_WRIST,
                        KP_R_WRIST,
                        KP_L_KNEE,
                        KP_R_KNEE,
                        KP_L_ANKLE,
                        KP_R_ANKLE,
                    ):
                        # Резкое появление высокоуверенной конечности (взмах руки HELP_CALL / свешивание ног)
                        has_fast_breakout = True

                if cur_gray_small is not None and prev_gray_valid and scene_textured:
                    in_motion = (opt_ratio >= self._motion_thr) or (
                        opt_ratio >= 0.005 and has_fast_breakout
                    )
                else:
                    # Вызов без текстурированного видеокадра (синтетические тесты): геометрическая мёртвая зона
                    in_motion = (max_active_disp >= self._deadzone) or has_fast_breakout

                # РЕЖИМ 1: Полная оптическая неподвижность (is_still = True) — 0.0 сдвига и 0 миганий костей!
                if not in_motion:
                    if on_bed_reclining and st.anchor_roi_blur is None and cur_gray_small is not None and scene_textured:
                        st.anchor_roi_blur = self._extract_blurred_roi(cur_gray_small, st.bbox)
                        st.anchor_frame_mean = float(np.mean(cur_gray_small))

                    anat_count_now = sum(1 for flag in st.kp_anat_active if flag)
                    frozen_pts: list[tuple[float, float, float]] = []
                    for k in range(17):
                        if k < len(obs.keypoints_norm):
                            kx, ky, kc = obs.keypoints_norm[k]
                        else:
                            kx, ky, kc = 0.0, 0.0, 0.0
                        kx = max(0.0, min(1.0, float(kx)))
                        ky = max(0.0, min(1.0, float(ky)))
                        kc = float(kc)
                        non_zero = kx > 0.0 or ky > 0.0

                        eff_keep = 0.30 if (deep_blanket_head_mode and k <= 6) else self._keep_conf
                        if st.kp_ui_visible[k]:
                            if non_zero and kc >= eff_keep:
                                st.kp_drop_streak[k] = 0
                                frozen_pts.append((st.kp_xy[k][0], st.kp_xy[k][1], st.kp_pub_conf[k]))
                            elif st.kp_drop_streak[k] < 3:
                                # Не даём случайному однокадровому ИК-провалу погасить видимую кость в покое
                                st.kp_drop_streak[k] += 1
                                frozen_pts.append((st.kp_xy[k][0], st.kp_xy[k][1], st.kp_pub_conf[k]))
                            elif non_zero and kc >= self._min_anat_conf:
                                st.kp_ui_visible[k] = False
                                st.kp_anat_active[k] = True
                                st.kp_drop_streak[k] = 0
                                st.kp_pub_conf[k] = round(min(sub_ui_cap, kc), 4)
                                frozen_pts.append((st.kp_xy[k][0], st.kp_xy[k][1], st.kp_pub_conf[k]))
                            else:
                                st.kp_ui_visible[k] = False
                                st.kp_anat_active[k] = False
                                st.kp_drop_streak[k] = 0
                                st.kp_pub_conf[k] = 0.0
                                frozen_pts.append((0.0, 0.0, 0.0))
                        else:
                            # Повышение в UI допускается только после >= 3 стабильных кадров подряд без джиттера (<= 0.010)
                            promoted_to_ui = False
                            if (
                                non_zero
                                and kc >= self._enter_conf
                                and (not on_bed_reclining or opt_ratio >= 0.006)
                            ):
                                cand_d = math.hypot(kx - st.kp_cand_xy[k][0], ky - st.kp_cand_xy[k][1])
                                if st.kp_birth_streak[k] == 0 or cand_d <= 0.010:
                                    st.kp_birth_streak[k] += 1
                                else:
                                    st.kp_birth_streak[k] = 1
                                st.kp_cand_xy[k] = [kx, ky]
                                req_frames = self._bed_enter_frames if on_bed_reclining else 2
                                if st.kp_birth_streak[k] >= req_frames:
                                    pub_c = round(kc, 4)
                                    st.kp_xy[k] = [kx, ky]
                                    st.kp_pub_conf[k] = pub_c
                                    st.kp_ui_visible[k] = True
                                    st.kp_anat_active[k] = True
                                    st.kp_drop_streak[k] = 0
                                    frozen_pts.append((kx, ky, pub_c))
                                    promoted_to_ui = True
                            else:
                                st.kp_birth_streak[k] = 0

                            if not promoted_to_ui:
                                if st.kp_anat_active[k]:
                                    if non_zero and kc >= 0.25:
                                        st.kp_drop_streak[k] = 0
                                        frozen_pts.append((st.kp_xy[k][0], st.kp_xy[k][1], st.kp_pub_conf[k]))
                                    elif st.kp_drop_streak[k] < 3:
                                        st.kp_drop_streak[k] += 1
                                        frozen_pts.append((st.kp_xy[k][0], st.kp_xy[k][1], st.kp_pub_conf[k]))
                                    else:
                                        st.kp_anat_active[k] = False
                                        st.kp_drop_streak[k] = 0
                                        st.kp_pub_conf[k] = 0.0
                                        frozen_pts.append((0.0, 0.0, 0.0))
                                elif anat_count_now < 5 and non_zero and kc >= self._min_anat_conf:
                                    pub_c = round(min(sub_ui_cap, kc), 4)
                                    st.kp_xy[k] = [kx, ky]
                                    st.kp_pub_conf[k] = pub_c
                                    st.kp_anat_active[k] = True
                                    st.kp_drop_streak[k] = 0
                                    anat_count_now += 1
                                    frozen_pts.append((kx, ky, pub_c))
                                else:
                                    frozen_pts.append((0.0, 0.0, 0.0))

                    stabilized_out.append(
                        PoseObservation(
                            track_id=obs.track_id,
                            bbox_norm=(st.bbox[0], st.bbox[1], st.bbox[2], st.bbox[3]),
                            conf=obs.conf,
                            keypoints_norm=tuple(frozen_pts),
                            liveness_verified=obs.liveness_verified,
                            spawn_center_hint=obs.spawn_center_hint,
                            confirmed_frames=obs.confirmed_frames,
                            outside_bed_disp=obs.outside_bed_disp,
                            is_still=True,
                            _kp_stabilized=True,
                        )
                    )
                    continue

                # РЕЖИМ 2: Реальное движение (is_still = False) — мгновенное скоростное раскрытие alpha -> 0.65..0.92!
                box_disp = max(
                    math.hypot(bx1 - st.bbox[0], by1 - st.bbox[1]),
                    math.hypot(bx2 - st.bbox[2], by2 - st.bbox[3]),
                )
                if box_disp < self._deadzone and opt_ratio < 0.025:
                    a_box = 0.35
                else:
                    a_box = min(0.90, max(0.65, 0.65 + (box_disp / 0.08) * 0.25))
                for idx_b, raw_val in enumerate((bx1, by1, bx2, by2)):
                    st.bbox[idx_b] = max(0.0, min(1.0, st.bbox[idx_b] + a_box * (raw_val - st.bbox[idx_b])))

                if on_bed_reclining and cur_gray_small is not None and scene_textured:
                    st.anchor_roi_blur = self._extract_blurred_roi(cur_gray_small, st.bbox)
                    st.anchor_frame_mean = float(np.mean(cur_gray_small))
                elif not on_bed_reclining:
                    st.anchor_roi_blur = None

                live_pts: list[tuple[float, float, float]] = []
                for k in range(17):
                    if k < len(obs.keypoints_norm):
                        kx, ky, kc = obs.keypoints_norm[k]
                    else:
                        kx, ky, kc = 0.0, 0.0, 0.0
                    kx = max(0.0, min(1.0, float(kx)))
                    ky = max(0.0, min(1.0, float(ky)))
                    kc = float(kc)
                    non_zero = kx > 0.0 or ky > 0.0

                    min_allowed_c = (
                        0.30
                        if (deep_blanket_head_mode and k <= 6)
                        else (self._min_pub_conf if (on_bed_reclining and k >= 7) else self._keep_conf)
                    )
                    if st.kp_ui_visible[k]:
                        if non_zero and kc >= min_allowed_c:
                            disp = math.hypot(kx - st.kp_xy[k][0], ky - st.kp_xy[k][1])
                            if disp < (self._deadzone * 0.5) and opt_ratio < 0.025:
                                alpha = 0.25
                            elif disp < self._deadzone and opt_ratio < 0.020:
                                alpha = 0.45
                            else:
                                alpha = min(0.92, max(0.65, 0.65 + (disp / 0.07) * 0.27))
                            nx = max(0.0, min(1.0, st.kp_xy[k][0] + alpha * (kx - st.kp_xy[k][0])))
                            ny = max(0.0, min(1.0, st.kp_xy[k][1] + alpha * (ky - st.kp_xy[k][1])))
                            pub_c = round(max(self._min_pub_conf, kc) if not deep_blanket_head_mode else kc, 4)
                            st.kp_xy[k] = [nx, ny]
                            st.kp_pub_conf[k] = pub_c
                            st.kp_anat_active[k] = True
                            st.kp_drop_streak[k] = 0
                            live_pts.append((nx, ny, pub_c))
                        elif st.kp_drop_streak[k] < 1 and st.kp_pub_conf[k] >= self._min_pub_conf:
                            st.kp_drop_streak[k] += 1
                            live_pts.append((st.kp_xy[k][0], st.kp_xy[k][1], st.kp_pub_conf[k]))
                        elif non_zero and kc >= self._min_anat_conf:
                            nx = max(0.0, min(1.0, st.kp_xy[k][0] + 0.65 * (kx - st.kp_xy[k][0])))
                            ny = max(0.0, min(1.0, st.kp_xy[k][1] + 0.65 * (ky - st.kp_xy[k][1])))
                            pub_c = round(min(sub_ui_cap, kc), 4)
                            st.kp_xy[k] = [nx, ny]
                            st.kp_pub_conf[k] = pub_c
                            st.kp_ui_visible[k] = False
                            st.kp_anat_active[k] = True
                            st.kp_drop_streak[k] = 0
                            live_pts.append((nx, ny, pub_c))
                        else:
                            st.kp_ui_visible[k] = False
                            st.kp_anat_active[k] = False
                            st.kp_drop_streak[k] = 0
                            st.kp_pub_conf[k] = 0.0
                            live_pts.append((0.0, 0.0, 0.0))
                    else:
                        eff_enter_live = (
                            0.32
                            if (deep_blanket_head_mode and k <= 6)
                            else (self._enter_conf if on_bed_reclining else self._min_pub_conf)
                        )
                        if non_zero and kc >= eff_enter_live:
                            # В фазе реального оптического движения конечности открываются без 3-кадрового лага!
                            pub_c = round(kc, 4)
                            st.kp_xy[k] = [kx, ky]
                            st.kp_pub_conf[k] = pub_c
                            st.kp_ui_visible[k] = True
                            st.kp_anat_active[k] = True
                            st.kp_drop_streak[k] = 0
                            st.kp_birth_streak[k] = self._bed_enter_frames
                            live_pts.append((kx, ky, pub_c))
                        elif non_zero and kc >= self._min_anat_conf:
                            if st.kp_anat_active[k]:
                                nx = max(0.0, min(1.0, st.kp_xy[k][0] + 0.65 * (kx - st.kp_xy[k][0])))
                                ny = max(0.0, min(1.0, st.kp_xy[k][1] + 0.65 * (ky - st.kp_xy[k][1])))
                            else:
                                nx, ny = kx, ky
                            pub_c = round(min(sub_ui_cap, kc), 4)
                            st.kp_xy[k] = [nx, ny]
                            st.kp_pub_conf[k] = pub_c
                            st.kp_anat_active[k] = True
                            st.kp_drop_streak[k] = 0
                            st.kp_birth_streak[k] = 0
                            live_pts.append((nx, ny, pub_c))
                        else:
                            st.kp_anat_active[k] = False
                            st.kp_birth_streak[k] = 0
                            live_pts.append((0.0, 0.0, 0.0))

                stabilized_out.append(
                    PoseObservation(
                        track_id=obs.track_id,
                        bbox_norm=(st.bbox[0], st.bbox[1], st.bbox[2], st.bbox[3]),
                        conf=obs.conf,
                        keypoints_norm=tuple(live_pts),
                        liveness_verified=obs.liveness_verified,
                        spawn_center_hint=obs.spawn_center_hint,
                        confirmed_frames=obs.confirmed_frames,
                        outside_bed_disp=obs.outside_bed_disp,
                        is_still=False,
                        _kp_stabilized=True,
                    )
                )

            # Оптически привязанное удержание (coast, до 3600 с) спящего трека в bed_polygon при провалах YOLO на одеяле:
            # держит спящего человека, пока ОДНОВРЕМЕННО межкадровое движение < 0.012 И кумулятивный дрейф
            # относительно эталонного патча сна anchor_roi_blur < 0.050. Сбрасывается на 1-м же кадре реального сдвига!
            bed_poly = getattr(zones, "bed_polygon", None) if zones is not None else None
            if (
                self._bed_coast_ttl > 0.0
                and bed_poly
                and len(bed_poly) >= 3
                and cur_gray_small is not None
                and prev_gray_valid
                and scene_textured
                and not same_frame_obj
            ):
                has_bed_obs_now = any(
                    is_person_on_bed(o, bed_poly, zones=zones) for o in stabilized_out
                )
                if not has_bed_obs_now:
                    best_coast_tid: int | None = None
                    best_coast_st: _OpticalTrackState | None = None
                    for c_tid, c_st in self._states.items():
                        if c_tid in incoming_tids or not c_st.on_bed_reclining:
                            continue
                        if not (0.0 <= (ts - c_st.last_detected_ts) <= self._bed_coast_ttl):
                            continue
                        if best_coast_st is None or c_st.last_detected_ts > best_coast_st.last_detected_ts:
                            best_coast_tid = c_tid
                            best_coast_st = c_st

                    if best_coast_tid is not None and best_coast_st is not None:
                        coast_box = (
                            best_coast_st.bbox[0],
                            best_coast_st.bbox[1],
                            best_coast_st.bbox[2],
                            best_coast_st.bbox[3],
                        )
                        coast_opt_ratio = self.compute_roi_optical_motion(
                            cur_gray_small,
                            self._prev_gray_small,
                            coast_box,
                            coast_box,
                        )
                        coast_anchor_drift = self.compute_anchor_drift(
                            cur_gray_small,
                            best_coast_st.anchor_roi_blur,
                            best_coast_st.anchor_frame_mean,
                            coast_box,
                        )
                        if coast_opt_ratio >= self._motion_thr:
                            best_coast_st.coast_motion_streak += 1
                        else:
                            best_coast_st.coast_motion_streak = 0

                        should_break_coast = (
                            coast_anchor_drift >= self._anchor_drift_thr
                            or coast_opt_ratio >= 0.035
                            or best_coast_st.coast_motion_streak >= 3
                        )
                        if not should_break_coast:
                            best_coast_st.last_ts = ts
                            coasted_pts = tuple(
                                (
                                    best_coast_st.kp_xy[k][0],
                                    best_coast_st.kp_xy[k][1],
                                    best_coast_st.kp_pub_conf[k],
                                )
                                if (best_coast_st.kp_ui_visible[k] or best_coast_st.kp_anat_active[k])
                                else (0.0, 0.0, 0.0)
                                for k in range(17)
                            )
                            stabilized_out.append(
                                PoseObservation(
                                    track_id=best_coast_tid,
                                    bbox_norm=coast_box,
                                    conf=best_coast_st.last_conf,
                                    keypoints_norm=coasted_pts,
                                    liveness_verified=best_coast_st.liveness_verified,
                                    spawn_center_hint=best_coast_st.spawn_center_hint,
                                    confirmed_frames=best_coast_st.confirmed_frames,
                                    outside_bed_disp=best_coast_st.outside_bed_disp,
                                    is_still=True,
                                    _kp_stabilized=True,
                                )
                            )
                        else:
                            best_coast_st.on_bed_reclining = False
                            del self._states[best_coast_tid]

            if cur_gray_small is not None and not same_frame_obj:
                self._prev_gray_small = cur_gray_small
                self._prev_gray_ts = ts
                self._prev_scene_textured = scene_textured
                self._last_frame_id = id(frame_bgr)
            elif same_frame_obj and ts > self._prev_gray_ts:
                self._prev_gray_ts = ts

            return stabilized_out
