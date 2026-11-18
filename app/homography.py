from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Sequence

import cv2
import numpy as np

# Калиброванные 4 угловые точки свободного прямоугольника пола (200 × 120 см) перед кроватью
# под стационарный верхне-боковой ракурс боевой камеры 192.168.1.50:
# [дальняя-левая у ножек кровати, дальняя-правая у ножек кровати, ближняя-правая, ближняя-левая]
DEFAULT_FLOOR_POLYGON_STD: list[list[float]] = [
    [0.265, 0.81],
    [0.635, 0.82],
    [0.685, 0.97],
    [0.215, 0.96],
]
DEFAULT_FLOOR_POLYGON: list[list[float]] = DEFAULT_FLOOR_POLYGON_STD
DEFAULT_ROOM_FLOOR_POLYGON: list[list[float]] = DEFAULT_FLOOR_POLYGON_STD
DEFAULT_CAMERA_FLOOR_POLYGON: list[list[float]] = DEFAULT_FLOOR_POLYGON_STD
DEFAULT_FLOOR_WIDTH_CM: float = 200.0
DEFAULT_FLOOR_DEPTH_CM: float = 120.0
DEFAULT_MATTRESS_HEIGHT_CM: float = 55.0


class FootFloorMeasurement(dict):  # type: ignore[type-arg]
    """
    Метрический 3D-замер человека относительно плоскости пола (Z = 0) и матраса (Z = 55 см).
    Одновременно поддерживает обращение по атрибутам (.height_cm, .floor_xy_cm, ...) и как dict (.get(...)).
    """

    floor_xy_cm: tuple[float, float] | None
    height_cm: float | None
    hip_z_cm: float | None
    standing_on_floor_in_front: bool
    on_floor_quad: bool

    def __init__(
        self,
        floor_xy_cm: tuple[float, float] | None,
        height_cm: float | None,
        hip_z_cm: float | None,
        standing_on_floor_in_front: bool,
        on_floor_quad: bool,
    ) -> None:
        super().__init__(
            floor_xy_cm=list(floor_xy_cm) if floor_xy_cm is not None else None,
            height_cm=height_cm,
            hip_z_cm=hip_z_cm,
            standing_on_floor_in_front=bool(standing_on_floor_in_front),
            on_floor_quad=bool(on_floor_quad),
        )
        self.floor_xy_cm = floor_xy_cm
        self.height_cm = height_cm
        self.hip_z_cm = hip_z_cm
        self.standing_on_floor_in_front = bool(standing_on_floor_in_front)
        self.on_floor_quad = bool(on_floor_quad)


def _order_floor_quad(pts: Sequence[Sequence[float]]) -> list[list[float]]:
    """
    Упорядочивает 4 точки перспективной трапеции пола в канонический обход:
    0: дальняя-левая (у переднего борта кровати, Y=0, X=0)
    1: дальняя-правая (у переднего борта кровати, Y=0, X=W)
    2: ближняя-правая (к камере, Y=D, X=W)
    3: ближняя-левая (к камере, Y=D, X=0)
    """
    clean = [[float(p[0]), float(p[1])] for p in pts[:4]]
    by_y = sorted(clean, key=lambda p: (p[1], p[0]))
    far_two = sorted(by_y[:2], key=lambda p: p[0])
    near_two = sorted(by_y[2:], key=lambda p: p[0])
    return [far_two[0], far_two[1], near_two[1], near_two[0]]


def derive_default_floor_polygon(
    bed_polygon: Sequence[Sequence[float]] | None = None,
) -> list[list[float]]:
    """
    Возвращает 4 опорные точки пола перед кроватью:
    - для стационарного боевого ракурса 192.168.1.50 — эталонный DEFAULT_FLOOR_POLYGON_STD;
    - для произвольно размеченной кровати — перспективную трапецию пола сразу ниже переднего края матраса.
    """
    if not bed_polygon or len(bed_polygon) < 3:
        return [list(pt) for pt in DEFAULT_FLOOR_POLYGON_STD]

    xs = [float(p[0]) for p in bed_polygon]
    ys = [float(p[1]) for p in bed_polygon]
    min_x, max_x = min(xs), max(xs)
    max_y = max(ys)

    # Боевой ракурс 192.168.1.50 (передний борт матраса y ≈ 0.78..0.80, x ≈ 0.26..0.69)
    if 0.75 <= max_y <= 0.82 and 0.22 <= min_x <= 0.33 and 0.58 <= max_x <= 0.74:
        return [list(pt) for pt in DEFAULT_FLOOR_POLYGON_STD]

    span_x = max(0.24, min(0.60, max_x - min_x))
    cx = (min_x + max_x) * 0.5
    half_far = span_x * 0.44
    half_near = min(0.46, half_far * 1.24)
    y_far = min(0.86, max_y + 0.025)
    y_near = min(0.985, max(y_far + 0.11, y_far + 0.15))
    return [
        [round(max(0.02, cx - half_far), 5), round(y_far, 5)],
        [round(min(0.98, cx + half_far), 5), round(y_far, 5)],
        [round(min(0.99, cx + half_near), 5), round(y_near, 5)],
        [round(max(0.01, cx - half_near), 5), round(y_near, 5)],
    ]


class FloorHomography:
    """
    Проективная 3D-геометрия плоскости пола (Z = 0 см) и поверхности матраса (Z = mattress_height_cm).
    Строит прямую и обратную матрицы гомографии через cv2.findHomography и восстанавливает
    полную 3D-матрицу проекции камеры P = K [r1 r2 r3 t] для расчёта координат стоп на полу (см),
    реального физического роста стоящего человека (см) и устранения 2D-параллакса у борта кровати.
    """

    _cache_lock = threading.Lock()
    _cached_key: tuple[Any, ...] | None = None
    _cached_instance: "FloorHomography | None" = None

    def __init__(
        self,
        floor_polygon: Sequence[Sequence[float]] | None = None,
        floor_width_cm: float = DEFAULT_FLOOR_WIDTH_CM,
        floor_depth_cm: float = DEFAULT_FLOOR_DEPTH_CM,
        mattress_height_cm: float = DEFAULT_MATTRESS_HEIGHT_CM,
        bed_polygon: Sequence[Sequence[float]] | None = None,
        ref_width_px: float = 1920.0,
        ref_height_px: float = 1080.0,
    ) -> None:
        self._w_px = float(max(320.0, ref_width_px))
        self._h_px = float(max(180.0, ref_height_px))
        self.floor_width_cm = float(max(50.0, min(600.0, floor_width_cm)))
        self.floor_depth_cm = float(max(40.0, min(500.0, floor_depth_cm)))
        self.mattress_height_cm = float(max(20.0, min(110.0, mattress_height_cm)))

        custom_valid = (
            floor_polygon is not None
            and len(floor_polygon) >= 4
            and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in floor_polygon[:4])
        )
        raw_quad = (
            [[float(p[0]), float(p[1])] for p in floor_polygon[:4]]  # type: ignore[index]
            if custom_valid
            else derive_default_floor_polygon(bed_polygon)
        )
        self.calibrated: bool = bool(custom_valid)
        self.floor_polygon: list[list[float]] = _order_floor_quad(raw_quad)

        self._h_img_to_world: np.ndarray | None = None
        self._h_world_to_img: np.ndarray | None = None
        self._proj_3d: np.ndarray | None = None
        self._cam_height_cm: float = 215.0
        self.is_valid: bool = False

        self._build_matrices(bed_polygon)
        if not self.is_valid and custom_valid:
            # Если пользовательские 4 точки оказались коллинеарными — откатываемся к безопасному дефолту
            self.calibrated = False
            self.floor_polygon = _order_floor_quad(derive_default_floor_polygon(bed_polygon))
            self._build_matrices(bed_polygon)

    def _build_matrices(self, bed_polygon: Sequence[Sequence[float]] | None) -> None:
        img_pts = np.asarray(
            [[p[0] * self._w_px, p[1] * self._h_px] for p in self.floor_polygon],
            dtype=np.float64,
        )
        area_px = float(cv2.contourArea(img_pts.astype(np.float32)))
        if area_px < (0.002 * self._w_px * self._h_px):
            return

        world_pts = np.asarray(
            [
                [0.0, 0.0],
                [self.floor_width_cm, 0.0],
                [self.floor_width_cm, self.floor_depth_cm],
                [0.0, self.floor_depth_cm],
            ],
            dtype=np.float64,
        )
        h_w2i, _ = cv2.findHomography(world_pts, img_pts, method=0)
        h_i2w, _ = cv2.findHomography(img_pts, world_pts, method=0)
        if h_w2i is None or h_i2w is None:
            return
        if abs(float(np.linalg.det(h_w2i))) < 1e-12 or abs(float(np.linalg.det(h_i2w))) < 1e-12:
            return

        # Восстанавливаем согласованную 3D-матрицу проекции P = K [r1 r2 r3 t] (HFOV ~ 86° -> f ≈ 0.54 * W)
        f_px = 0.54 * self._w_px
        cx, cy = self._w_px * 0.5, self._h_px * 0.5
        k_mat = np.asarray([[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        k_inv = np.linalg.inv(k_mat)

        ext_raw = k_inv @ h_w2i
        if ext_raw[2, 2] < 0.0:
            h_w2i = -h_w2i
            ext = -ext_raw
        else:
            ext = ext_raw
        norm1 = float(np.linalg.norm(ext[:, 0]))
        norm2 = float(np.linalg.norm(ext[:, 1]))
        if norm1 < 1e-9 or norm2 < 1e-9:
            return
        scale = 2.0 / (norm1 + norm2)
        h_norm = h_w2i * scale
        r1_raw = ext[:, 0] * scale
        r2_raw = ext[:, 1] * scale
        t_vec = ext[:, 2] * scale

        # Вектор нормали к плоскости пола (+Z строго вверх к потолку) с сохранением точной гомографии Z = 0
        r3 = np.cross(r1_raw, r2_raw)
        r3_norm = float(np.linalg.norm(r3))
        if r3_norm < 1e-9:
            return
        r3 = r3 / r3_norm

        k_r3 = k_mat @ r3
        p_pos = np.column_stack([h_norm[:, 0], h_norm[:, 1], k_r3, h_norm[:, 2]])
        mid_x = self.floor_width_cm * 0.5
        mid_y = self.floor_depth_cm * 0.4
        pt0 = p_pos @ np.asarray([mid_x, mid_y, 0.0, 1.0], dtype=np.float64)
        pt1 = p_pos @ np.asarray([mid_x, mid_y, 100.0, 1.0], dtype=np.float64)
        if abs(pt0[2]) > 1e-9 and abs(pt1[2]) > 1e-9 and (pt1[1] / pt1[2]) > (pt0[1] / pt0[2]):
            r3 = -r3
            k_r3 = -k_r3

        # Высота оптического центра камеры над плоскостью пола Z = 0: C_z ≈ |r3 · t|
        cam_z = abs(float(-np.dot(r3, t_vec)))
        self._cam_height_cm = float(max(120.0, min(420.0, cam_z)))
        self._h_world_to_img = h_norm
        self._h_img_to_world = h_i2w
        self._proj_3d = np.column_stack([h_norm[:, 0], h_norm[:, 1], k_r3, h_norm[:, 2]])
        self.is_valid = True

    @classmethod
    def from_zones(
        cls,
        zones: Any = None,
        bed_polygon: Sequence[Sequence[float]] | None = None,
    ) -> "FloorHomography":
        """Быстрое потокобезопасное получение калиброванной гомографии из объекта ZonesConfig."""
        if zones is None:
            raw_bp = bed_polygon or ()
            bp_tuple = tuple((round(float(p[0]), 5), round(float(p[1]), 5)) for p in raw_bp if len(p) >= 2)
            key: tuple[Any, ...] = (
                (),
                DEFAULT_FLOOR_WIDTH_CM,
                DEFAULT_FLOOR_DEPTH_CM,
                DEFAULT_MATTRESS_HEIGHT_CM,
                bp_tuple,
            )
        else:
            raw_fp = getattr(zones, "floor_polygon", None) or ()
            fp_tuple = tuple((round(float(p[0]), 5), round(float(p[1]), 5)) for p in raw_fp if len(p) >= 2)
            raw_bp = getattr(zones, "bed_polygon", None) or bed_polygon or ()
            bp_tuple = tuple((round(float(p[0]), 5), round(float(p[1]), 5)) for p in raw_bp if len(p) >= 2)
            fw = float(getattr(zones, "floor_width_cm", DEFAULT_FLOOR_WIDTH_CM) or DEFAULT_FLOOR_WIDTH_CM)
            fd = float(getattr(zones, "floor_depth_cm", DEFAULT_FLOOR_DEPTH_CM) or DEFAULT_FLOOR_DEPTH_CM)
            mh = float(getattr(zones, "mattress_height_cm", DEFAULT_MATTRESS_HEIGHT_CM) or DEFAULT_MATTRESS_HEIGHT_CM)
            key = (fp_tuple, round(fw, 2), round(fd, 2), round(mh, 2), bp_tuple)

        with cls._cache_lock:
            if cls._cached_key == key and cls._cached_instance is not None:
                return cls._cached_instance

        inst = cls(
            floor_polygon=list(key[0]) if key[0] else None,
            floor_width_cm=float(key[1]),
            floor_depth_cm=float(key[2]),
            mattress_height_cm=float(key[3]),
            bed_polygon=list(key[4]) if key[4] else None,
        )
        with cls._cache_lock:
            cls._cached_key = key
            cls._cached_instance = inst
        return inst

    @staticmethod
    def default_floor_polygon() -> list[list[float]]:
        return [list(pt) for pt in DEFAULT_FLOOR_POLYGON_STD]

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibrated and self.is_valid)

    @property
    def camera_height_cm(self) -> float:
        return round(self._cam_height_cm, 1)

    def image_to_floor_cm(self, uv: tuple[float, float]) -> tuple[float, float] | None:
        """Переводит нормализованную точку кадра (u, v) на плоскость пола Z = 0 см -> (x_cm, y_cm)."""
        if not self.is_valid or self._h_img_to_world is None:
            return None
        px = np.asarray([float(uv[0]) * self._w_px, float(uv[1]) * self._h_px, 1.0], dtype=np.float64)
        w = self._h_img_to_world @ px
        if abs(float(w[2])) < 1e-9:
            return None
        x_cm = float(w[0] / w[2])
        y_cm = float(w[1] / w[2])
        if not (math.isfinite(x_cm) and math.isfinite(y_cm)):
            return None
        return (x_cm, y_cm)

    def world_3d_to_image_uv(
        self,
        x_cm: float,
        y_cm: float,
        z_cm: float = 0.0,
    ) -> tuple[float, float] | None:
        """Проецирует мировую 3D-точку (x_cm, y_cm, z_cm) в нормализованные координаты кадра (u, v)."""
        if not self.is_valid or self._proj_3d is None:
            return None
        pt = self._proj_3d @ np.asarray([float(x_cm), float(y_cm), float(z_cm), 1.0], dtype=np.float64)
        if float(pt[2]) <= 1e-6:
            return None
        u = float(pt[0] / pt[2]) / self._w_px
        v = float(pt[1] / pt[2]) / self._h_px
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return (u, v)

    def image_to_plane_z_cm(
        self,
        uv: tuple[float, float],
        z_cm: float,
    ) -> tuple[float, float] | None:
        """Пересекает 3D-луч пикселя (u, v) с горизонтальной плоскостью высоты Z = z_cm (например Z = 55 см матраса)."""
        if not self.is_valid or self._proj_3d is None:
            return None
        p = self._proj_3d
        # Для фиксированного Z: p[:, 0]*X + p[:, 1]*Y + (p[:, 2]*Z + p[:, 3])*1 ~ [u*W, v*H, 1]^T
        h_z = np.column_stack([p[:, 0], p[:, 1], p[:, 2] * float(z_cm) + p[:, 3]])
        det = float(np.linalg.det(h_z))
        if abs(det) < 1e-12:
            return None
        w = np.linalg.inv(h_z) @ np.asarray(
            [float(uv[0]) * self._w_px, float(uv[1]) * self._h_px, 1.0],
            dtype=np.float64,
        )
        if abs(float(w[2])) < 1e-9:
            return None
        return (float(w[0] / w[2]), float(w[1] / w[2]))

    def estimate_standing_height_cm(
        self,
        foot_uv: tuple[float, float],
        top_uv: tuple[float, float],
    ) -> float | None:
        """
        Вычисляет физический рост (см) вертикального отрезка от точки опоры на полу foot_uv (Z = 0)
        до верхней точки макушки top_uv (X_foot, Y_foot, Z = height_cm).
        """
        if not self.is_valid or self._proj_3d is None:
            return None
        if float(foot_uv[1]) <= float(top_uv[1]) + 0.04:
            return None
        floor_xy = self.image_to_floor_cm(foot_uv)
        if floor_xy is None:
            return None
        x_cm, y_cm = floor_xy
        # Отсекаем далёкие точки за горизонтом камеры
        if not (-250.0 <= x_cm <= self.floor_width_cm + 250.0 and -220.0 <= y_cm <= self.floor_depth_cm + 220.0):
            return None

        p = self._proj_3d
        y_top_px = float(top_uv[1]) * self._h_px
        num = (
            p[1, 0] * x_cm
            + p[1, 1] * y_cm
            + p[1, 3]
            - y_top_px * (p[2, 0] * x_cm + p[2, 1] * y_cm + p[2, 3])
        )
        den = y_top_px * p[2, 2] - p[1, 2]
        if abs(float(den)) < 1e-9:
            return None
        z_cm = float(num / den)
        if not math.isfinite(z_cm) or z_cm <= 25.0 or z_cm >= 260.0:
            return None
        return round(z_cm, 1)

    def measure_pose(
        self,
        obs: Any,
        bed_polygon: Sequence[Sequence[float]] | None = None,
    ) -> FootFloorMeasurement:
        """
        Выполняет полный 3D-замер человека:
        - точка опоры стоп на плоскости пола (floor_x_cm, floor_y_cm);
        - рост вертикально стоящего человека в см (height_cm);
        - строгий 3D-критерий standing_on_floor_in_front (Z=0 пол перед бортом vs Z=55 см матрас).
        """
        bx1, by1, bx2, by2 = obs.bbox_norm
        cx = (bx1 + bx2) * 0.5

        # Точка контакта подошв с полом: видимые лодыжки (+1.2% на стопу) либо нижний срез бокса
        ankles: list[tuple[float, float]] = []
        for idx in (15, 16):
            ak = obs.kp(idx, 0.35)
            if ak is not None:
                ankles.append(ak)
        if ankles:
            low_ak = max(ankles, key=lambda pt: pt[1])
            foot_u = sum(pt[0] for pt in ankles) / float(len(ankles))
            foot_v = min(1.0, max(by2, low_ak[1] + 0.005)) if (by2 - low_ak[1]) <= 0.04 else min(1.0, low_ak[1] + 0.008)
        else:
            foot_u, foot_v = cx, by2

        top_uv = (cx, by1)
        floor_xy_raw = self.image_to_floor_cm((foot_u, foot_v))
        floor_xy: tuple[float, float] | None = None
        on_quad = False
        if floor_xy_raw is not None and (
            -220.0 <= floor_xy_raw[0] <= self.floor_width_cm + 220.0
            and -200.0 <= floor_xy_raw[1] <= self.floor_depth_cm + 220.0
        ):
            floor_xy = (round(floor_xy_raw[0], 1), round(floor_xy_raw[1], 1))
            on_quad = (
                -25.0 <= floor_xy_raw[0] <= self.floor_width_cm + 25.0
                and -10.0 <= floor_xy_raw[1] <= self.floor_depth_cm + 35.0
            )

        angle = obs.torso_angle_from_vertical_deg()
        upright_body = (
            (angle is None and obs.aspect_ratio_wh < 0.52 and (by2 - by1) >= 0.24)
            or (angle is not None and angle < 36.0 and obs.aspect_ratio_wh < 0.70 and (by2 - by1) >= 0.20)
        )

        bed_min_y = min((float(p[1]) for p in bed_polygon), default=0.39) if bed_polygon and len(bed_polygon) >= 3 else 0.39
        bed_max_y = max((float(p[1]) for p in bed_polygon), default=0.79) if bed_polygon and len(bed_polygon) >= 3 else 0.79
        # Стопы реально опираются на пол (а не лежат высоко на подушке внутри матраса), если низ силуэта
        # находится в зоне пола или ниже проекции матраса
        feet_reach_floor_zone = (
            (floor_xy_raw is not None and floor_xy_raw[1] >= -15.0)
            or foot_v >= (bed_max_y + 0.02)
            or by2 >= 0.80
        )

        height_cm: float | None = None
        hip_z_cm: float | None = None
        raw_h: float | None = None
        if upright_body and feet_reach_floor_zone and (foot_v <= 0.988 or by2 <= 0.992):
            raw_h = self.estimate_standing_height_cm((foot_u, foot_v), top_uv)
            if raw_h is not None and 105.0 <= raw_h <= 225.0:
                height_cm = raw_h
                hip_z_cm = round(0.53 * raw_h, 1)

        # 3D-проверка стояния на полу вплотную ПЕРЕД передним бортом кровати (устранение 2D-параллакса таза)
        standing_in_front = False
        if (
            bed_polygon is not None
            and len(bed_polygon) >= 3
            and angle is not None
            and angle < 28.0
            and obs.aspect_ratio_wh < 0.56
            and floor_xy_raw is not None
        ):
            fx_cm, fy_cm = floor_xy_raw
            # Точка опоры стоп физически находится на плоскости пола Z = 0 перед ножкой кровати (fy_cm >= -8 см)
            feet_in_front_3d = fy_cm >= -8.0 and foot_v >= (bed_max_y + 0.03)
            valid_human_scale = (raw_h is None and height_cm is None) or (
                height_cm is not None and 105.0 <= height_cm <= 225.0
            )
            if feet_in_front_3d and valid_human_scale:
                mh = obs.mid_hip
                if mh is None:
                    standing_in_front = True
                else:
                    bed_arr = np.asarray(bed_polygon, dtype=np.float32) * 1000.0
                    dist_in_bed = (
                        float(cv2.pointPolygonTest(bed_arr, (float(mh[0]) * 1000.0, float(mh[1]) * 1000.0), True))
                        / 1000.0
                    )
                    est_hip_z = hip_z_cm if hip_z_cm is not None else self.mattress_height_cm
                    pred_hip_uv = self.world_3d_to_image_uv(fx_cm, fy_cm, est_hip_z)
                    hip_on_mattress_xy = self.image_to_plane_z_cm(mh, self.mattress_height_cm)

                    # 3D-разделение пола (Z = 0) и матраса (Z = 55 см):
                    # у человека, стоящего на полу перед бортом (Y >= 0), луч таза пересекает Z = 55 см при Y >= -8 см
                    # и совпадает с 3D-колонной над стопами; у сидящего на матрасе со свешенными ногами таз уходит вглубь Y < -10 см!
                    tall_full_stander = (
                        height_cm is not None
                        and height_cm >= 172.0
                        and by1 <= (bed_min_y - 0.015)
                        and by2 >= (bed_max_y + 0.06)
                    )
                    has_bent_sitting_knee = False
                    if not tall_full_stander:
                        for k_idx in (13, 14):
                            kn = obs.kp(k_idx, 0.45)
                            if kn is not None and (kn[1] - mh[1]) <= 0.11 and dist_in_bed > 0.06:
                                has_bent_sitting_knee = True
                                break

                    if self.calibrated and hip_on_mattress_xy is not None and pred_hip_uv is not None:
                        vertical_3d_match = (
                            abs(mh[0] - pred_hip_uv[0]) <= 0.045
                            and abs(mh[1] - pred_hip_uv[1]) <= 0.055
                        ) or (
                            tall_full_stander
                            and fy_cm >= 20.0
                            and abs(mh[0] - pred_hip_uv[0]) <= 0.065
                            and abs(mh[1] - pred_hip_uv[1]) <= 0.095
                        )
                        seated_deep_on_mattress = has_bent_sitting_knee or (
                            hip_on_mattress_xy[1] < -10.0 and not vertical_3d_match
                        )
                    else:
                        seated_deep_on_mattress = has_bent_sitting_knee or (
                            mh[1] < (bed_max_y - 0.08) and dist_in_bed > 0.08
                        )

                    if not seated_deep_on_mattress:
                        consistent_3d_column = (
                            pred_hip_uv is not None and abs(mh[1] - pred_hip_uv[1]) <= 0.095
                        )
                        not_deep_on_mattress = (
                            hip_on_mattress_xy is None
                            or hip_on_mattress_xy[1] >= -35.0
                            or (tall_full_stander and consistent_3d_column)
                        )
                        if (consistent_3d_column or mh[1] >= (bed_max_y - 0.08)) and not_deep_on_mattress:
                            standing_in_front = True

        return FootFloorMeasurement(
            floor_xy_cm=floor_xy,
            height_cm=height_cm,
            hip_z_cm=hip_z_cm,
            standing_on_floor_in_front=standing_in_front,
            on_floor_quad=on_quad,
        )
