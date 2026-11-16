from __future__ import annotations

import enum
import json
import logging
import math
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .geometry import (
    KP_L_EYE,
    KP_L_SHOULDER,
    KP_NOSE,
    KP_R_EYE,
    KP_R_SHOULDER,
    OpticalKeypointStabilizer,
    PoseObservation,
    StaticObjectGate,
    are_legs_hanging_outside_bed,
    bbox_iou,
    bbox_polygon_overlap_ratio,
    filter_valid_poses,
    is_confident_foreground_seated_person,
    is_person_on_bed,
    is_pose_touching_door,
    is_standing_in_front_of_bed,
    is_valid_human_pose,
    point_in_polygon,
    polygon_distance_norm,
    polygon_y_bounds,
    sanitize_collapsed_bed_keypoints,
)
from .homography import FloorHomography
from .zones import ZonesConfig

logger = logging.getLogger("fallguard.patient_id")


def _suppress_opencv_log_spam() -> None:
    """Подавляет избыточный C++ лог-спам OpenCV DNN/ONNX при инициализации моделей."""
    try:
        if hasattr(cv2, "setLogLevel"):
            cv_log = getattr(getattr(cv2, "utils", None), "logging", None)
            err_level = getattr(cv_log, "LOG_LEVEL_ERROR", 2) if cv_log is not None else 2
            cv2.setLogLevel(int(err_level))
    except Exception:
        pass


_suppress_opencv_log_spam()

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

DEFAULT_ANCHOR_TTL_SECONDS = 12.0 * 3600.0  # 12 часов памяти текущей одежды бабушки
DEFAULT_VISITOR_OUTFIT_TTL_SECONDS = 8.0 * 3600.0  # 8 часов памяти одежды родственников («Свой»)
DEFAULT_DOOR_VISIT_IDLE_TIMEOUT = 4.0 * 3600.0  # 4 часа удержания визита родственника до реального выхода в дверь
MAX_VISITOR_OUTFITS = 8
ANCHOR_SCHEMA_VERSION = 4
NIGHT_IR_SAT_THRESHOLD = 12.0
DAY_DESCRIPTOR_DIM = 48
NIGHT_DESCRIPTOR_DIM = 64


class Role(str, enum.Enum):
    PATIENT = "patient"
    VISITOR = "visitor"
    UNKNOWN = "unknown"


def _coerce_role(role: Role | str) -> Role | None:
    if isinstance(role, Role):
        return role if role in (Role.PATIENT, Role.VISITOR) else None
    if isinstance(role, str):
        val = role.strip().lower()
        if val == Role.PATIENT.value:
            return Role.PATIENT
        if val == Role.VISITOR.value:
            return Role.VISITOR
    return None


@dataclass
class TrackIdentity:
    track_id: int
    role: Role = Role.UNKNOWN
    reason: str = "init"
    first_seen: float = 0.0
    last_seen: float = 0.0
    bed_accum_sec: float = 0.0
    last_face_check: float = 0.0
    last_appearance_calc: float = -1e9
    face_score: float = 0.0
    spawned_outside_bed: bool = False
    spawned_in_door: bool = False
    has_moved: bool = False
    was_on_bed: bool = False
    confirmed_second_track: bool = False
    outfit_saved: bool = False
    face_patient_hits: int = 0
    face_visitor_hits: int = 0
    manual_assigned_at: float = -1e9
    last_center: tuple[float, float] = (0.5, 0.5)
    height_cm: float | None = None
    floor_xy_cm: tuple[float, float] | None = None
    appearance_is_night: bool = False
    appearance_vec: np.ndarray | None = field(default=None, repr=False)
    appearance_vec_day: np.ndarray | None = field(default=None, repr=False)
    appearance_vec_night: np.ndarray | None = field(default=None, repr=False)


def is_night_ir_frame(
    frame_bgr: np.ndarray | None,
    sat_threshold: float = NIGHT_IR_SAT_THRESHOLD,
) -> bool:
    """
    Автоматически определяет ночной монохромный Ч/Б ИК-режим камеры (850 нм):
    считает среднюю цветовую насыщенность кадра в пространстве HSV (mean(S) < 12.0).
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.size == 0:
        return False
    h, w = frame_bgr.shape[:2]
    if h > 180 or w > 320:
        sample = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    else:
        sample = frame_bgr
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) < float(sat_threshold)


def _extract_patch_descriptor_48(patch: np.ndarray) -> np.ndarray | None:
    """Извлекает 48-мерный цвето-текстурный дескриптор из BGR-патча одежды."""
    ph, pw = patch.shape[:2]
    if pw < 8 or ph < 8:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    val_chan = hsv[:, :, 2]
    total_px = float(max(1, ph * pw))

    # 1. Цветовая компонента (24 бина) строго по пикселям с насыщенностью S >= 28
    color_mask = cv2.inRange(hsv, (0, 28, 20), (180, 255, 255))
    colored_px = float(cv2.countNonZero(color_mask))
    color_ratio = colored_px / total_px
    if colored_px >= max(8.0, total_px * 0.08):
        hist_hs = cv2.calcHist([hsv], [0, 1], color_mask, [6, 4], [0, 180, 28, 256]).flatten().astype(np.float32)
        hs_norm = float(np.linalg.norm(hist_hs))
        if hs_norm > 1e-6:
            hist_hs = (hist_hs / hs_norm) * min(1.0, 0.45 + color_ratio)
    else:
        hist_hs = np.zeros(24, dtype=np.float32)

    # 2. Пространственный ИК-профиль яркости по 3 горизонтальным зонам (3 × 6 = 18 бинов)
    s1 = max(1, ph // 3)
    s2 = max(s1 + 1, (2 * ph) // 3)
    stripes = (val_chan[:s1, :], val_chan[s1:s2, :], val_chan[s2:, :])
    v_parts: list[np.ndarray] = []
    for stripe in stripes:
        if stripe.size == 0:
            v_parts.append(np.zeros(6, dtype=np.float32))
            continue
        hv = cv2.calcHist([stripe], [0], None, [6], [0, 256]).flatten().astype(np.float32)
        hv_norm = float(np.linalg.norm(hv))
        v_parts.append(hv / hv_norm if hv_norm > 1e-6 else hv)
    hist_v_spatial = np.concatenate(v_parts) * 0.65

    # 3. Градиентная текстура складок/ткани (6 бинов)
    gx = cv2.Sobel(val_chan, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(val_chan, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.clip(cv2.magnitude(gx, gy), 0.0, 255.0).astype(np.uint8)
    hist_tex = cv2.calcHist([mag], [0], None, [6], [0, 256]).flatten().astype(np.float32)
    tex_norm = float(np.linalg.norm(hist_tex))
    if tex_norm > 1e-6:
        hist_tex = (hist_tex / tex_norm) * 0.40

    vec = np.concatenate([hist_hs, hist_v_spatial, hist_tex]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def _extract_night_ir_patch_64(patch: np.ndarray) -> np.ndarray | None:
    """
    Извлекает 64-мерный текстурно-градиентный дескриптор для ночного монохромного Ч/Б ИК (850 нм):
    1) 36 бинов — Rotation-Invariant Uniform 8-neighbor LBP (верх/низ 2×10) + 16-биновая гистограмма кода LBP;
    2) 12 бинов — ориентированная гистограмма градиентов Sobel (8 угловых бинов 0..180° + 4 бина силы);
    3) 16 бинов — многозонный 2×2 CLAHE-профиль яркости (4×3) + 4 квантиля ИК-альбедо ткани.
    """
    ph, pw = patch.shape[:2]
    if pw < 8 or ph < 8:
        return None

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if (ph >= 48 and pw >= 48) else cv2.INTER_LINEAR
    raw48 = cv2.resize(gray, (48, 48), interpolation=interp)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
    eq = clahe.apply(raw48)

    # 1. Векторизованный 8-соседний Uniform LBP на внутренней сетке 46x46
    c = eq[1:-1, 1:-1]
    neighbors = (
        eq[0:-2, 0:-2] >= c,
        eq[0:-2, 1:-1] >= c,
        eq[0:-2, 2:] >= c,
        eq[1:-1, 2:] >= c,
        eq[2:, 2:] >= c,
        eq[2:, 1:-1] >= c,
        eq[2:, 0:-2] >= c,
        eq[1:-1, 0:-2] >= c,
    )
    transitions = np.zeros_like(c, dtype=np.uint8)
    ones = np.zeros_like(c, dtype=np.uint8)
    lbp_code = np.zeros_like(c, dtype=np.uint8)
    for bit_idx in range(8):
        cur_b = neighbors[bit_idx]
        nxt_b = neighbors[(bit_idx + 1) % 8]
        transitions += (cur_b != nxt_b).astype(np.uint8)
        ones += cur_b.astype(np.uint8)
        lbp_code |= cur_b.astype(np.uint8) << bit_idx

    riu2 = np.where(transitions <= 2, ones, 9).astype(np.int32)
    mid_r = riu2.shape[0] // 2
    lbp_top = np.bincount(riu2[:mid_r, :].ravel(), minlength=10).astype(np.float32)
    lbp_bot = np.bincount(riu2[mid_r:, :].ravel(), minlength=10).astype(np.float32)
    lbp_u_vec = np.concatenate([lbp_top, lbp_bot])
    lbp_u_norm = float(np.linalg.norm(lbp_u_vec))
    if lbp_u_norm > 1e-6:
        lbp_u_vec = (lbp_u_vec / lbp_u_norm) * 0.55

    lbp_hist16 = cv2.calcHist([lbp_code], [0], None, [16], [0, 256]).flatten().astype(np.float32)
    lbp_h_norm = float(np.linalg.norm(lbp_hist16))
    if lbp_h_norm > 1e-6:
        lbp_hist16 = (lbp_hist16 / lbp_h_norm) * 0.35

    # 2. Ориентированная гистограмма градиентов складок/вязки Sobel (8 углов + 4 магнитуды = 12 бинов)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang_deg = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ori180 = np.mod(ang_deg, 180.0)
    ori_hist, _ = np.histogram(ori180, bins=8, range=(0.0, 180.0), weights=mag)
    ori_vec = ori_hist.astype(np.float32)
    ori_norm = float(np.linalg.norm(ori_vec))
    if ori_norm > 1e-6:
        ori_vec = (ori_vec / ori_norm) * 0.45

    mag_clipped = np.clip(mag, 0.0, 255.0)
    mag_hist, _ = np.histogram(mag_clipped, bins=4, range=(0.0, 256.0))
    mag_vec = mag_hist.astype(np.float32)
    mag_norm = float(np.linalg.norm(mag_vec))
    if mag_norm > 1e-6:
        mag_vec = (mag_vec / mag_norm) * 0.25

    # 3. Многозонный 2x2 CLAHE-профиль (12 бинов) + 4 квантиля ИК-отражения ткани (16 бинов)
    quads = (eq[:24, :24], eq[:24, 24:], eq[24:, :24], eq[24:, 24:])
    q_parts: list[np.ndarray] = []
    for q in quads:
        hq = cv2.calcHist([q], [0], None, [3], [0, 256]).flatten().astype(np.float32)
        hq_n = float(np.linalg.norm(hq))
        q_parts.append(hq / hq_n if hq_n > 1e-6 else hq)
    clahe_quad_vec = np.concatenate(q_parts) * 0.40

    raw_f = raw48.astype(np.float32)
    p15, p85 = np.percentile(raw_f, [15.0, 85.0])
    albedo_stats = (
        np.asarray(
            [
                float(np.mean(raw_f)) / 255.0,
                float(np.std(raw_f)) / 128.0,
                float(p15) / 255.0,
                float(p85) / 255.0,
            ],
            dtype=np.float32,
        )
        * 0.45
    )

    vec = np.concatenate([lbp_u_vec, lbp_hist16, ori_vec, mag_vec, clahe_quad_vec, albedo_stats]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def _crop_collar_and_torso_patches(
    frame_bgr: np.ndarray,
    obs: PoseObservation,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Вырезает область воротника/плеч (70% веса) и общего торса (30% веса)."""
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None, None
    h, w = frame_bgr.shape[:2]
    if h < 32 or w < 32:
        return None, None

    ms = obs.mid_shoulder
    mh = obs.mid_hip
    bx1, by1, bx2, by2 = obs.bbox_norm
    bw = max(0.02, bx2 - bx1)
    bh = max(0.03, by2 - by1)

    if ms is not None and mh is not None:
        ux = ms[0] * 0.76 + mh[0] * 0.24
        uy = ms[1] * 0.76 + mh[1] * 0.24
        u_hw = max(0.028, bw * 0.25)
        u_hh = max(0.032, abs(mh[1] - ms[1]) * 0.30, bh * 0.14)
    elif ms is not None:
        ux = ms[0]
        uy = min(0.98, ms[1] + bh * 0.10)
        u_hw = max(0.028, bw * 0.28)
        u_hh = max(0.032, bh * 0.20)
    else:
        ux = (bx1 + bx2) * 0.5
        uy = by1 + bh * 0.30
        u_hw = max(0.028, bw * 0.28)
        u_hh = max(0.032, bh * 0.16)

    ux1 = int(max(0, (ux - u_hw) * w))
    ux2 = int(min(w, (ux + u_hw) * w))
    uy1 = int(max(0, (uy - u_hh) * h))
    uy2 = int(min(h, (uy + u_hh) * h))
    collar_patch = frame_bgr[uy1:uy2, ux1:ux2] if (ux2 - ux1) >= 8 and (uy2 - uy1) >= 8 else None

    if ms is not None and mh is not None:
        cx = (ms[0] + mh[0]) * 0.5
        cy = (ms[1] + mh[1]) * 0.5
        half_w = max(0.03, bw * 0.28)
        half_h = max(0.04, abs(mh[1] - ms[1]) * 0.45, bh * 0.18)
        tx1 = int(max(0, (cx - half_w) * w))
        tx2 = int(min(w, (cx + half_w) * w))
        ty1 = int(max(0, (cy - half_h) * h))
        ty2 = int(min(h, (cy + half_h) * h))
    else:
        tx1 = int(max(0, (bx1 + bw * 0.2) * w))
        tx2 = int(min(w, (bx2 - bw * 0.2) * w))
        ty1 = int(max(0, (by1 + bh * 0.2) * h))
        ty2 = int(min(h, (by1 + bh * 0.6) * h))
    torso_patch = frame_bgr[ty1:ty2, tx1:tx2] if (tx2 - tx1) >= 10 and (ty2 - ty1) >= 10 else None
    return collar_patch, torso_patch


def compute_night_ir_appearance(frame_bgr: np.ndarray, obs: PoseObservation) -> np.ndarray | None:
    """
    Вычисляет 64-мерный ночной ИК-дескриптор одежды (LBP + Sobel + многозонный CLAHE):
    70% вес верхней зоны плеч/воротника + 30% вес торса.
    """
    collar_p, torso_p = _crop_collar_and_torso_patches(frame_bgr, obs)
    vec_collar = _extract_night_ir_patch_64(collar_p) if collar_p is not None else None
    vec_torso = _extract_night_ir_patch_64(torso_p) if torso_p is not None else None
    if vec_collar is not None and vec_torso is not None:
        blended = 0.70 * vec_collar + 0.30 * vec_torso
        norm = float(np.linalg.norm(blended))
        return (blended / norm).astype(np.float32) if norm > 1e-6 else vec_collar
    return vec_collar if vec_collar is not None else vec_torso


def compute_torso_appearance(frame_bgr: np.ndarray, obs: PoseObservation) -> np.ndarray | None:
    """
    Вычисляет дневной RGB/HSV дескриптор одежды (размерность 48):
    70% вес верхней части груди и плеч вокруг mid_shoulder (воротник/кофта, открытые даже под одеялом)
    + 30% вес общего торса.
    """
    collar_p, torso_p = _crop_collar_and_torso_patches(frame_bgr, obs)
    vec_collar = _extract_patch_descriptor_48(collar_p) if collar_p is not None else None
    vec_torso = _extract_patch_descriptor_48(torso_p) if torso_p is not None else None

    if vec_collar is not None and vec_torso is not None:
        blended = 0.70 * vec_collar + 0.30 * vec_torso
        norm = float(np.linalg.norm(blended))
        return (blended / norm).astype(np.float32) if norm > 1e-6 else vec_collar
    return vec_collar if vec_collar is not None else vec_torso


def cosine_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def _has_chromatic_signal(vec: np.ndarray | None) -> bool:
    """Проверяет наличие цветовой HS-компоненты в первых 24 бинах (отличает дневной RGB от ночного Ч/Б ИК)."""
    if vec is None or vec.shape[0] < 24:
        return False
    return float(np.linalg.norm(vec[:24])) >= 0.14


class FaceRegistry:
    """Локальная биометрия лиц на базе OpenCV Zoo YuNet + SFace (ONNX, без внешних C++ зависимостей)."""

    def __init__(self, faces_dir: Path, models_dir: Path, threshold: float = 0.38) -> None:
        _suppress_opencv_log_spam()
        self._faces_dir = faces_dir
        self._models_dir = models_dir
        self._threshold = threshold
        self._lock = threading.RLock()
        self._detector: cv2.FaceDetectorYN | None = None
        self._recognizer: cv2.FaceRecognizerSF | None = None
        self._models_ready = False
        self._embeddings: dict[str, dict[str, np.ndarray]] = {
            Role.PATIENT.value: {},
            Role.VISITOR.value: {},
        }
        for role_key in self._embeddings:
            (self._faces_dir / role_key).mkdir(parents=True, exist_ok=True)

    @property
    def faces_dir(self) -> Path:
        return self._faces_dir

    def ensure_models(self) -> bool:
        with self._lock:
            if self._models_ready:
                return True
            _suppress_opencv_log_spam()
            if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
                logger.warning("Текущая сборка OpenCV не поддерживает FaceDetectorYN/FaceRecognizerSF")
                return False
            self._models_dir.mkdir(parents=True, exist_ok=True)
            yunet_path = self._models_dir / "face_detection_yunet_2023mar.onnx"
            sface_path = self._models_dir / "face_recognition_sface_2021dec.onnx"
            try:
                for url, target in ((YUNET_URL, yunet_path), (SFACE_URL, sface_path)):
                    if not target.exists() or target.stat().st_size < 10_000:
                        logger.info("Загрузка ONNX-модели лиц %s ...", target.name)
                        tmp = target.with_suffix(".tmp")
                        req = urllib.request.Request(url, headers={"User-Agent": "FallGuard/1.0"})
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            tmp.write_bytes(resp.read())
                        tmp.replace(target)
                self._detector = cv2.FaceDetectorYN.create(
                    str(yunet_path), "", (320, 320), 0.48, 0.3, 5000
                )
                self._recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
                self._models_ready = True
                self.reload_gallery()
                return True
            except Exception as exc:
                logger.warning("Не удалось инициализировать модели лиц SFace: %s", exc)
                return False

    def _detect_best_face_and_embedding(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        if not self._models_ready or self._detector is None or self._recognizer is None:
            return None, None
        h, w = img_bgr.shape[:2]
        if h < 24 or w < 24:
            return None, None
        scale = 1.0
        work = img_bgr
        if max(h, w) > 640:
            scale = 640.0 / float(max(h, w))
            work = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        wh, ww = work.shape[:2]
        self._detector.setInputSize((ww, wh))
        _, faces = self._detector.detect(work)
        if faces is None or len(faces) == 0:
            return None, None
        best_face = max(faces, key=lambda f: float(f[2] * f[3]))
        aligned = self._recognizer.alignCrop(work, best_face)
        feat = self._recognizer.feature(aligned).flatten().astype(np.float32)
        inv_s = 1.0 / scale
        fx = int(max(0, best_face[0] * inv_s))
        fy = int(max(0, best_face[1] * inv_s))
        fw = int(max(1, best_face[2] * inv_s))
        fh = int(max(1, best_face[3] * inv_s))
        return feat, (fx, fy, fw, fh)

    def _detect_face_with_rot_code(
        self,
        img_bgr: np.ndarray,
        allow_rotations: bool = True,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, np.ndarray, int | None]:
        """
        Ищет лицо на прямом кропе (0°), а если не найдено и allow_rotations=True —
        дополнительно проверяет повороты +90° и -90° (для лежащего на подушке человека),
        возвращая также код применённого поворота OpenCV (или None для 0°).
        """
        emb, rect = self._detect_best_face_and_embedding(img_bgr)
        if emb is not None or not allow_rotations:
            return emb, rect, img_bgr, None

        for rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rotated = cv2.rotate(img_bgr, rot_code)
            emb_r, rect_r = self._detect_best_face_and_embedding(rotated)
            if emb_r is not None:
                return emb_r, rect_r, rotated, rot_code
        return None, None, img_bgr, None

    def _detect_face_with_rotations(
        self,
        img_bgr: np.ndarray,
        allow_rotations: bool = True,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, np.ndarray]:
        """
        Ищет лицо на прямом кропе (0°), а если не найдено и allow_rotations=True —
        дополнительно проверяет повороты +90° и -90° (для лежащего на подушке человека).
        """
        emb, rect, oriented_img, _ = self._detect_face_with_rot_code(
            img_bgr,
            allow_rotations=allow_rotations,
        )
        return emb, rect, oriented_img

    def _extract_embedding(self, img_bgr: np.ndarray, allow_rotations: bool = False) -> np.ndarray | None:
        emb, _, _, _ = self._detect_face_with_rot_code(img_bgr, allow_rotations=allow_rotations)
        return emb

    @staticmethod
    def _is_back_turned_posture(obs: PoseObservation) -> bool:
        """
        Гейт затылка (Волна 10): отсекает распознавание лица для человека, стоящего или сидящего
        спиной к камере (чтобы кроп не захватил лицо бабушки на подушке позади него):
        1) ни нос, ни глаза не видны (< 0.28), но видны оба плеча (>= 0.38) при вертикальном/сидящем корпусе;
        2) либо высокий стоящий силуэт с анатомической инверсией плеч спиной (l_sh.x + 0.035 < r_sh.x)
           при слабых фронтальных точках лица (< 0.52).
        """
        angle = obs.torso_angle_from_vertical_deg()
        if angle is None or angle >= 45.0 or obs.bbox_height < 0.34:
            return False
        l_sh = obs.kp(KP_L_SHOULDER, 0.38)
        r_sh = obs.kp(KP_R_SHOULDER, 0.38)
        if l_sh is None or r_sh is None:
            return False
        no_front_face = (
            obs.kp(KP_NOSE, 0.28) is None
            and obs.kp(KP_L_EYE, 0.28) is None
            and obs.kp(KP_R_EYE, 0.28) is None
        )
        if no_front_face:
            return True
        max_front_conf = max(
            (obs.keypoints_norm[i][2] if i < len(obs.keypoints_norm) else 0.0)
            for i in (KP_NOSE, KP_L_EYE, KP_R_EYE)
        )
        if (
            max_front_conf < 0.52
            and angle < 35.0
            and obs.bbox_height >= 0.50
            and obs.kp(KP_L_SHOULDER, 0.50) is not None
            and obs.kp(KP_R_SHOULDER, 0.50) is not None
            and (l_sh[0] + 0.035) < r_sh[0]
        ):
            return True
        return False

    @staticmethod
    def _map_face_center_to_frame_norm(
        rect: tuple[int, int, int, int],
        rot_code: int | None,
        crop_shape: tuple[int, int],
        crop_origin_px: tuple[int, int],
        frame_shape: tuple[int, int],
    ) -> tuple[float, float]:
        """Переводит центр найденного лица с (возможно повёрнутого) кропа в нормализованные координаты кадра."""
        fx, fy, fw, fh = rect
        u_rot = float(fx) + 0.5 * float(fw)
        v_rot = float(fy) + 0.5 * float(fh)
        ch, cw = int(crop_shape[0]), int(crop_shape[1])
        if rot_code == cv2.ROTATE_90_CLOCKWISE:
            u = v_rot
            v = float(ch) - u_rot
        elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            u = float(cw) - v_rot
            v = u_rot
        else:
            u = u_rot
            v = v_rot
        fh_px = max(1, int(frame_shape[0]))
        fw_px = max(1, int(frame_shape[1]))
        fcx = (float(crop_origin_px[0]) + u) / float(fw_px)
        fcy = (float(crop_origin_px[1]) + v) / float(fh_px)
        return fcx, fcy

    @staticmethod
    def _build_head_and_upper_crops(
        frame_bgr: np.ndarray,
        obs: PoseObservation,
    ) -> list[tuple[np.ndarray, bool]]:
        """Формирует кропы головы и верха бокса для ручной регистрации эталона (enroll_from_frame)."""
        h, w = frame_bgr.shape[:2]
        crops: list[tuple[np.ndarray, bool]] = []
        bx1, by1, bx2, by2 = obs.bbox_norm
        bw = max(0.05, bx2 - bx1)
        bh = max(0.05, by2 - by1)
        angle = obs.torso_angle_from_vertical_deg()
        allow_rot = angle is None or angle >= 42.0 or obs.bbox_height <= 0.34

        hc = obs.head_center
        if hc is not None:
            rad = max(0.075, min(0.20, max(bw * 0.45, min(bh, 0.32) * 0.34)))
            hx1 = int(max(0, (hc[0] - rad) * w))
            hx2 = int(min(w, (hc[0] + rad) * w))
            hy1 = int(max(0, (hc[1] - rad) * h))
            hy2 = int(min(h, (hc[1] + rad) * h))
            if (hx2 - hx1) >= 32 and (hy2 - hy1) >= 32:
                crops.append((frame_bgr[hy1:hy2, hx1:hx2], allow_rot))

        cx1 = int(max(0, (bx1 - bw * 0.14) * w))
        cx2 = int(min(w, (bx2 + bw * 0.14) * w))
        cy1 = int(max(0, (by1 - bh * 0.10) * h))
        cy2 = int(min(h, (by1 + min(bh * 0.48, 0.28)) * h))
        if (cx2 - cx1) >= 32 and (cy2 - cy1) >= 32:
            crops.append((frame_bgr[cy1:cy2, cx1:cx2], allow_rot))
        return crops

    @staticmethod
    def _build_strict_match_crops(
        frame_bgr: np.ndarray,
        obs: PoseObservation,
        on_bed_hint: bool = False,
    ) -> list[tuple[np.ndarray, tuple[int, int], bool]]:
        """
        Строит строго локализованный головной кроп для автоматического сопоставления в потоке:
        - при наличии obs.head_center используется ТОЛЬКО кроп вокруг головы (без широкого торсового 0.74*bh);
        - повороты ±90° разрешены только для поз на кровати / горизонтально лежащих поз (никогда для стоящего вне кровати!).
        """
        h, w = frame_bgr.shape[:2]
        bx1, by1, bx2, by2 = obs.bbox_norm
        bw = max(0.05, bx2 - bx1)
        bh = max(0.05, by2 - by1)
        angle = obs.torso_angle_from_vertical_deg()
        allow_rot = angle is None or angle >= 42.0 or obs.bbox_height <= 0.34 or on_bed_hint
        crops: list[tuple[np.ndarray, tuple[int, int], bool]] = []

        hc = obs.head_center
        if hc is not None:
            max_rad = 0.20 if allow_rot else 0.17
            rad = max(0.075, min(max_rad, max(bw * 0.45, min(bh, 0.49) * 0.34)))
            hx1 = int(max(0, (hc[0] - rad) * w))
            hx2 = int(min(w, (hc[0] + rad) * w))
            hy1 = int(max(0, (hc[1] - rad) * h))
            hy2 = int(min(h, (hc[1] + rad) * h))
            if (hx2 - hx1) >= 32 and (hy2 - hy1) >= 32:
                crops.append((frame_bgr[hy1:hy2, hx1:hx2], (hx1, hy1), allow_rot))
            return crops

        cx1 = int(max(0, (bx1 - bw * 0.08) * w))
        cx2 = int(min(w, (bx2 + bw * 0.08) * w))
        cy1 = int(max(0, (by1 - bh * 0.08) * h))
        cy2 = int(min(h, (by1 + min(bh * 0.34, 0.18)) * h))
        if (cx2 - cx1) >= 32 and (cy2 - cy1) >= 32:
            crops.append((frame_bgr[cy1:cy2, cx1:cx2], (cx1, cy1), allow_rot))
        return crops

    def reload_gallery(self) -> None:
        with self._lock:
            if not self._models_ready:
                return
            for role_key in (Role.PATIENT.value, Role.VISITOR.value):
                folder = self._faces_dir / role_key
                folder.mkdir(parents=True, exist_ok=True)
                loaded: dict[str, np.ndarray] = {}
                for img_path in sorted(folder.glob("*")):
                    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                        continue
                    data = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    emb = self._extract_embedding(img, allow_rotations=True)
                    if emb is not None:
                        loaded[img_path.name] = emb
                self._embeddings[role_key] = loaded

    def add_photo(self, role: Role | str, filename: str, raw_bytes: bytes) -> tuple[bool, str]:
        parsed_role = _coerce_role(role)
        if parsed_role is None:
            return False, "Допустимы только группы patient или visitor"
        with self._lock:
            if not self.ensure_models():
                return False, "Модели распознавания лиц недоступны"
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return False, "Файл не является корректным изображением"
            emb, _, oriented_img = self._detect_face_with_rotations(img, allow_rotations=True)
            if emb is None:
                return False, "На фото не обнаружено чёткое лицо — выберите более крупный снимок анфас"
            safe_stem = "".join(c for c in Path(filename).stem if c.isalnum() or c in ("-", "_"))[:32] or "face"
            out_name = f"{safe_stem}_{int(time.time() * 1000) % 1_000_000}.jpg"
            out_path = self._faces_dir / parsed_role.value / out_name
            ok, encoded = cv2.imencode(".jpg", oriented_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                return False, "Ошибка кодирования JPEG"
            out_path.write_bytes(encoded.tobytes())
            self._embeddings[parsed_role.value][out_name] = emb
            return True, out_name

    def enroll_from_frame(
        self,
        role: Role | str,
        frame_bgr: np.ndarray | None,
        obs: PoseObservation | None = None,
    ) -> tuple[bool, str]:
        """Захватывает эталон лица («Бабушка» или «Свой») прямо с живого кадра камеры."""
        parsed_role = _coerce_role(role)
        if parsed_role is None:
            return False, "Допустимы только группы patient или visitor"
        if frame_bgr is None or frame_bgr.ndim != 3 or min(frame_bgr.shape[:2]) < 32:
            return False, "Кадр с камеры пуст или ещё не получен"

        with self._lock:
            if not self.ensure_models():
                return False, "Модели распознавания лиц недоступны"

            h, w = frame_bgr.shape[:2]
            candidate_crops: list[tuple[np.ndarray, bool]] = []

            if obs is not None:
                candidate_crops.extend(self._build_head_and_upper_crops(frame_bgr, obs))
                bx1, by1, bx2, by2 = obs.bbox_norm
                bw = max(0.05, bx2 - bx1)
                bh = max(0.05, by2 - by1)
                fx1 = int(max(0, (bx1 - bw * 0.20) * w))
                fx2 = int(min(w, (bx2 + bw * 0.20) * w))
                fy1 = int(max(0, (by1 - bh * 0.15) * h))
                fy2 = int(min(h, (by2 + bh * 0.15) * h))
                if (fx2 - fx1) >= 32 and (fy2 - fy1) >= 32:
                    candidate_crops.append((frame_bgr[fy1:fy2, fx1:fx2], True))

            candidate_crops.append((frame_bgr, False))

            chosen_img: np.ndarray | None = None
            chosen_emb: np.ndarray | None = None

            for region, allow_rot in candidate_crops:
                emb, face_rect, oriented_region = self._detect_face_with_rotations(
                    region, allow_rotations=allow_rot
                )
                if emb is not None and face_rect is not None:
                    rx, ry, rw, rh = face_rect
                    rh_img, rw_img = oriented_region.shape[:2]
                    px1 = int(max(0, rx - rw * 0.55))
                    py1 = int(max(0, ry - rh * 0.55))
                    px2 = int(min(rw_img, rx + rw * 1.55))
                    py2 = int(min(rh_img, ry + rh * 1.55))
                    portrait = oriented_region[py1:py2, px1:px2]
                    chosen_img = (
                        portrait if (portrait.shape[0] >= 48 and portrait.shape[1] >= 48) else oriented_region
                    )
                    chosen_emb = emb
                    break

            if chosen_emb is None or chosen_img is None:
                return (
                    False,
                    "На текущем кадре камеры не видно открытого лица — повернитесь лицом к камере или добавьте свет",
                )

            out_name = f"cam_{parsed_role.value}_{int(time.time() * 1000) % 1_000_000}.jpg"
            out_path = self._faces_dir / parsed_role.value / out_name
            ok, encoded = cv2.imencode(".jpg", chosen_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if not ok:
                return False, "Ошибка сохранения снимка лица"
            out_path.write_bytes(encoded.tobytes())
            self._embeddings[parsed_role.value][out_name] = chosen_emb
            return True, out_name

    def delete_photo(self, role: Role | str, filename: str) -> bool:
        parsed_role = _coerce_role(role)
        if parsed_role is None:
            return False
        safe_name = Path(filename).name
        with self._lock:
            target = self._faces_dir / parsed_role.value / safe_name
            if target.exists():
                target.unlink()
            self._embeddings[parsed_role.value].pop(safe_name, None)
            return True

    def list_photos(self) -> dict[str, list[str]]:
        with self._lock:
            return {
                role_key: sorted(
                    p.name
                    for p in (self._faces_dir / role_key).glob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                )
                for role_key in (Role.PATIENT.value, Role.VISITOR.value)
            }

    @staticmethod
    def _is_face_spatially_owned(
        fcx: float,
        fcy: float,
        obs: PoseObservation,
        allow_rot: bool,
        other_observations: Sequence[PoseObservation] | None = None,
        zones: ZonesConfig | None = None,
        latched_patient_bed_bbox: tuple[float, float, float, float] | None = None,
    ) -> bool:
        """
        Проверяет, что найденное на кропе лицо физически принадлежит голове самого obs,
        а не соседнему человеку в кадре и не спящей на подушке бабушке позади стоящего гостя.
        """
        bx1, by1, bx2, by2 = obs.bbox_norm
        bw = max(0.05, bx2 - bx1)
        bh = max(0.05, by2 - by1)
        hc = obs.head_center

        own_dist: float | None = None
        if hc is not None:
            own_dist = math.hypot(fcx - hc[0], fcy - hc[1])
            max_own_dist = (
                max(0.085, min(0.145, max(bw, bh) * 0.48))
                if allow_rot
                else max(0.068, min(0.115, bw * 0.42))
            )
            if own_dist > max_own_dist:
                return False
        else:
            if not (
                (bx1 - 0.02) <= fcx <= (bx2 + 0.02)
                and (by1 - 0.03) <= fcy <= (by1 + min(0.28 * bh, 0.16))
            ):
                return False

        if other_observations:
            for other in other_observations:
                if other.track_id == obs.track_id:
                    continue
                other_hc = other.head_center
                if other_hc is not None:
                    other_dist = math.hypot(fcx - other_hc[0], fcy - other_hc[1])
                    if own_dist is not None:
                        if other_dist + 0.012 < own_dist:
                            return False
                    elif other_dist <= 0.12:
                        return False

        if zones is not None and zones.has_bed and not is_person_on_bed(obs, zones.bed_polygon, zones=zones):
            if latched_patient_bed_bbox is not None:
                lx1, ly1, lx2, ly2 = latched_patient_bed_bbox
                if (lx1 - 0.02) <= fcx <= (lx2 + 0.02) and (ly1 - 0.02) <= fcy <= (ly2 + 0.02):
                    return False
            _, bed_max_y = polygon_y_bounds(zones.bed_polygon)
            if by2 >= (bed_max_y + 0.05) and point_in_polygon((fcx, fcy), zones.bed_polygon):
                return False

        return True

    def match_person_crop(
        self,
        frame_bgr: np.ndarray,
        obs: PoseObservation,
        other_observations: Sequence[PoseObservation] | None = None,
        zones: ZonesConfig | None = None,
        latched_patient_bed_bbox: tuple[float, float, float, float] | None = None,
    ) -> tuple[Role, float]:
        with self._lock:
            if not self._models_ready:
                return Role.UNKNOWN, 0.0
            if not self._embeddings[Role.PATIENT.value] and not self._embeddings[Role.VISITOR.value]:
                return Role.UNKNOWN, 0.0
            if self._is_back_turned_posture(obs):
                return Role.UNKNOWN, 0.0

            fh_px, fw_px = frame_bgr.shape[:2]
            on_bed_hint = bool(
                zones is not None
                and zones.has_bed
                and is_person_on_bed(obs, zones.bed_polygon, zones=zones)
            )
            emb: np.ndarray | None = None
            matched_face_w: int = 0
            matched_rot_code: int | None = None
            for crop_img, origin_px, allow_rot in self._build_strict_match_crops(
                frame_bgr,
                obs,
                on_bed_hint=on_bed_hint,
            ):
                cand_emb, cand_rect, _, rot_code = self._detect_face_with_rot_code(
                    crop_img,
                    allow_rotations=allow_rot,
                )
                if cand_emb is None or cand_rect is None:
                    continue
                fcx, fcy = self._map_face_center_to_frame_norm(
                    rect=cand_rect,
                    rot_code=rot_code,
                    crop_shape=(crop_img.shape[0], crop_img.shape[1]),
                    crop_origin_px=origin_px,
                    frame_shape=(fh_px, fw_px),
                )
                if not self._is_face_spatially_owned(
                    fcx=fcx,
                    fcy=fcy,
                    obs=obs,
                    allow_rot=allow_rot,
                    other_observations=other_observations,
                    zones=zones,
                    latched_patient_bed_bbox=latched_patient_bed_bbox,
                ):
                    continue
                emb = cand_emb
                matched_face_w = int(cand_rect[2])
                matched_rot_code = rot_code
                break
            if emb is None:
                return Role.UNKNOWN, 0.0

            best_patient = max(
                (cosine_sim(emb, ref) for ref in self._embeddings[Role.PATIENT.value].values()),
                default=0.0,
            )
            best_visitor = max(
                (cosine_sim(emb, ref) for ref in self._embeddings[Role.VISITOR.value].values()),
                default=0.0,
            )

            if zones is not None and zones.has_bed:
                if on_bed_hint:
                    eff_pat_thr = max(self._threshold, 0.42)
                    if best_patient >= eff_pat_thr and best_patient >= best_visitor + 0.04:
                        return Role.PATIENT, best_patient

                    # Лежащий/полулежащий на матрасе трек, спальный ROI-трек (>= 9000) или повернутый ±90° кроп
                    # НИКОГДА не может получить Role.VISITOR ("face_visitor")!
                    angle_bed = obs.torso_angle_from_vertical_deg()
                    upright_frontal_on_bed = (
                        obs.track_id < 9000
                        and matched_rot_code is None
                        and angle_bed is not None
                        and angle_bed < 40.0
                        and obs.bbox_height > 0.35
                        and matched_face_w >= 44
                        and obs.kp(KP_NOSE, 0.40) is not None
                        and obs.kp(KP_L_EYE, 0.40) is not None
                        and obs.kp(KP_R_EYE, 0.40) is not None
                    )
                    if (
                        upright_frontal_on_bed
                        and best_visitor >= max(self._threshold, 0.50)
                        and best_visitor >= best_patient + 0.08
                    ):
                        return Role.VISITOR, best_visitor
                    return Role.UNKNOWN, max(best_patient, best_visitor)

                if (
                    matched_rot_code is None
                    and obs.track_id < 9000
                    and best_visitor >= self._threshold
                    and best_visitor >= best_patient + 0.02
                ):
                    return Role.VISITOR, best_visitor

                # Вне кровати: мелкий/боковой профиль никогда не должен превращать гостя на стуле или у двери в PATIENT!
                frontal_outside = (
                    obs.kp(KP_NOSE, 0.45) is not None
                    and obs.kp(KP_L_EYE, 0.45) is not None
                    and obs.kp(KP_R_EYE, 0.45) is not None
                )
                in_door_now = zones.has_door and is_pose_touching_door(
                    obs,
                    zones.door_polygon,
                    min_overlap=0.20,
                )
                if not frontal_outside or matched_face_w < 48:
                    return Role.UNKNOWN, best_visitor
                eff_outside_thr = max(self._threshold, 0.68 if in_door_now else 0.54)
                if best_patient >= eff_outside_thr and best_patient >= best_visitor + 0.08:
                    return Role.PATIENT, best_patient
                return Role.UNKNOWN, max(best_patient, best_visitor)

            if (
                matched_rot_code is None
                and obs.track_id < 9000
                and best_visitor >= self._threshold
                and best_visitor >= best_patient + 0.02
            ):
                return Role.VISITOR, best_visitor
            if best_patient >= self._threshold and best_patient > best_visitor:
                return Role.PATIENT, best_patient
            return Role.UNKNOWN, max(best_patient, best_visitor)

    MAX_AUTO_VISITOR_FACES: int = 24

    def auto_enroll_visitor_from_false_alert(
        self,
        frame_bgr: np.ndarray | None,
        obs: PoseObservation,
        zones: ZonesConfig,
        other_observations: Sequence[PoseObservation] | None = None,
        latched_patient_bed_bbox: tuple[float, float, float, float] | None = None,
        alert_id: str = "",
    ) -> tuple[bool, str, str]:
        """
        Безопасный авто-энролл лица «Своего» в data/faces/visitor/ при нажатии «❌ Ты ошибся»:
        строго только вне кровати, без поворотов ±90°, с открытой лицевой точкой, вне bed_polygon
        (с запасом 1.5%) и при полном отсутствии сходства с эталонами бабушки (< 0.48).
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or min(frame_bgr.shape[:2]) < 32:
            return False, "", "empty_frame"

        if zones.has_bed and is_person_on_bed(obs, zones.bed_polygon, zones=zones):
            return False, "", "on_bed"

        angle = obs.torso_angle_from_vertical_deg()
        if angle is not None and angle >= 42.0:
            return False, "", "horizontal_posture"
        if self._is_back_turned_posture(obs):
            return False, "", "back_turned"

        has_face_kp = (
            obs.kp(KP_NOSE, 0.35) is not None
            or obs.kp(KP_L_EYE, 0.35) is not None
            or obs.kp(KP_R_EYE, 0.35) is not None
        )
        if not has_face_kp:
            return False, "", "no_face_keypoints"

        with self._lock:
            if not self.ensure_models():
                return False, "", "models_unavailable"

            fh_px, fw_px = frame_bgr.shape[:2]
            strict_crops = self._build_strict_match_crops(frame_bgr, obs, on_bed_hint=False)
            if not strict_crops:
                return False, "", "no_head_crop"

            chosen_emb: np.ndarray | None = None
            chosen_img: np.ndarray | None = None
            last_reject = "no_face_detected"

            for crop_img, origin_px, _ in strict_crops:
                cand_emb, cand_rect, oriented_crop, rot_code = self._detect_face_with_rot_code(
                    crop_img,
                    allow_rotations=False,
                )
                if cand_emb is None or cand_rect is None:
                    continue

                rx, ry, rw, rh = cand_rect
                fcx, fcy = self._map_face_center_to_frame_norm(
                    rect=cand_rect,
                    rot_code=rot_code,
                    crop_shape=(crop_img.shape[0], crop_img.shape[1]),
                    crop_origin_px=origin_px,
                    frame_shape=(fh_px, fw_px),
                )

                if zones.has_bed and (
                    point_in_polygon((fcx, fcy), zones.bed_polygon)
                    or polygon_distance_norm((fcx, fcy), zones.bed_polygon) >= -0.015
                ):
                    last_reject = "inside_bed_polygon"
                    continue

                if latched_patient_bed_bbox is not None:
                    lx1, ly1, lx2, ly2 = latched_patient_bed_bbox
                    if (lx1 - 0.02) <= fcx <= (lx2 + 0.02) and (ly1 - 0.02) <= fcy <= (ly2 + 0.02):
                        last_reject = "inside_latched_bed_bbox"
                        continue

                if not self._is_face_spatially_owned(
                    fcx=fcx,
                    fcy=fcy,
                    obs=obs,
                    allow_rot=False,
                    other_observations=other_observations,
                    zones=zones,
                    latched_patient_bed_bbox=latched_patient_bed_bbox,
                ):
                    last_reject = "not_spatially_owned"
                    continue

                if rw < 36 or rh < 36:
                    last_reject = "face_too_small"
                    continue

                ch_px, cw_px = oriented_crop.shape[:2]
                px1 = int(max(0, rx - rw * 0.55))
                py1 = int(max(0, ry - rh * 0.55))
                px2 = int(min(cw_px, rx + rw * 1.55))
                py2 = int(min(ch_px, ry + rh * 1.55))
                portrait = oriented_crop[py1:py2, px1:px2]
                chosen_img = (
                    portrait
                    if (portrait.shape[0] >= 48 and portrait.shape[1] >= 48)
                    else oriented_crop
                )
                chosen_emb = cand_emb
                break

            if chosen_emb is None or chosen_img is None:
                return False, "", last_reject

            best_visitor = max(
                (cosine_sim(chosen_emb, ref) for ref in self._embeddings[Role.VISITOR.value].values()),
                default=0.0,
            )
            if best_visitor >= 0.82:
                return False, "", "already_known"

            best_patient = max(
                (cosine_sim(chosen_emb, ref) for ref in self._embeddings[Role.PATIENT.value].values()),
                default=0.0,
            )
            in_door_zone = bool(
                zones.has_door and is_pose_touching_door(obs, zones.door_polygon, min_overlap=0.15)
            )
            patient_co_present_on_bed = bool(
                latched_patient_bed_bbox is not None
                or (
                    zones.has_bed
                    and other_observations
                    and any(
                        is_person_on_bed(o, zones.bed_polygon, zones=zones)
                        for o in other_observations
                    )
                )
            )
            max_patient_sim = 0.58 if (in_door_zone or patient_co_present_on_bed) else 0.54
            if best_patient >= max_patient_sim:
                return False, "", "too_close_to_patient"

            visitor_dir = self._faces_dir / Role.VISITOR.value
            visitor_dir.mkdir(parents=True, exist_ok=True)
            safe_tag = "".join(
                c for c in str(alert_id or "").strip() if c.isalnum() or c in ("-", "_")
            )[:28] or "alert"
            ms_stamp = int(time.time() * 1000) % 1_000_000_000
            out_name = f"auto_visitor_{safe_tag}_{ms_stamp}.jpg"
            out_path = visitor_dir / out_name
            if out_path.exists():
                out_name = f"auto_visitor_{safe_tag}_{ms_stamp}_{obs.track_id}.jpg"
                out_path = visitor_dir / out_name

            ok, encoded = cv2.imencode(".jpg", chosen_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if not ok:
                return False, "", "jpeg_encode_error"
            out_path.write_bytes(encoded.tobytes())
            self._embeddings[Role.VISITOR.value][out_name] = chosen_emb

            auto_files = sorted(
                (
                    p
                    for p in visitor_dir.glob("auto_visitor_*")
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                ),
                key=lambda p: (p.stat().st_mtime_ns, p.name),
            )
            while len(auto_files) > self.MAX_AUTO_VISITOR_FACES:
                oldest_file = auto_files.pop(0)
                try:
                    oldest_file.unlink(missing_ok=True)
                except OSError:
                    pass
                self._embeddings[Role.VISITOR.value].pop(oldest_file.name, None)

            return True, out_name, "enrolled"


_HARD_VISITOR_REASONS = frozenset(
    {
        "door_entry",
        "door_visit_companion",
        "foreground_seated_companion",
        "border_standing_entry",
        "face_visitor",
        "manual_visitor_override",
        "visitor_outfit_match",
        "visitor_id_switch_continuity",
        "demoted_by_manual_override",
        "id_switch_demoted_visitor",
    }
)


@dataclass
class _RecentVisitorMemory:
    """Краткосрочная память недавно активных гостей для защиты от перескока ID у края кровати."""

    track_id: int
    last_seen: float
    last_center: tuple[float, float]
    bbox_norm: tuple[float, float, float, float]
    appearance_vec: np.ndarray | None = None
    appearance_vec_day: np.ndarray | None = None
    appearance_vec_night: np.ndarray | None = None
    reason: str = "visitor"


@dataclass
class _VisitorOutfitRecord:
    """Сохранённый профиль одежды родственника («Свой») сроком на 8 часов."""

    vec: np.ndarray
    updated_wall_ts: float


class PatientIdentifier:
    """
    5-слойный идентификатор «Только бабушка» с защитой от висящей одежды, ИК-бликов и теней:
    0) Фильтр ignore_polygons + StaticObjectGate + 4-кадровый гейт движения второго трека вне двери;
    1) Bed-Origin Lock vs Door/Border Entry + самоисцеление sole_bed_sleeper_recovery + 3D-гомография пола;
    2) Локальная биометрия лиц SFace («Бабушка» vs «Свои», вкл. повороты головы ±90° на подушке);
    3) Раздельные персистентные 12-часовые якоря бабушки (Day RGB/HSV 48D vs Night IR LBP+Sobel+CLAHE 64D)
       и раздельные 8-часовые галереи одежды гостей без смешивания при смене освещения;
    4) Совместный двухперсонный решатель _arbitrate_multi_person_roles + Caregiver Co-Presence.
    """

    def __init__(
        self,
        bed_lock_seconds: float = 20.0,
        initial_bed_lock_seconds: float = 5.0,
        face_registry: FaceRegistry | None = None,
        track_ttl_seconds: float = 15.0,
        state_path: Path | None = None,
        anchor_ttl_seconds: float = DEFAULT_ANCHOR_TTL_SECONDS,
        visitor_outfit_ttl_seconds: float = DEFAULT_VISITOR_OUTFIT_TTL_SECONDS,
        wall_clock: Callable[[], float] = time.time,
        door_visit_idle_timeout_sec: float = DEFAULT_DOOR_VISIT_IDLE_TIMEOUT,
    ) -> None:
        self._bed_lock_sec = bed_lock_seconds
        self._initial_bed_lock_sec = initial_bed_lock_seconds
        self._face_registry = face_registry
        self._track_ttl = track_ttl_seconds
        self._anchor_ttl_sec = anchor_ttl_seconds
        self._visitor_outfit_ttl_sec = visitor_outfit_ttl_seconds
        self._wall_clock = wall_clock
        self._door_visit_idle_timeout_sec = max(1.0, float(door_visit_idle_timeout_sec))
        self._lock = threading.RLock()

        if state_path is not None:
            self._state_path: Path | None = Path(state_path)
        elif face_registry is not None and hasattr(face_registry, "faces_dir"):
            self._state_path = face_registry.faces_dir.parent / "patient_anchor.json"
        else:
            self._state_path = None

        self._tracks: dict[int, TrackIdentity] = {}
        self._recent_visitors: dict[int, _RecentVisitorMemory] = {}
        self._visitor_memory_ttl: float = 35.0
        self._last_obs_by_id: dict[int, PoseObservation] = {}
        self._last_zones: ZonesConfig | None = None
        self._liveness_gate = StaticObjectGate(min_displacement=0.045, ttl_seconds=track_ttl_seconds)
        self._ever_locked_patient = False

        # Строго изолированные банки эталонов День (48D RGB/HSV) и Ночь ИК (64D LBP+Sobel+CLAHE)
        self._is_night_ir: bool = False
        self._anchor_vec_day: np.ndarray | None = None
        self._clean_bed_anchor_day: np.ndarray | None = None
        self._anchor_updated_at_day: float = -1e9
        self._anchor_wall_ts_day: float = 0.0
        self._restored_day_from_disk: bool = False
        self._visitor_outfits_day: list[_VisitorOutfitRecord] = []

        self._anchor_vec_night: np.ndarray | None = None
        self._clean_bed_anchor_night: np.ndarray | None = None
        self._anchor_updated_at_night: float = -1e9
        self._anchor_wall_ts_night: float = 0.0
        self._restored_night_from_disk: bool = False
        self._visitor_outfits_night: list[_VisitorOutfitRecord] = []

        self._last_patient_pos: tuple[float, float] | None = None
        self._last_patient_track_id: int | None = None
        self._last_patient_seen_at: float = -1e9

        # Закон сохранения Бабушки в кровати (Волна 7: patient_in_bed_latched)
        self._patient_in_bed_latched: bool = False
        self._latched_patient_bed_bbox: tuple[float, float, float, float] | None = None
        self._latched_patient_track_id: int = 1
        self._latched_bed_wall_ts: float = 0.0

        # Закон сохранения Своих через дверной портал (Волна 7: DoorPortalTracker)
        self._first_update_ts: float | None = None
        self._door_visit_active: bool = False
        self._door_visit_started_at: float = -1e9
        self._last_visitor_seen_at: float = -1e9
        self._last_visitor_at_door_ts: float = -1e9
        self._latched_visitor_bbox: tuple[float, float, float, float] | None = None
        self._last_visitor_roi_blur: np.ndarray | None = None
        self._last_visitor_frame_mean: float = 0.0
        self._last_visitor_roi_frame_id: int | None = None
        self._pre_visit_gray_small: np.ndarray | None = None
        self._pre_visit_frame_mean: float = 0.0
        self._visitor_entered_interior: bool = False
        self._visitor_spot_occupied: bool = False
        self._visitor_exiting_at_door: bool = False
        self._portal_prev_gray: np.ndarray | None = None
        self._portal_prev_gray_ts: float = -1e9
        self._portal_prev_textured: bool | None = None
        self._portal_last_frame_id: int | None = None
        self._sole_mover_at_door_ts: float = -1e9
        self._sole_mover_last_seen_ts: float = -1e9
        # Когда в кадре последний раз реально наблюдался второй человек (а не «защёлка» визита)
        self._other_person_seen_at: float = -1e9
        self._last_update_now: float = 0.0
        self._face_recheck_min_score: float = (
            float(getattr(face_registry, "_threshold", 0.38)) + 0.08
            if face_registry is not None
            else 0.46
        )

        self._load_anchor_state()

    @property
    def state_path(self) -> Path | None:
        return self._state_path

    CAREGIVER_VISIBLE_GRACE_SEC: float = 30.0

    @property
    def caregiver_visible(self) -> bool:
        """
        Помощник реально наблюдался в кадре в последние 30 с. В отличие от door_visit_active
        (может висеть часами, если выход гостя не замечен), этот флаг гаснет сам —
        и тогда падение пациента перестаёт подавляться.
        """
        with self._lock:
            return (self._last_update_now - self._other_person_seen_at) <= self.CAREGIVER_VISIBLE_GRACE_SEC

    @property
    def is_night_ir(self) -> bool:
        with self._lock:
            return self._is_night_ir

    @property
    def anchor_day_ready(self) -> bool:
        with self._lock:
            self._check_anchor_expiry()
            return self._anchor_vec_day is not None

    @property
    def anchor_night_ready(self) -> bool:
        with self._lock:
            self._check_anchor_expiry()
            return self._anchor_vec_night is not None

    @property
    def _anchor_vec(self) -> np.ndarray | None:
        """Совместимый геттер активного якоря текущей модальности (без смешивания day и night)."""
        if self._is_night_ir:
            return self._anchor_vec_night
        return self._anchor_vec_day

    @_anchor_vec.setter
    def _anchor_vec(self, value: np.ndarray | None) -> None:
        if value is None:
            self._anchor_vec_day = None
            self._anchor_updated_at_day = -1e9
            self._anchor_wall_ts_day = 0.0
            self._restored_day_from_disk = False
            self._anchor_vec_night = None
            self._anchor_updated_at_night = -1e9
            self._anchor_wall_ts_night = 0.0
            self._restored_night_from_disk = False
            return
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape == (NIGHT_DESCRIPTOR_DIM,):
            self._anchor_vec_night = arr.copy()
        elif arr.shape == (DAY_DESCRIPTOR_DIM,):
            self._anchor_vec_day = arr.copy()

    def _get_mode_anchor(self, night_ir: bool) -> np.ndarray | None:
        return self._anchor_vec_night if night_ir else self._anchor_vec_day

    def _get_mode_updated_at(self, night_ir: bool) -> float:
        return self._anchor_updated_at_night if night_ir else self._anchor_updated_at_day

    def _is_mode_restored_from_disk(self, night_ir: bool) -> bool:
        return self._restored_night_from_disk if night_ir else self._restored_day_from_disk

    def _set_mode_anchor(self, vec: np.ndarray, night_ir: bool, now: float) -> None:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return
        unit = (vec / norm).astype(np.float32)
        wall_ts = self._wall_clock()
        if night_ir and unit.shape == (NIGHT_DESCRIPTOR_DIM,):
            self._anchor_vec_night = unit
            self._anchor_updated_at_night = now
            self._anchor_wall_ts_night = wall_ts
        elif (not night_ir) and unit.shape == (DAY_DESCRIPTOR_DIM,):
            self._anchor_vec_day = unit
            self._anchor_updated_at_day = now
            self._anchor_wall_ts_day = wall_ts

    def _clear_mode_anchor(self, night_ir: bool) -> None:
        if night_ir:
            self._anchor_vec_night = None
            self._anchor_updated_at_night = -1e9
            self._anchor_wall_ts_night = 0.0
            self._restored_night_from_disk = False
        else:
            self._anchor_vec_day = None
            self._anchor_updated_at_day = -1e9
            self._anchor_wall_ts_day = 0.0
            self._restored_day_from_disk = False

    def _rollback_or_clear_mode_anchor_locked(
        self,
        night_ir: bool,
        polluted_vec: np.ndarray | None = None,
        force: bool = False,
        now: float | None = None,
    ) -> bool:
        """
        Откатывает загрязнённый якорь внешности к последнему чистому спальному эталону
        (_clean_bed_anchor_day / _clean_bed_anchor_night), либо очищает его, если чистый эталон отсутствует.
        """
        cur = self._anchor_vec_night if night_ir else self._anchor_vec_day
        if cur is None:
            return False
        if not force and polluted_vec is not None and cosine_sim(cur, polluted_vec) < 0.78:
            return False
        clean = self._clean_bed_anchor_night if night_ir else self._clean_bed_anchor_day
        ts = now if now is not None else time.monotonic()
        if clean is not None and (polluted_vec is None or cosine_sim(clean, polluted_vec) < 0.80):
            self._set_mode_anchor(clean, night_ir=night_ir, now=ts)
            return True
        if night_ir:
            self._clean_bed_anchor_night = None
        else:
            self._clean_bed_anchor_day = None
        self._clear_mode_anchor(night_ir)
        return True

    def rollback_or_clear_polluted_anchor(
        self,
        visitor_track_id: int | None = None,
        now: float | None = None,
    ) -> bool:
        """Публичное самоисцеление загрязнённого якоря одежды при отметке ложной тревоги («Ты ошибся»)."""
        ts = now if now is not None else time.monotonic()
        with self._lock:
            t_day: np.ndarray | None = None
            t_night: np.ndarray | None = None
            force_rb = visitor_track_id is None
            if visitor_track_id is not None:
                tr = self._tracks.get(visitor_track_id)
                if tr is not None:
                    t_day = self._get_track_mode_vec(tr, night_ir=False)
                    t_night = self._get_track_mode_vec(tr, night_ir=True)
                    if tr.role == Role.PATIENT or self._last_patient_track_id == visitor_track_id:
                        force_rb = True
            changed = False
            if t_day is not None or (force_rb and not self._is_night_ir):
                if self._rollback_or_clear_mode_anchor_locked(
                    False,
                    polluted_vec=t_day,
                    force=force_rb,
                    now=ts,
                ):
                    changed = True
            if t_night is not None or (force_rb and self._is_night_ir):
                if self._rollback_or_clear_mode_anchor_locked(
                    True,
                    polluted_vec=t_night,
                    force=force_rb,
                    now=ts,
                ):
                    changed = True
            if changed:
                self._save_anchor_state()
            return changed

    @staticmethod
    def _get_track_mode_vec(tr: TrackIdentity, night_ir: bool) -> np.ndarray | None:
        if night_ir:
            if tr.appearance_vec_night is not None:
                return tr.appearance_vec_night
            if tr.appearance_vec is not None and tr.appearance_vec.shape == (NIGHT_DESCRIPTOR_DIM,):
                return tr.appearance_vec
            return None
        if tr.appearance_vec_day is not None:
            return tr.appearance_vec_day
        if tr.appearance_vec is not None and tr.appearance_vec.shape == (DAY_DESCRIPTOR_DIM,):
            return tr.appearance_vec
        return None

    @property
    def patient_in_bed_latched(self) -> bool:
        with self._lock:
            self._check_anchor_expiry()
            return self._patient_in_bed_latched

    @patient_in_bed_latched.setter
    def patient_in_bed_latched(self, value: bool) -> None:
        with self._lock:
            self._patient_in_bed_latched = bool(value)
            if not self._patient_in_bed_latched:
                self._latched_bed_wall_ts = 0.0
            elif self._latched_bed_wall_ts <= 0.0:
                self._latched_bed_wall_ts = self._wall_clock()

    @property
    def latched_patient_bed_bbox(self) -> tuple[float, float, float, float] | None:
        with self._lock:
            self._check_anchor_expiry()
            return self._latched_patient_bed_bbox if self._patient_in_bed_latched else None

    @latched_patient_bed_bbox.setter
    def latched_patient_bed_bbox(self, value: tuple[float, float, float, float] | Sequence[float] | None) -> None:
        with self._lock:
            if value is None:
                self._latched_patient_bed_bbox = None
            else:
                vals = tuple(float(v) for v in value)
                if len(vals) == 4:
                    self._latched_patient_bed_bbox = (vals[0], vals[1], vals[2], vals[3])

    @property
    def latched_patient_track_id(self) -> int:
        with self._lock:
            return int(self._latched_patient_track_id)

    @latched_patient_track_id.setter
    def latched_patient_track_id(self, value: int) -> None:
        with self._lock:
            self._latched_patient_track_id = int(value)

    def clear_bed_presence(self) -> None:
        """Ручной сброс защёлки присутствия бабушки в кровати («Кровать пуста»)."""
        with self._lock:
            was_latched = self._patient_in_bed_latched or (self._latched_patient_bed_bbox is not None)
            self._patient_in_bed_latched = False
            self._latched_patient_bed_bbox = None
            self._latched_bed_wall_ts = 0.0
            self._sole_mover_at_door_ts = -1e9
            self._sole_mover_last_seen_ts = -1e9
            for tr in self._tracks.values():
                if tr.role == Role.PATIENT:
                    tr.role = Role.UNKNOWN
                    tr.reason = "bed_presence_cleared"
                    tr.bed_accum_sec = 0.0
            self._last_patient_track_id = None
            self._last_patient_pos = None
            self._last_patient_seen_at = -1e9
            if was_latched:
                self._save_anchor_state()

    def restore_bed_presence(self) -> None:
        """Восстанавливает защёлку присутствия бабушки в кровати после отметки ложной тревоги («Ты ошибся»)."""
        with self._lock:
            self._patient_in_bed_latched = True
            self._latched_bed_wall_ts = self._wall_clock()
            self._sole_mover_at_door_ts = -1e9
            self._sole_mover_last_seen_ts = -1e9
            self._ever_locked_patient = True
            if self._latched_patient_bed_bbox is None and self._last_zones is not None and self._last_zones.has_bed:
                xs = [float(p[0]) for p in self._last_zones.bed_polygon]
                ys = [float(p[1]) for p in self._last_zones.bed_polygon]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                bw = max(0.06, max_x - min_x)
                bh = max(0.06, max_y - min_y)
                self._latched_patient_bed_bbox = (
                    max(0.0, min_x + 0.12 * bw),
                    max(0.0, min_y + 0.15 * bh),
                    min(1.0, max_x - 0.12 * bw),
                    min(1.0, max_y - 0.15 * bh),
                )
            if self._last_patient_pos is None and self._latched_patient_bed_bbox is not None:
                bx1, by1, bx2, by2 = self._latched_patient_bed_bbox
                self._last_patient_pos = ((bx1 + bx2) * 0.5, (by1 + by2) * 0.5)
            self._save_anchor_state()

    def _reset_door_visit_portal(self) -> None:
        self._door_visit_active = False
        self._door_visit_started_at = -1e9
        self._last_visitor_seen_at = -1e9
        self._last_visitor_at_door_ts = -1e9
        self._latched_visitor_bbox = None
        self._last_visitor_roi_blur = None
        self._last_visitor_frame_mean = 0.0
        self._last_visitor_roi_frame_id = None
        self._visitor_entered_interior = False
        self._visitor_spot_occupied = False
        self._visitor_exiting_at_door = False

    @property
    def door_visit_active(self) -> bool:
        with self._lock:
            return self._door_visit_active

    @door_visit_active.setter
    def door_visit_active(self, value: bool) -> None:
        with self._lock:
            if not bool(value):
                self._reset_door_visit_portal()
            else:
                self._door_visit_active = True

    @property
    def latched_visitor_bbox(self) -> tuple[float, float, float, float] | None:
        with self._lock:
            return self._latched_visitor_bbox if self._door_visit_active else None

    @latched_visitor_bbox.setter
    def latched_visitor_bbox(self, value: tuple[float, float, float, float] | Sequence[float] | None) -> None:
        with self._lock:
            if value is None:
                self._latched_visitor_bbox = None
            else:
                vals = tuple(float(v) for v in value)
                if len(vals) == 4:
                    self._latched_visitor_bbox = (vals[0], vals[1], vals[2], vals[3])

    @property
    def visitor_spot_occupied(self) -> bool:
        with self._lock:
            return bool(self._door_visit_active and self._visitor_spot_occupied)

    @visitor_spot_occupied.setter
    def visitor_spot_occupied(self, value: bool) -> None:
        with self._lock:
            self._visitor_spot_occupied = bool(value)

    def clear_door_visit(self) -> None:
        """Принудительно сбрасывает режим дверного визита родственника и возвращает охрану кровати."""
        with self._lock:
            self._reset_door_visit_portal()

    def _activate_door_visit(self, now: float) -> None:
        if not self._door_visit_active:
            self._door_visit_active = True
            self._door_visit_started_at = now
            self._last_visitor_at_door_ts = -1e9
            self._visitor_entered_interior = False
            self._visitor_exiting_at_door = False
        self._last_visitor_seen_at = now

    @staticmethod
    def _is_touching_door(obs: PoseObservation, zones: ZonesConfig) -> bool:
        if not zones.has_door:
            return False
        return is_pose_touching_door(obs, zones.door_polygon, min_overlap=0.20)

    @property
    def has_appearance_anchor(self) -> bool:
        with self._lock:
            self._check_anchor_expiry()
            return (self._anchor_vec_day is not None) or (self._anchor_vec_night is not None)

    @property
    def visitor_outfit_count(self) -> int:
        with self._lock:
            self._check_anchor_expiry()
            return len(self._visitor_outfits_day) + len(self._visitor_outfits_night)

    def clear_anchor(self) -> None:
        """
        Удаляет дневной и ночной якоря внешности в памяти и patient_anchor.json на диске,
        сбрасывает закэшированные роли текущих треков и историю статических фантомов.
        """
        with self._lock:
            self._clean_bed_anchor_day = None
            self._clean_bed_anchor_night = None
            self._clear_mode_anchor(False)
            self._clear_mode_anchor(True)
            self._ever_locked_patient = False
            self._last_patient_pos = None
            self._last_patient_track_id = None
            self._last_patient_seen_at = -1e9
            self._patient_in_bed_latched = False
            self._latched_patient_bed_bbox = None
            self._latched_bed_wall_ts = 0.0
            self._first_update_ts = None
            self._sole_mover_at_door_ts = -1e9
            self._sole_mover_last_seen_ts = -1e9
            self._visitor_outfits_day.clear()
            self._visitor_outfits_night.clear()
            if self._state_path is not None and self._state_path.exists():
                try:
                    self._state_path.unlink()
                except Exception as exc:
                    logger.warning("Не удалось удалить якорь %s: %s", self._state_path, exc)
            for tr in self._tracks.values():
                if tr.role == Role.PATIENT or tr.reason in ("visitor_id_switch_continuity", "co_present_with_patient"):
                    tr.role = Role.UNKNOWN
                    tr.reason = "anchor_cleared"
                    tr.bed_accum_sec = 0.0
            self._recent_visitors.clear()
            self._reset_door_visit_portal()
            self._portal_prev_gray = None
            self._portal_prev_gray_ts = -1e9
            self._portal_prev_textured = None
            self._portal_last_frame_id = None
            self._pre_visit_gray_small = None
            self._pre_visit_frame_mean = 0.0
            self._liveness_gate.reset()

    def _load_outfit_bank_from_raw(
        self,
        raw_list: object,
        expected_dim: int,
        now_wall: float,
    ) -> list[_VisitorOutfitRecord]:
        loaded: list[_VisitorOutfitRecord] = []
        if not isinstance(raw_list, list):
            return loaded
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            v_ts = float(item.get("updated_wall_ts", 0.0))
            v_age = now_wall - v_ts
            if v_ts <= 0.0 or v_age > self._visitor_outfit_ttl_sec or v_age < -300.0:
                continue
            v_list = item.get("vec")
            if not isinstance(v_list, list) or len(v_list) != expected_dim:
                continue
            v_arr = np.asarray(v_list, dtype=np.float32)
            v_norm = float(np.linalg.norm(v_arr))
            if v_norm > 1e-6:
                loaded.append(_VisitorOutfitRecord(vec=(v_arr / v_norm).astype(np.float32), updated_wall_ts=v_ts))
        return loaded[-MAX_VISITOR_OUTFITS:]

    def _load_anchor_state(self) -> None:
        """Загружает раздельные дневной (RGB 48D) и ночной (ИК LBP 64D) якоря бабушки и галереи гостей с диска."""
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            schema_ver = int(raw.get("schema_version", 1))
            if schema_ver not in (3, ANCHOR_SCHEMA_VERSION):
                logger.info("Схема якоря одежды обновилась (%s -> %s) — старый якорь сброшен", schema_ver, ANCHOR_SCHEMA_VERSION)
                return

            now_wall = self._wall_clock()

            # 1. Раздельные галереи одежды родственников («Свой», TTL 8 часов): День (48D) и Ночь ИК (64D)
            raw_vis_day = raw.get("visitor_vecs_day")
            if raw_vis_day is None:
                raw_vis_day = raw.get("visitor_vecs")
            self._visitor_outfits_day = self._load_outfit_bank_from_raw(raw_vis_day, DAY_DESCRIPTOR_DIM, now_wall)
            if schema_ver == 3 and self._visitor_outfits_day:
                self._visitor_outfits_day = [
                    rec for rec in self._visitor_outfits_day if _has_chromatic_signal(rec.vec)
                ]
            self._visitor_outfits_night = self._load_outfit_bank_from_raw(
                raw.get("visitor_vecs_night"),
                NIGHT_DESCRIPTOR_DIM,
                now_wall,
            )

            # 2. Дневной цветовой якорь бабушки (48D RGB+HS)
            day_ts = float(raw.get("updated_wall_ts_day", raw.get("updated_wall_ts", 0.0)) or 0.0)
            day_vec_list = raw.get("anchor_vec_day")
            if day_vec_list is None:
                day_vec_list = raw.get("anchor_vec")
            day_age = now_wall - day_ts
            if (
                day_ts > 0.0
                and -300.0 <= day_age <= self._anchor_ttl_sec
                and isinstance(day_vec_list, list)
                and len(day_vec_list) == DAY_DESCRIPTOR_DIM
            ):
                d_vec = np.asarray(day_vec_list, dtype=np.float32)
                d_norm = float(np.linalg.norm(d_vec))
                if d_norm >= 1e-6 and (schema_ver >= ANCHOR_SCHEMA_VERSION or _has_chromatic_signal(d_vec)):
                    self._anchor_vec_day = (d_vec / d_norm).astype(np.float32)
                    self._anchor_wall_ts_day = day_ts
                    self._restored_day_from_disk = True

            # 3. Ночной текстурный ИК-якорь бабушки (64D LBP+Sobel+CLAHE)
            night_ts = float(raw.get("updated_wall_ts_night", 0.0) or 0.0)
            night_vec_list = raw.get("anchor_vec_night")
            night_age = now_wall - night_ts
            if (
                night_ts > 0.0
                and -300.0 <= night_age <= self._anchor_ttl_sec
                and isinstance(night_vec_list, list)
                and len(night_vec_list) == NIGHT_DESCRIPTOR_DIM
            ):
                n_vec = np.asarray(night_vec_list, dtype=np.float32)
                n_norm = float(np.linalg.norm(n_vec))
                if n_norm >= 1e-6:
                    self._anchor_vec_night = (n_vec / n_norm).astype(np.float32)
                    self._anchor_wall_ts_night = night_ts
                    self._restored_night_from_disk = True

            # 4. Закон сохранения Бабушки в кровати (patient_in_bed_latched, TTL 12 часов)
            bed_latch_flag = bool(raw.get("patient_in_bed_latched", False))
            bed_latch_ts = float(
                raw.get("updated_wall_ts_bed_latch")
                or raw.get("updated_wall_ts_day")
                or raw.get("updated_wall_ts_night")
                or raw.get("updated_wall_ts")
                or 0.0
            )
            bed_latch_age = now_wall - bed_latch_ts
            if bed_latch_flag and bed_latch_ts > 0.0 and -300.0 <= bed_latch_age <= self._anchor_ttl_sec:
                self._patient_in_bed_latched = True
                self._latched_bed_wall_ts = bed_latch_ts
                if self._anchor_vec_day is not None:
                    self._clean_bed_anchor_day = self._anchor_vec_day.copy()
                if self._anchor_vec_night is not None:
                    self._clean_bed_anchor_night = self._anchor_vec_night.copy()
                raw_bbox = raw.get("latched_patient_bed_bbox")
                if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                    self._latched_patient_bed_bbox = (
                        float(raw_bbox[0]),
                        float(raw_bbox[1]),
                        float(raw_bbox[2]),
                        float(raw_bbox[3]),
                    )
                raw_tid = raw.get("latched_patient_track_id")
                if isinstance(raw_tid, int) and raw_tid > 0:
                    self._latched_patient_track_id = int(raw_tid)

            if (
                self._anchor_vec_day is not None
                or self._anchor_vec_night is not None
                or self._patient_in_bed_latched
            ):
                pos = raw.get("last_patient_pos")
                if isinstance(pos, list) and len(pos) == 2:
                    self._last_patient_pos = (float(pos[0]), float(pos[1]))
                elif self._latched_patient_bed_bbox is not None:
                    bx1, by1, bx2, by2 = self._latched_patient_bed_bbox
                    self._last_patient_pos = ((bx1 + bx2) * 0.5, (by1 + by2) * 0.5)
                self._ever_locked_patient = True
                logger.info(
                    "Загружены персистентные якоря бабушки (day=%s, night=%s, bed_latched=%s, гостей day=%d/night=%d)",
                    self._anchor_vec_day is not None,
                    self._anchor_vec_night is not None,
                    self._patient_in_bed_latched,
                    len(self._visitor_outfits_day),
                    len(self._visitor_outfits_night),
                )
        except Exception as exc:
            logger.warning("Не удалось прочитать якорь одежды %s: %s", self._state_path, exc)

    def _save_anchor_state(self) -> None:
        if self._state_path is None:
            return
        has_any_data = (
            self._anchor_vec_day is not None
            or self._anchor_vec_night is not None
            or bool(self._visitor_outfits_day)
            or bool(self._visitor_outfits_night)
            or self._patient_in_bed_latched
        )
        if not has_any_data:
            if self._state_path.exists():
                try:
                    self._state_path.unlink()
                except Exception:
                    pass
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            wall_ts = self._wall_clock()
            if self._anchor_vec_day is not None and self._anchor_wall_ts_day <= 0.0:
                self._anchor_wall_ts_day = wall_ts
            if self._anchor_vec_night is not None and self._anchor_wall_ts_night <= 0.0:
                self._anchor_wall_ts_night = wall_ts
            if self._patient_in_bed_latched and self._latched_bed_wall_ts <= 0.0:
                self._latched_bed_wall_ts = wall_ts

            day_list = (
                [round(float(x), 6) for x in self._anchor_vec_day.tolist()]
                if self._anchor_vec_day is not None
                else None
            )
            night_list = (
                [round(float(x), 6) for x in self._anchor_vec_night.tolist()]
                if self._anchor_vec_night is not None
                else None
            )
            vis_day_serialized = [
                {
                    "vec": [round(float(x), 6) for x in rec.vec.tolist()],
                    "updated_wall_ts": round(rec.updated_wall_ts, 3),
                }
                for rec in self._visitor_outfits_day[-MAX_VISITOR_OUTFITS:]
            ]
            vis_night_serialized = [
                {
                    "vec": [round(float(x), 6) for x in rec.vec.tolist()],
                    "updated_wall_ts": round(rec.updated_wall_ts, 3),
                }
                for rec in self._visitor_outfits_night[-MAX_VISITOR_OUTFITS:]
            ]
            has_pat_anchor = (
                (self._anchor_vec_day is not None)
                or (self._anchor_vec_night is not None)
                or self._patient_in_bed_latched
            )
            payload = {
                "schema_version": ANCHOR_SCHEMA_VERSION,
                # Legacy-совместимые ключи зеркалируют СТРОГО дневной 48D банк (никогда не смешиваясь с ночным 64D)
                "updated_wall_ts": round(self._anchor_wall_ts_day if self._anchor_vec_day is not None else 0.0, 3),
                "anchor_vec": day_list,
                "visitor_vecs": vis_day_serialized,
                # Раздельные банки День (48D RGB/HSV) и Ночь ИК (64D LBP+Sobel+CLAHE)
                "updated_wall_ts_day": round(self._anchor_wall_ts_day if self._anchor_vec_day is not None else 0.0, 3),
                "anchor_vec_day": day_list,
                "visitor_vecs_day": vis_day_serialized,
                "updated_wall_ts_night": round(
                    self._anchor_wall_ts_night if self._anchor_vec_night is not None else 0.0,
                    3,
                ),
                "anchor_vec_night": night_list,
                "visitor_vecs_night": vis_night_serialized,
                "last_patient_pos": (
                    [round(self._last_patient_pos[0], 5), round(self._last_patient_pos[1], 5)]
                    if self._last_patient_pos is not None and has_pat_anchor
                    else None
                ),
                "patient_in_bed_latched": bool(self._patient_in_bed_latched),
                "latched_patient_bed_bbox": (
                    [round(float(v), 5) for v in self._latched_patient_bed_bbox]
                    if self._latched_patient_bed_bbox is not None and self._patient_in_bed_latched
                    else None
                ),
                "latched_patient_track_id": int(self._latched_patient_track_id),
                "updated_wall_ts_bed_latch": round(
                    self._latched_bed_wall_ts if self._patient_in_bed_latched else 0.0,
                    3,
                ),
            }
            tmp_path = self._state_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(self._state_path)
        except Exception as exc:
            logger.warning("Не удалось сохранить якорь одежды %s: %s", self._state_path, exc)

    def _check_anchor_expiry(self) -> None:
        now_wall = self._wall_clock()
        if self._anchor_vec_day is not None and self._anchor_wall_ts_day > 0.0:
            if (now_wall - self._anchor_wall_ts_day) > self._anchor_ttl_sec:
                self._clear_mode_anchor(False)
        if self._anchor_vec_night is not None and self._anchor_wall_ts_night > 0.0:
            if (now_wall - self._anchor_wall_ts_night) > self._anchor_ttl_sec:
                self._clear_mode_anchor(True)
        if self._patient_in_bed_latched and self._latched_bed_wall_ts > 0.0:
            if (now_wall - self._latched_bed_wall_ts) > self._anchor_ttl_sec:
                self._patient_in_bed_latched = False
                self._latched_patient_bed_bbox = None
                self._latched_bed_wall_ts = 0.0
        if self._visitor_outfits_day:
            self._visitor_outfits_day = [
                rec
                for rec in self._visitor_outfits_day
                if (now_wall - rec.updated_wall_ts) <= self._visitor_outfit_ttl_sec
            ]
        if self._visitor_outfits_night:
            self._visitor_outfits_night = [
                rec
                for rec in self._visitor_outfits_night
                if (now_wall - rec.updated_wall_ts) <= self._visitor_outfit_ttl_sec
            ]

    def _remember_visitor_outfit(self, vec: np.ndarray, night_ir: bool | None = None) -> None:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-6:
            return
        unit = (arr / norm).astype(np.float32)
        if unit.shape == (NIGHT_DESCRIPTOR_DIM,):
            bank = self._visitor_outfits_night
        elif unit.shape == (DAY_DESCRIPTOR_DIM,):
            if night_ir is True:
                return
            bank = self._visitor_outfits_day
        else:
            return

        wall_ts = self._wall_clock()
        for idx, rec in enumerate(bank):
            if cosine_sim(rec.vec, unit) >= 0.88:
                blended = 0.5 * rec.vec + 0.5 * unit
                b_norm = float(np.linalg.norm(blended))
                bank[idx] = _VisitorOutfitRecord(
                    vec=(blended / b_norm).astype(np.float32) if b_norm > 1e-6 else unit,
                    updated_wall_ts=wall_ts,
                )
                return
        bank.append(_VisitorOutfitRecord(vec=unit, updated_wall_ts=wall_ts))
        if len(bank) > MAX_VISITOR_OUTFITS:
            del bank[:-MAX_VISITOR_OUTFITS]

    def _max_visitor_outfit_sim(self, vec: np.ndarray | None, night_ir: bool | None = None) -> float:
        if vec is None:
            return 0.0
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape == (NIGHT_DESCRIPTOR_DIM,):
            if night_ir is False or not self._visitor_outfits_night:
                return 0.0
            return max((cosine_sim(arr, rec.vec) for rec in self._visitor_outfits_night), default=0.0)
        if arr.shape != (DAY_DESCRIPTOR_DIM,) or night_ir is True or not self._visitor_outfits_day:
            return 0.0

        vec_chroma = _has_chromatic_signal(arr)
        best_sim = 0.0
        for rec in self._visitor_outfits_day:
            rec_chroma = _has_chromatic_signal(rec.vec)
            # Дневной RGB и монохромный сигнал несопоставимы по цветовым бинам
            if vec_chroma != rec_chroma:
                continue
            raw_sim = cosine_sim(arr, rec.vec)
            eff_sim = raw_sim if vec_chroma else max(0.0, raw_sim - 0.10)
            if eff_sim > best_sim:
                best_sim = eff_sim
        return best_sim

    def force_assign_patient(self, track_id: int, now: float | None = None) -> bool:
        """Ручная привязка трека как бабушки из веб-пульта (обновляет якорь строго текущей модальности день/ночь)."""
        ts = now if now is not None else time.monotonic()
        with self._lock:
            target = self._tracks.get(track_id)
            if target is None:
                return False
            for tid, tr in self._tracks.items():
                if tid != track_id and tr.role == Role.PATIENT:
                    tr.role = Role.VISITOR
                    tr.reason = "demoted_by_manual_override"
            target.role = Role.PATIENT
            target.reason = "manual_override"
            target.has_moved = True
            target.manual_assigned_at = ts
            self._recent_visitors.pop(track_id, None)
            has_other_visitor = any(
                tid != track_id and tr.role == Role.VISITOR and (ts - tr.last_seen) <= 3.0
                for tid, tr in self._tracks.items()
            )
            if not has_other_visitor:
                self._reset_door_visit_portal()
            self._ever_locked_patient = True
            self._last_patient_track_id = track_id
            self._last_patient_pos = target.last_center
            self._last_patient_seen_at = ts

            saved_any = False
            obs_t = self._last_obs_by_id.get(track_id)
            zones_t = self._last_zones
            if obs_t is not None and zones_t is not None and zones_t.has_bed:
                if is_person_on_bed(obs_t, zones_t.bed_polygon, zones=zones_t):
                    self._patient_in_bed_latched = True
                    self._latched_patient_track_id = track_id
                    self._latched_patient_bed_bbox = obs_t.bbox_norm
                    self._latched_bed_wall_ts = self._wall_clock()
                    saved_any = True
            day_vec = self._get_track_mode_vec(target, night_ir=False)
            if day_vec is not None and (not self._is_night_ir or target.appearance_vec_day is not None):
                self._visitor_outfits_day = [
                    rec for rec in self._visitor_outfits_day if cosine_sim(rec.vec, day_vec) < 0.85
                ]
                self._set_mode_anchor(day_vec, night_ir=False, now=ts)
                saved_any = True
            night_vec = self._get_track_mode_vec(target, night_ir=True)
            if night_vec is not None and (self._is_night_ir or target.appearance_vec_night is not None):
                self._visitor_outfits_night = [
                    rec for rec in self._visitor_outfits_night if cosine_sim(rec.vec, night_vec) < 0.85
                ]
                self._set_mode_anchor(night_vec, night_ir=True, now=ts)
                saved_any = True
            if saved_any:
                self._save_anchor_state()
            return True

    def force_assign_visitor(self, track_id: int, now: float | None = None) -> bool:
        """
        Ручная привязка трека как «Свой» (родственник/помощник) из веб-пульта:
        запоминает одежду гостя на 8 часов в соответствующий банк (день 48D / ночь 64D),
        очищает загрязнённый якорь ТОЛЬКО совпадающей модальности и мгновенно повышает
        другого реально лежащего в кровати человека до Role.PATIENT.
        """
        ts = now if now is not None else time.monotonic()
        with self._lock:
            target = self._tracks.get(track_id)
            if target is None:
                return False

            was_patient = target.role == Role.PATIENT or self._last_patient_track_id == track_id
            target.role = Role.VISITOR
            target.reason = "manual_visitor_override"
            target.confirmed_second_track = True
            target.bed_accum_sec = 0.0
            target.was_on_bed = False
            target.manual_assigned_at = ts
            self._activate_door_visit(ts)

            if was_patient:
                self._last_patient_track_id = None
                self._last_patient_pos = None
                self._last_patient_seen_at = -1e9
                self._patient_in_bed_latched = False
                self._latched_patient_bed_bbox = None
                self._latched_bed_wall_ts = 0.0

            t_day = self._get_track_mode_vec(target, night_ir=False)
            if t_day is not None and (not self._is_night_ir or target.appearance_vec_day is not None):
                self._rollback_or_clear_mode_anchor_locked(
                    False,
                    polluted_vec=t_day,
                    force=was_patient,
                    now=ts,
                )
                self._remember_visitor_outfit(t_day, night_ir=False)
            elif was_patient and not self._is_night_ir:
                self._rollback_or_clear_mode_anchor_locked(False, polluted_vec=None, force=True, now=ts)

            t_night = self._get_track_mode_vec(target, night_ir=True)
            if t_night is not None and (self._is_night_ir or target.appearance_vec_night is not None):
                self._rollback_or_clear_mode_anchor_locked(
                    True,
                    polluted_vec=t_night,
                    force=was_patient,
                    now=ts,
                )
                self._remember_visitor_outfit(t_night, night_ir=True)
            elif was_patient and self._is_night_ir:
                self._rollback_or_clear_mode_anchor_locked(True, polluted_vec=None, force=True, now=ts)

            obs_target = self._last_obs_by_id.get(track_id)
            zones = self._last_zones
            if obs_target is not None and (
                zones is None or not self._is_sleeping_or_reclining_on_bed(obs_target, target, zones)
            ):
                self._recent_visitors[track_id] = _RecentVisitorMemory(
                    track_id=track_id,
                    last_seen=ts,
                    last_center=obs_target.body_center,
                    bbox_norm=obs_target.bbox_norm,
                    appearance_vec=target.appearance_vec.copy() if target.appearance_vec is not None else None,
                    appearance_vec_day=t_day.copy() if t_day is not None else None,
                    appearance_vec_night=t_night.copy() if t_night is not None else None,
                    reason="manual_visitor_override",
                )

            # Если в этот же момент на кровати находится другой валидный человек (лежащий или сидящий) — повышаем до Role.PATIENT!
            ref_ts = (
                ts
                if now is not None
                else max((tr_x.last_seen for tr_x in self._tracks.values()), default=ts)
            )
            has_other_patient = any(
                tid != track_id
                and tr.role == Role.PATIENT
                and (tid in self._last_obs_by_id or (ref_ts - tr.last_seen) <= 3.0)
                for tid, tr in self._tracks.items()
            )
            if not has_other_patient and zones is not None and zones.has_bed:
                hom_z = FloorHomography.from_zones(zones, bed_polygon=zones.bed_polygon)
                best_bed_tid: int | None = None
                best_bed_score = -1.0
                for tid, other_tr in self._tracks.items():
                    if tid == track_id or (
                        tid not in self._last_obs_by_id and (ref_ts - other_tr.last_seen) > 3.0
                    ):
                        continue
                    if other_tr.reason in (
                        "manual_visitor_override",
                        "face_visitor",
                        "visitor_outfit_match",
                        "door_entry",
                    ):
                        continue
                    other_obs = self._last_obs_by_id.get(tid)
                    reclining_other = (
                        self._is_sleeping_or_reclining_on_bed(other_obs, other_tr, zones)
                        if other_obs is not None
                        else other_tr.was_on_bed
                    )
                    seated_bed_other = (
                        other_obs is not None
                        and is_person_on_bed(other_obs, zones.bed_polygon, zones=zones)
                        and not self._is_standing_static_posture(other_obs, other_tr)
                        and not self._is_tall_floor_stander(
                            other_obs,
                            other_tr,
                            zones,
                            hom_calibrated=hom_z.calibrated,
                        )
                    )
                    if not (reclining_other or seated_bed_other):
                        continue
                    base_pts = 3.0 if reclining_other else 2.0
                    cand_score = base_pts + min(2.0, other_tr.bed_accum_sec)
                    if cand_score > best_bed_score:
                        best_bed_score = cand_score
                        best_bed_tid = tid

                if best_bed_tid is not None:
                    promoted = self._tracks[best_bed_tid]
                    promoted.role = Role.PATIENT
                    promoted.reason = "promoted_after_visitor_override"
                    self._recent_visitors.pop(best_bed_tid, None)
                    self._ever_locked_patient = True
                    self._last_patient_track_id = best_bed_tid
                    self._last_patient_pos = promoted.last_center
                    self._last_patient_seen_at = ts
                    self._patient_in_bed_latched = True
                    self._latched_patient_track_id = best_bed_tid
                    self._latched_bed_wall_ts = self._wall_clock()
                    prom_obs = self._last_obs_by_id.get(best_bed_tid)
                    if prom_obs is not None:
                        self._latched_patient_bed_bbox = prom_obs.bbox_norm

                    mode_night = self._is_night_ir
                    p_vec = self._get_track_mode_vec(promoted, mode_night)
                    v_vec = self._get_track_mode_vec(target, mode_night)
                    cur_anc = self._get_mode_anchor(mode_night)
                    if (
                        p_vec is not None
                        and (cur_anc is None or promoted.bed_accum_sec >= min(1.0, self._initial_bed_lock_sec))
                        and (v_vec is None or cosine_sim(p_vec, v_vec) < 0.82)
                    ):
                        self._set_mode_anchor(p_vec, night_ir=mode_night, now=ts)

            self._save_anchor_state()
            return True

    def _is_near_or_on_bed(self, obs: PoseObservation, zones: ZonesConfig) -> bool:
        if not zones.has_bed:
            return False
        if is_person_on_bed(obs, zones.bed_polygon, zones=zones):
            return True
        anchor = obs.mid_hip or obs.body_center
        return polygon_distance_norm(anchor, zones.bed_polygon) >= -0.05

    def _is_standing_static_posture(self, obs: PoseObservation, tr: TrackIdentity) -> bool:
        if obs.conf >= 0.65 and (obs.kp(13, 0.50) is not None or obs.kp(14, 0.50) is not None):
            return False
        angle = obs.torso_angle_from_vertical_deg()
        return (
            not tr.has_moved
            and angle is not None
            and angle < 28.0
            and obs.bbox_height >= 0.36
            and obs.aspect_ratio_wh < 0.68
        )

    def _is_sleeping_or_reclining_on_bed(
        self,
        obs: PoseObservation,
        tr: TrackIdentity,
        zones: ZonesConfig,
    ) -> bool:
        """
        Проверяет, что человек действительно лежит/полулежит на матрасе,
        а НЕ сидит на краю кровати со свешенными ногами или с высоким вертикальным корпусом.
        Сохраняет поддержку торцевого ракурса сна вдоль камеры (укороченная проекция внутри матраса).
        """
        if not zones.has_bed or not is_person_on_bed(obs, zones.bed_polygon, zones=zones):
            return False
        if self._is_standing_static_posture(obs, tr):
            return False
        if are_legs_hanging_outside_bed(obs, zones.bed_polygon, zones=zones):
            return False
        angle = obs.torso_angle_from_vertical_deg()
        if angle is not None and angle < 28.0 and (
            obs.bbox_height >= 0.34 or obs.torso_vertical_rise() >= 0.15
        ):
            return False
        return True

    def _is_tall_floor_stander(
        self,
        obs: PoseObservation,
        tr: TrackIdentity,
        zones: ZonesConfig,
        hom_calibrated: bool = False,
    ) -> bool:
        """
        Антропометрический фильтр полноростового взрослого гостя на полу (Волна 10):
        высокий человек (height_cm >= 174 см либо макушка <= 0.35 на калиброванном полу при стопах >= 0.88)
        без истории сна в кровати (bed_accum_sec < 1.0) не может являться спящей бабушкой.
        """
        if tr.reason == "manual_override":
            return False
        if tr.bed_accum_sec >= 1.0 or tr.was_on_bed:
            return False
        if self._is_sleeping_or_reclining_on_bed(obs, tr, zones):
            return False
        if obs.bbox_norm[3] < 0.88 or obs.bbox_height < 0.48:
            return False
        est_h = tr.height_cm
        if est_h is None and zones.has_bed:
            hom_loc = FloorHomography.from_zones(zones, bed_polygon=zones.bed_polygon)
            if hom_loc.is_valid:
                est_h = hom_loc.measure_pose(obs, bed_polygon=zones.bed_polygon).height_cm
                hom_calibrated = hom_calibrated or hom_loc.calibrated
        if hom_calibrated and est_h is not None and est_h >= 174.0:
            return True
        if (
            zones.has_bed
            and hom_calibrated
            and obs.bbox_norm[1] <= 0.35
            and not is_person_on_bed(obs, zones.bed_polygon, zones=zones)
        ):
            return True
        return False

    def _match_recent_visitor_continuity(
        self,
        obs: PoseObservation,
        tr: TrackIdentity,
        active_ids: set[int],
        zones: ZonesConfig,
        now: float,
        night_ir: bool = False,
    ) -> int | None:
        """
        Восстанавливает Role.VISITOR, если у сидящего/стоящего родственника произошёл
        перескок track_id на том же месте и с той же одеждой торса (строго в той же модальности день/ночь).
        ЖЁСТКИЙ ЗАПРЕТ: никогда не срабатывает на треке, реально лежащем в кровати!
        """
        if self._is_sleeping_or_reclining_on_bed(obs, tr, zones):
            return None

        best_old_tid: int | None = None
        best_score = -1.0
        tr_vec = self._get_track_mode_vec(tr, night_ir)

        eff_vis_ttl = (
            max(self._visitor_memory_ttl, 900.0)
            if self._door_visit_active
            else self._visitor_memory_ttl
        )
        for old_tid, rv in self._recent_visitors.items():
            if old_tid in active_ids:
                continue
            age = now - rv.last_seen
            if age < 0.0 or age > eff_vis_ttl:
                continue
            dist = math.hypot(
                obs.body_center[0] - rv.last_center[0],
                obs.body_center[1] - rv.last_center[1],
            )
            iou = bbox_iou(obs.bbox_norm, rv.bbox_norm)
            if dist > 0.15 and iou < 0.35:
                continue

            rv_vec = rv.appearance_vec_night if night_ir else rv.appearance_vec_day
            if rv_vec is None and rv.appearance_vec is not None:
                exp_dim = NIGHT_DESCRIPTOR_DIM if night_ir else DAY_DESCRIPTOR_DIM
                if rv.appearance_vec.shape == (exp_dim,):
                    rv_vec = rv.appearance_vec

            if tr_vec is not None and rv_vec is not None:
                sim_vis = cosine_sim(rv_vec, tr_vec)
                if sim_vis < 0.72:
                    continue
                score = sim_vis + (0.15 - min(0.15, dist))
            else:
                if age > 25.0 or (dist > 0.14 and iou < 0.35):
                    continue
                score = 0.70 + (0.14 - min(0.14, dist))

            if score > best_score:
                best_score = score
                best_old_tid = old_tid

        return best_old_tid

    def _resolve_bed_id_switch(
        self,
        observations: Sequence[PoseObservation],
        zones: ZonesConfig,
        night_ir: bool = False,
    ) -> int | None:
        mode_anchor = self._get_mode_anchor(night_ir)
        if len(observations) < 2 or not zones.has_bed or mode_anchor is None:
            for obs in observations:
                tr = self._tracks.get(obs.track_id)
                if tr and tr.role == Role.PATIENT:
                    return obs.track_id
            return None

        obs_by_id = {obs.track_id: obs for obs in observations}
        current_patient_obs: PoseObservation | None = None
        for obs in observations:
            tr = self._tracks.get(obs.track_id)
            if tr and tr.role == Role.PATIENT:
                current_patient_obs = obs
                break

        if current_patient_obs is None:
            return None

        pat_tr = self._tracks[current_patient_obs.track_id]
        pat_near_bed = self._is_near_or_on_bed(current_patient_obs, zones)
        if pat_near_bed or pat_tr.reason == "manual_override":
            return current_patient_obs.track_id

        pat_vec = self._get_track_mode_vec(pat_tr, night_ir)
        pat_sim = cosine_sim(mode_anchor, pat_vec)
        pat_in_door = (
            point_in_polygon(current_patient_obs.body_center, zones.door_polygon)
            if zones.has_door
            else False
        )

        best_bed_obs: PoseObservation | None = None
        best_bed_sim = 0.0
        for tid, obs in obs_by_id.items():
            if tid == current_patient_obs.track_id:
                continue
            tr = self._tracks.get(tid)
            if tr is None or tr.reason in ("face_visitor", "manual_visitor_override"):
                continue
            if not self._is_near_or_on_bed(obs, zones):
                continue
            if self._is_standing_static_posture(obs, tr):
                continue
            sim = cosine_sim(mode_anchor, self._get_track_mode_vec(tr, night_ir))
            if sim > best_bed_sim:
                best_bed_sim = sim
                best_bed_obs = obs

        if best_bed_obs is not None and best_bed_sim >= 0.78:
            if (pat_vec is not None and pat_sim < 0.72 and (best_bed_sim - pat_sim) >= 0.14) or (
                pat_in_door and pat_sim < 0.78
            ):
                bed_tr = self._tracks[best_bed_obs.track_id]
                pat_tr.role = Role.VISITOR
                pat_tr.reason = "id_switch_demoted_visitor"
                pat_tr.confirmed_second_track = True
                bed_tr.role = Role.PATIENT
                bed_tr.reason = "id_switch_recovered"
                logger.info(
                    "Перехвачен перескок ID: бабушка возвращена на трек #%d (sim=%.2f), уходящий #%d -> visitor (sim=%.2f)",
                    best_bed_obs.track_id,
                    best_bed_sim,
                    current_patient_obs.track_id,
                    pat_sim,
                )
                return best_bed_obs.track_id

        return current_patient_obs.track_id

    def _score_patient_candidate(
        self,
        obs: PoseObservation,
        tr: TrackIdentity,
        zones: ZonesConfig,
        now: float,
        night_ir: bool = False,
        has_other_reclining_sleeper: bool = False,
    ) -> float:
        """Комплексный скоринг кандидата на роль бабушки при совместном присутствии >= 2 людей."""
        if tr.reason == "manual_override":
            return 100.0
        if tr.reason == "manual_visitor_override":
            return -100.0

        score = 0.0
        if tr.reason == "face_patient":
            score += 40.0
        elif tr.reason == "face_visitor":
            score -= 40.0

        mode_anchor = self._get_mode_anchor(night_ir)
        tr_vec = self._get_track_mode_vec(tr, night_ir)
        vis_sim = self._max_visitor_outfit_sim(tr_vec, night_ir=night_ir)
        pat_sim = cosine_sim(mode_anchor, tr_vec) if mode_anchor is not None else 0.0
        if vis_sim >= 0.82 and vis_sim >= pat_sim + 0.04:
            score -= 12.0 * vis_sim
        if mode_anchor is not None and tr_vec is not None:
            if pat_sim >= 0.76:
                score += 5.5 * pat_sim
            elif pat_sim < 0.55:
                score -= 2.5

        reclining_bed = self._is_sleeping_or_reclining_on_bed(obs, tr, zones)
        on_bed = is_person_on_bed(obs, zones.bed_polygon, zones=zones) if zones.has_bed else False
        standing_floor_figure = zones.has_bed and is_standing_in_front_of_bed(
            obs, zones.bed_polygon, zones=zones
        )
        seated_legs_bed = (
            on_bed and are_legs_hanging_outside_bed(obs, zones.bed_polygon, zones=zones)
            if zones.has_bed
            else False
        )
        if self._is_tall_floor_stander(obs, tr, zones):
            score -= 12.0
        foreground_floor = not on_bed and obs.bbox_norm[3] >= 0.92 and obs.bbox_norm[1] >= 0.50
        angle = obs.torso_angle_from_vertical_deg()
        upright_narrow = (
            angle is not None
            and angle < 28.0
            and (obs.bbox_height >= 0.42 or (not on_bed and obs.aspect_ratio_wh < 0.55))
            and not seated_legs_bed
        )
        is_hard_visitor_track = (
            tr.reason not in ("id_switch_recovered", "manual_override", "face_patient")
            and (tr.spawned_in_door or tr.reason in _HARD_VISITOR_REASONS)
        )

        if reclining_bed:
            score += 7.5
            if obs.track_id >= 9000 and not tr.spawned_in_door:
                score += 3.0
        elif has_other_reclining_sleeper:
            # Когда в кровати уже лежит спящая бабушка, вертикально присевший с краю матраса или стоящий человек — гость!
            score -= 8.0
            if seated_legs_bed and tr.bed_accum_sec < 1.5:
                score -= 4.0
        elif (
            seated_legs_bed or (on_bed and not upright_narrow and not standing_floor_figure)
        ) and not is_hard_visitor_track:
            score += 2.6
        elif foreground_floor or (standing_floor_figure and tr.bed_accum_sec <= 0.0):
            score -= 3.8
        elif upright_narrow and not on_bed and tr.bed_accum_sec <= 0.0:
            score -= 2.8
        elif upright_narrow and on_bed and tr.bed_accum_sec <= 0.0:
            score -= 2.0

        if tr.was_on_bed and (reclining_bed or not is_hard_visitor_track):
            score += 2.8
        score += min(1.0, tr.bed_accum_sec / max(0.8, self._initial_bed_lock_sec)) * 2.5

        if tr.role == Role.PATIENT and (mode_anchor is None or tr_vec is None or pat_sim >= 0.65):
            score += 2.5

        if tr.reason != "id_switch_recovered":
            if is_hard_visitor_track and not reclining_bed:
                score -= 9.5
            elif tr.spawned_in_door or tr.reason in ("door_entry", "border_standing_entry", "foreground_floor_entry"):
                score -= 3.8
            elif tr.spawned_outside_bed and not tr.was_on_bed:
                score -= 2.2

        if self._last_patient_pos is not None and 0.0 <= (now - self._last_patient_seen_at) <= 30.0:
            dist = math.hypot(
                obs.body_center[0] - self._last_patient_pos[0],
                obs.body_center[1] - self._last_patient_pos[1],
            )
            score += 2.0 * max(0.0, 1.0 - dist / 0.28)

        return score

    def _arbitrate_multi_person_roles(
        self,
        observations: Sequence[PoseObservation],
        zones: ZonesConfig,
        now: float,
        night_ir: bool = False,
        allow_non_door_visitor: bool = True,
    ) -> int | None:
        """
        Совместный решатель ролей при нахождении >= 2 человек в кадре:
        однозначно отделяет лежащую/сидящую в кровати бабушку от вошедшего или стоящего у края гостя.
        """
        reclining_sleeper_ids = {
            obs.track_id
            for obs in observations
            if self._is_sleeping_or_reclining_on_bed(obs, self._tracks[obs.track_id], zones)
            and (
                not self._tracks[obs.track_id].spawned_in_door
                or self._tracks[obs.track_id].reason == "id_switch_recovered"
            )
            and self._tracks[obs.track_id].reason not in ("manual_visitor_override", "door_entry")
        }

        scored: list[tuple[float, PoseObservation, TrackIdentity]] = []
        for obs in observations:
            tr = self._tracks[obs.track_id]
            s = self._score_patient_candidate(
                obs,
                tr,
                zones,
                now,
                night_ir=night_ir,
                has_other_reclining_sleeper=bool(reclining_sleeper_ids - {obs.track_id}),
            )
            scored.append((s, obs, tr))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_obs, best_tr = scored[0]
        second_score = scored[1][0]

        best_on_bed = (
            zones.has_bed
            and is_person_on_bed(best_obs, zones.bed_polygon, zones=zones)
            and not self._is_standing_static_posture(best_obs, best_tr)
        )
        others_outside_bed = zones.has_bed and all(
            not is_person_on_bed(other_obs, zones.bed_polygon, zones=zones)
            for _, other_obs, _ in scored[1:]
        )
        sole_reclining_sleeper = (
            best_obs.track_id in reclining_sleeper_ids and len(reclining_sleeper_ids) == 1
        )
        can_elect_patient = best_tr.role == Role.PATIENT or (
            best_score >= 2.4
            and (best_score - second_score) >= 2.0
            and (
                self._ever_locked_patient
                or best_tr.bed_accum_sec >= min(1.2, self._initial_bed_lock_sec)
                or best_tr.reason in ("manual_override", "face_patient")
                or (best_on_bed and others_outside_bed and (best_score - second_score) >= 3.5)
                or (sole_reclining_sleeper and (best_score - second_score) >= 3.5)
            )
        )

        if can_elect_patient and best_score >= 1.5:
            if best_tr.role != Role.PATIENT:
                best_tr.role = Role.PATIENT
                best_tr.reason = "multi_person_bed_arbitration"
                self._ever_locked_patient = True
            for s_val, other_obs, other_tr in scored[1:]:
                if other_tr.role != Role.VISITOR:
                    other_on_bed_edge_with_sleeper = (
                        sole_reclining_sleeper
                        and zones.has_bed
                        and is_person_on_bed(other_obs, zones.bed_polygon, zones=zones)
                    )
                    if (
                        allow_non_door_visitor
                        or other_tr.spawned_in_door
                        or other_tr.reason in _HARD_VISITOR_REASONS
                        or other_on_bed_edge_with_sleeper
                    ):
                        other_tr.role = Role.VISITOR
                        other_tr.reason = "co_present_with_patient"
                    elif other_tr.role == Role.UNKNOWN:
                        other_tr.reason = "blocked_no_door_entry"
            return best_obs.track_id

        # Если ни один из нескольких людей не похож на пациента (например, 2 гостя при пустой кровати)
        if allow_non_door_visitor:
            for s_val, obs_item, tr_item in scored:
                if s_val <= -2.0 and tr_item.role == Role.UNKNOWN:
                    tr_item.role = Role.VISITOR
                    tr_item.reason = "multi_person_visitor_score"
        return None

    @staticmethod
    def _mattress_boxes_overlap(
        cand_box: tuple[float, float, float, float],
        prim_box: tuple[float, float, float, float],
    ) -> bool:
        if bbox_iou(cand_box, prim_box) >= 0.15:
            return True
        cx1, cy1, cx2, cy2 = cand_box
        px1, py1, px2, py2 = prim_box
        cw = max(1e-6, cx2 - cx1)
        ch = max(1e-6, cy2 - cy1)
        pw = max(1e-6, px2 - px1)
        ph = max(1e-6, py2 - py1)
        min_area = max(1e-6, min(cw * ch, pw * ph))
        inter_w = max(0.0, min(cx2, px2) - max(cx1, px1))
        inter_h = max(0.0, min(cy2, py2) - max(cy1, py1))
        if (inter_w * inter_h) / min_area >= 0.30:
            return True
        ccx, ccy = (cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5
        pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        if math.hypot(ccx - pcx, ccy - pcy) <= 0.20:
            return True
        if (inter_w / max(1e-6, min(cw, pw))) >= 0.50 and abs(ccy - pcy) <= 0.18:
            return True
        return False

    def _deduplicate_bed_reclining_observations(
        self,
        observations: Sequence[PoseObservation],
        zones: ZonesConfig,
        now: float,
    ) -> list[PoseObservation]:
        """
        Отсекает фантомные дублирующие боксы на одной спящей под одеялом бабушке
        (второй полнокадровый бокс с высоким IoU или горизонтальный фрагмент ног/одеяла на односпальном матрасе),
        чтобы они никогда не получали Role.VISITOR и не зажигали ложный caregiver_nearby / door_visit_active.
        """
        if not zones.has_bed or len(observations) < 2:
            return list(observations)

        _, bed_max_y = polygon_y_bounds(zones.bed_polygon)
        reclining_ids: set[int] = set()
        reclining_cands: list[PoseObservation] = []
        for o in observations:
            existing = self._tracks.get(o.track_id)
            if existing is not None and existing.reason in (
                "door_entry",
                "face_visitor",
                "manual_visitor_override",
            ):
                continue
            probe_tr = existing or TrackIdentity(
                track_id=o.track_id,
                first_seen=now,
                last_seen=now,
                has_moved=True,
            )
            if self._is_sleeping_or_reclining_on_bed(o, probe_tr, zones):
                reclining_ids.add(o.track_id)
                reclining_cands.append(o)

        if not reclining_cands:
            return list(observations)

        def _primary_priority(o: PoseObservation) -> tuple[int, int, float, float]:
            ex = self._tracks.get(o.track_id)
            return (
                1 if (ex is not None and ex.role == Role.PATIENT) else 0,
                1 if o.track_id == self._last_patient_track_id else 0,
                ex.bed_accum_sec if ex is not None else 0.0,
                o.conf,
            )

        primary = max(reclining_cands, key=_primary_priority)
        kept: list[PoseObservation] = []
        for o in observations:
            if o.track_id == primary.track_id:
                kept.append(o)
                continue
            cand_tr = self._tracks.get(o.track_id)
            if cand_tr is not None and cand_tr.reason in (
                "door_entry",
                "face_visitor",
                "manual_visitor_override",
            ):
                kept.append(o)
                continue
            if (
                are_legs_hanging_outside_bed(o, zones.bed_polygon, zones=zones)
                or o.bbox_norm[3] > (bed_max_y + 0.04)
                or o.bbox_height > 0.34
            ):
                kept.append(o)
                continue

            is_reclining_dup = o.track_id in reclining_ids
            is_blanket_fold_dup = (
                not is_reclining_dup
                and o.bbox_norm[3] <= (bed_max_y + 0.03)
                and o.bbox_height <= 0.28
                and o.aspect_ratio_wh >= 1.15
                and bbox_polygon_overlap_ratio(o.bbox_norm, zones.bed_polygon) >= 0.65
            )
            if (is_reclining_dup or is_blanket_fold_dup) and (
                self._mattress_boxes_overlap(o.bbox_norm, primary.bbox_norm)
                or math.hypot(
                    o.body_center[0] - primary.body_center[0],
                    o.body_center[1] - primary.body_center[1],
                )
                <= 0.20
            ):
                self._recent_visitors.pop(o.track_id, None)
                if cand_tr is not None and cand_tr.role != Role.PATIENT and cand_tr.reason not in (
                    "manual_override",
                    "manual_visitor_override",
                    "face_visitor",
                    "door_entry",
                ):
                    self._tracks.pop(o.track_id, None)
                continue

            kept.append(o)
        return kept

    def _select_primary_patient_track_id(
        self,
        observations: Sequence[PoseObservation],
        rescued_sit_ids: set[int],
        zones: ZonesConfig,
        now: float,
    ) -> int | None:
        for o in observations:
            ex = self._tracks.get(o.track_id)
            if ex is not None and ex.role == Role.PATIENT:
                return o.track_id
        for o in observations:
            if o.track_id in rescued_sit_ids:
                return o.track_id
        if zones.has_bed:
            bed_cands: list[tuple[tuple[int, int, float, float], int]] = []
            for o in observations:
                if not is_person_on_bed(o, zones.bed_polygon, zones=zones):
                    continue
                if self._is_touching_door(o, zones):
                    continue
                ex = self._tracks.get(o.track_id)
                if ex is not None and ex.role == Role.VISITOR and len(observations) > 1:
                    continue
                probe = ex or TrackIdentity(
                    track_id=o.track_id,
                    first_seen=now,
                    last_seen=now,
                    has_moved=True,
                )
                recl = 1 if self._is_sleeping_or_reclining_on_bed(o, probe, zones) else 0
                is_last = 1 if o.track_id == self._last_patient_track_id else 0
                accum = ex.bed_accum_sec if ex is not None else 0.0
                bed_cands.append(((recl, is_last, accum, o.conf), o.track_id))
            if bed_cands:
                bed_cands.sort(key=lambda item: item[0], reverse=True)
                return bed_cands[0][1]
        elif len(observations) == 1:
            return observations[0].track_id
        return None

    def _enforce_second_track_motion_gate(
        self,
        observations: Sequence[PoseObservation],
        rescued_sit_ids: set[int],
        zones: ZonesConfig,
        now: float,
    ) -> list[PoseObservation]:
        """
        Многокадровый гейт второго трека (Задача 7):
        Любой второй трек вне подтверждённого входа через дверь обязан накопить
        confirmed_frames >= 4 И реальное пространственное смещение центра >= 0.045
        строго ВНЕ матраса и ВНЕ ignore_polygons, прежде чем попадёт в tracks / caregiver_nearby.
        Мимолётные тени, складки одеяла и спинка кровати за 1–3 кадра бесследно отсекаются.
        """
        if not observations:
            return []

        primary_pat_tid = self._select_primary_patient_track_id(
            observations=observations,
            rescued_sit_ids=rescued_sit_ids,
            zones=zones,
            now=now,
        )
        passed: list[PoseObservation] = []
        for o in observations:
            if o.track_id == primary_pat_tid:
                passed.append(o)
                continue

            ex = self._tracks.get(o.track_id)
            if ex is not None and (
                ex.confirmed_second_track
                or ex.role == Role.PATIENT
                or ex.reason
                in (
                    "manual_override",
                    "manual_visitor_override",
                    "face_patient",
                    "face_visitor",
                    "door_entry",
                )
            ):
                passed.append(o)
                continue

            confirmed_door_entry = zones.has_door and (
                self._is_touching_door(o, zones)
                or self._liveness_gate.is_spawned_in_door(o.track_id)
                or (
                    o.spawn_center_hint is not None
                    and point_in_polygon(o.spawn_center_hint, zones.door_polygon)
                )
                or (ex is not None and ex.spawned_in_door)
            ) and (self._liveness_gate.is_track_moved(o.track_id) or o.liveness_verified)
            if confirmed_door_entry:
                passed.append(o)
                continue

            # Не задерживаем перехват падения/сползания самой бабушки со сменой track_id у кровати
            if (
                primary_pat_tid is None
                and self._ever_locked_patient
                and self._last_patient_pos is not None
                and (
                    0.0 <= (now - self._last_patient_seen_at) <= 6.0
                    or (self._patient_in_bed_latched and not self._door_visit_active)
                )
                and math.hypot(
                    o.body_center[0] - self._last_patient_pos[0],
                    o.body_center[1] - self._last_patient_pos[1],
                )
                <= 0.24
            ):
                ang = o.torso_angle_from_vertical_deg()
                if ang is None or ang >= 38.0 or o.aspect_ratio_wh >= 0.85:
                    passed.append(o)
                    continue

            frames = max(int(o.confirmed_frames), self._liveness_gate.get_frames_seen(o.track_id))
            out_disp = max(
                float(o.outside_bed_disp),
                self._liveness_gate.get_outside_bed_displacement(o.track_id),
            )
            seated_visitor_rebind = (
                self._door_visit_active
                and (not zones.has_bed or not is_person_on_bed(o, zones.bed_polygon, zones=zones))
                and any(
                    rv.track_id not in {x.track_id for x in observations}
                    and 0.0 <= (now - rv.last_seen) <= 120.0
                    and (
                        math.hypot(
                            o.body_center[0] - rv.last_center[0],
                            o.body_center[1] - rv.last_center[1],
                        )
                        <= 0.15
                        or bbox_iou(o.bbox_norm, rv.bbox_norm) >= 0.35
                    )
                    for rv in self._recent_visitors.values()
                )
            )
            fg_seated_caregiver = is_confident_foreground_seated_person(
                o,
                zones.bed_polygon if zones.has_bed else None,
                zones=zones,
            ) and (
                (primary_pat_tid is not None and primary_pat_tid != o.track_id)
                or self._patient_in_bed_latched
            )
            if (frames >= 4 and out_disp >= 0.045) or seated_visitor_rebind or fg_seated_caregiver:
                passed.append(o)
            else:
                self._recent_visitors.pop(o.track_id, None)
                if ex is not None and ex.role != Role.PATIENT:
                    self._tracks.pop(o.track_id, None)

        return passed

    def update(
        self,
        observations: Sequence[PoseObservation],
        zones: ZonesConfig,
        now: float,
        frame_bgr: np.ndarray | None = None,
    ) -> tuple[dict[int, TrackIdentity], bool]:
        """
        Обновляет роли всех подтверждённо живых треков в кадре.
        Неподвижные предметы вне кровати, 1–3-кадровые фантомы и скелеты в ignore_polygons отфильтровываются.
        Возвращает (словарь {track_id: TrackIdentity}, caregiver_nearby: bool).
        """
        with self._lock:
            self._check_anchor_expiry()
            if self._first_update_ts is None or now < self._first_update_ts:
                self._first_update_ts = now
            night_ir = is_night_ir_frame(frame_bgr) if frame_bgr is not None else self._is_night_ir
            self._is_night_ir = night_ir
            homography = FloorHomography.from_zones(
                zones,
                bed_polygon=zones.bed_polygon if zones.has_bed else None,
            )

            cur_gray_small = OpticalKeypointStabilizer._to_gray_small(frame_bgr)
            same_portal_frame = frame_bgr is not None and id(frame_bgr) == self._portal_last_frame_id
            cur_textured = int(np.ptp(cur_gray_small)) >= 8 if cur_gray_small is not None else False
            has_portal_opt_pair = (
                not same_portal_frame
                and cur_gray_small is not None
                and self._portal_prev_gray is not None
                and cur_gray_small.shape == self._portal_prev_gray.shape
                and cur_textured
                and bool(self._portal_prev_textured)
                and 0.0 <= (now - self._portal_prev_gray_ts) <= 3.5
            )
            if has_portal_opt_pair and cur_gray_small is not None and self._portal_prev_gray is not None:
                c_i16 = cur_gray_small.astype(np.int16)
                p_i16 = self._portal_prev_gray.astype(np.int16)
                if (
                    abs(float(np.mean(c_i16)) - float(np.mean(p_i16))) > 12.0
                    or float(np.mean(np.abs(c_i16 - p_i16) > 24)) > 0.45
                ):
                    has_portal_opt_pair = False

            # Слой 0: Санация смятых точек сна + анатомический фильтр + ignore_polygons + фильтр неподвижных вещей
            if zones.has_bed:
                observations = [
                    sanitize_collapsed_bed_keypoints(o, zones.bed_polygon) for o in observations
                ]
            valid_obs = filter_valid_poses(
                observations,
                min_kp_conf=0.30,
                min_valid_kps=4,
                min_box_conf=0.48,
                zones=zones,
            )
            observations = self._liveness_gate.update_and_filter(
                valid_obs,
                zones=zones,
                now=now,
                frame_bgr=frame_bgr,
            )
            # Если бабушка только что села вертикально на матрасе и ByteTrack выдал новый track_id,
            # StaticObjectGate на 1-м кадре мог придержать вертикальный силуэт (angle < 28°) как «неподвижный».
            # Но в точке, где <= 12 с назад на кровати спала бабушка, куртка возникнуть не может — пропускаем без задержки!
            rescued_sit_ids: set[int] = set()
            rescue_gap_limit = (
                86400.0
                if (self._patient_in_bed_latched and not self._door_visit_active and not self._recent_visitors)
                else (120.0 if not self._recent_visitors else 12.0)
            )
            if (
                self._ever_locked_patient
                and self._last_patient_pos is not None
                and (
                    0.0 <= (now - self._last_patient_seen_at) <= rescue_gap_limit
                    or (self._patient_in_bed_latched and self._last_patient_seen_at < 0.0)
                )
                and zones.has_bed
            ):
                live_ids = {o.track_id for o in observations}
                last_pat_obs_live = next(
                    (o for o in observations if o.track_id == self._last_patient_track_id),
                    None,
                )
                last_pat_on_bed_live = (
                    last_pat_obs_live is not None
                    and is_person_on_bed(last_pat_obs_live, zones.bed_polygon, zones=zones)
                )
                if not last_pat_on_bed_live:
                    obs_mut = list(observations)
                    latched_bed_center: tuple[float, float] | None = None
                    if self._latched_patient_bed_bbox is not None:
                        lbx1, lby1, lbx2, lby2 = self._latched_patient_bed_bbox
                        latched_bed_center = ((lbx1 + lbx2) * 0.5, (lby1 + lby2) * 0.5)
                    for v_obs in valid_obs:
                        if v_obs.track_id in live_ids:
                            continue
                        ex_v_tr = self._tracks.get(v_obs.track_id)
                        if ex_v_tr is not None and ex_v_tr.role == Role.PATIENT and ex_v_tr.has_moved:
                            obs_mut.append(v_obs)
                            live_ids.add(v_obs.track_id)
                            rescued_sit_ids.add(v_obs.track_id)
                            continue
                        v_on_bed = is_person_on_bed(v_obs, zones.bed_polygon, zones=zones)
                        if last_pat_obs_live is not None and not v_on_bed:
                            continue
                        v_near_bed_wakeup = (
                            not v_on_bed
                            and self._patient_in_bed_latched
                            and not self._door_visit_active
                            and self._is_near_or_on_bed(v_obs, zones)
                            and (
                                frame_bgr is None
                                or v_obs.track_id in self._tracks
                                or (
                                    has_portal_opt_pair
                                    and OpticalKeypointStabilizer.compute_roi_optical_motion(
                                        cur_gray_small,
                                        self._portal_prev_gray,
                                        v_obs.bbox_norm,
                                        v_obs.bbox_norm,
                                    )
                                    >= 0.015
                                )
                            )
                        )
                        if not (v_on_bed or v_near_bed_wakeup):
                            continue
                        if zones.has_door and point_in_polygon(v_obs.body_center, zones.door_polygon):
                            continue
                        near_last_pos = (
                            math.hypot(
                                v_obs.body_center[0] - self._last_patient_pos[0],
                                v_obs.body_center[1] - self._last_patient_pos[1],
                            )
                            <= 0.20
                        )
                        near_latched_bed = (
                            v_on_bed
                            and latched_bed_center is not None
                            and math.hypot(
                                v_obs.body_center[0] - latched_bed_center[0],
                                v_obs.body_center[1] - latched_bed_center[1],
                            )
                            <= 0.22
                        )
                        if near_last_pos or near_latched_bed:
                            obs_mut.append(v_obs)
                            live_ids.add(v_obs.track_id)
                            rescued_sit_ids.add(v_obs.track_id)
                    observations = obs_mut

            observations = self._deduplicate_bed_reclining_observations(
                observations=observations,
                zones=zones,
                now=now,
            )
            observations = self._enforce_second_track_motion_gate(
                observations=observations,
                rescued_sit_ids=rescued_sit_ids,
                zones=zones,
                now=now,
            )
            self._last_obs_by_id = {obs.track_id: obs for obs in observations}
            self._last_zones = zones

            # 1. Очистка устаревших треков и старой памяти гостей
            stale = [
                tid for tid, tr in self._tracks.items() if (now - tr.last_seen) > self._track_ttl
            ]
            for tid in stale:
                del self._tracks[tid]
            eff_vis_ttl = (
                max(self._visitor_memory_ttl, 900.0)
                if self._door_visit_active
                else self._visitor_memory_ttl
            )
            stale_vis = [
                tid
                for tid, rv in self._recent_visitors.items()
                if (now - rv.last_seen) > eff_vis_ttl
            ]
            for tid in stale_vis:
                del self._recent_visitors[tid]

            active_ids = {obs.track_id for obs in observations}
            confirmed_patient_id: int | None = None
            for obs in observations:
                existing = self._tracks.get(obs.track_id)
                if existing and existing.role == Role.PATIENT:
                    confirmed_patient_id = obs.track_id
                    break

            max_door_x = (
                max((float(p[0]) for p in zones.door_polygon), default=0.0)
                if zones.has_door
                else 0.0
            )
            any_obs_on_bed = zones.has_bed and any(
                is_person_on_bed(o, zones.bed_polygon, zones=zones) for o in observations
            )

            # 2. Регистрация живых треков, 3D-замер пола, биометрия лица и расчёт дескрипторов одежды
            for obs in observations:
                tr = self._tracks.get(obs.track_id)
                on_bed = is_person_on_bed(obs, zones.bed_polygon, zones=zones) if zones.has_bed else False
                in_door = self._is_touching_door(obs, zones)
                moved_flag = (
                    self._liveness_gate.is_track_moved(obs.track_id)
                    or obs.liveness_verified
                    or (obs.track_id in rescued_sit_ids)
                )
                frames_seen = max(
                    int(obs.confirmed_frames),
                    self._liveness_gate.get_frames_seen(obs.track_id),
                )
                out_disp = max(
                    float(obs.outside_bed_disp),
                    self._liveness_gate.get_outside_bed_displacement(obs.track_id),
                )

                if tr is None:
                    origin_center = (
                        obs.spawn_center_hint
                        or self._liveness_gate.get_spawn_center(obs.track_id)
                        or obs.body_center
                    )
                    origin_in_door = (
                        (
                            in_door
                            or self._liveness_gate.is_spawned_in_door(obs.track_id)
                            or point_in_polygon(origin_center, zones.door_polygon)
                        )
                        if zones.has_door
                        else False
                    )
                    origin_outside_bed = (
                        (not on_bed or not point_in_polygon(origin_center, zones.bed_polygon))
                        if zones.has_bed
                        else True
                    )
                    angle_init = obs.torso_angle_from_vertical_deg()
                    upright_entry = (
                        angle_init is not None
                        and angle_init < 28.0
                        and obs.aspect_ratio_wh < 0.85
                    )
                    in_left_door_corridor = (
                        zones.has_door
                        and not origin_in_door
                        and origin_outside_bed
                        and upright_entry
                        and origin_center[0] <= max(0.28, min(0.30, max_door_x + 0.02))
                    )
                    near_border = (
                        obs.is_near_border(0.08)
                        or origin_center[0] <= 0.12
                        or origin_center[0] >= 0.88
                        or origin_center[1] <= 0.12
                        or origin_center[1] >= 0.88
                    )
                    far_from_last_patient = (
                        self._last_patient_pos is None
                        or math.hypot(
                            origin_center[0] - self._last_patient_pos[0],
                            origin_center[1] - self._last_patient_pos[1],
                        )
                        >= 0.22
                    )

                    near_door_portal_entry = (
                        zones.has_door
                        and moved_flag
                        and (
                            origin_in_door
                            or polygon_distance_norm(origin_center, zones.door_polygon) >= -0.05
                            or polygon_distance_norm(obs.body_center, zones.door_polygon) >= -0.05
                        )
                    )
                    strict_door_track = (
                        zones.has_door
                        and not self._door_visit_active
                        and self._first_update_ts is not None
                        and (now - self._first_update_ts) > 15.0
                    )

                    fg_seated_companion = (
                        origin_outside_bed
                        and is_confident_foreground_seated_person(
                            obs,
                            zones.bed_polygon if zones.has_bed else None,
                            zones=zones,
                        )
                        and (
                            (confirmed_patient_id is not None and confirmed_patient_id != obs.track_id)
                            or any(
                                o2.track_id != obs.track_id
                                and zones.has_bed
                                and is_person_on_bed(o2, zones.bed_polygon, zones=zones)
                                for o2 in observations
                            )
                            or self._patient_in_bed_latched
                        )
                        and (
                            not zones.has_door
                            or is_pose_touching_door(obs, zones.door_polygon, min_overlap=0.05)
                            or polygon_distance_norm(obs.body_center, zones.door_polygon) >= -0.16
                        )
                    )

                    tr = TrackIdentity(
                        track_id=obs.track_id,
                        first_seen=now,
                        last_seen=now,
                        spawned_outside_bed=origin_outside_bed,
                        spawned_in_door=bool(origin_in_door or near_door_portal_entry),
                        has_moved=moved_flag,
                        was_on_bed=False,
                        last_center=obs.body_center,
                        confirmed_second_track=bool(
                            origin_in_door
                            or near_door_portal_entry
                            or fg_seated_companion
                            or (frames_seen >= 4 and out_disp >= 0.045)
                        ),
                    )
                    if (origin_in_door or near_door_portal_entry) and origin_outside_bed:
                        tr.role = Role.VISITOR
                        tr.reason = "door_entry"
                        tr.spawned_in_door = True
                        tr.confirmed_second_track = True
                        self._activate_door_visit(now)
                    elif fg_seated_companion:
                        tr.role = Role.VISITOR
                        tr.reason = "foreground_seated_companion"
                        tr.confirmed_second_track = True
                        self._activate_door_visit(now)
                    elif not strict_door_track:
                        if in_left_door_corridor and origin_outside_bed:
                            tr.role = Role.VISITOR
                            tr.reason = "border_standing_entry"
                        elif confirmed_patient_id is not None and confirmed_patient_id != obs.track_id:
                            tr.role = Role.VISITOR
                            tr.reason = "second_person_while_patient_active"
                        elif (
                            zones.has_bed
                            and origin_outside_bed
                            and obs.bbox_norm[3] >= 0.88
                            and obs.bbox_norm[1] >= 0.48
                            and (
                                far_from_last_patient
                                or any(
                                    o2.track_id != obs.track_id
                                    and is_person_on_bed(o2, zones.bed_polygon, zones=zones)
                                    for o2 in observations
                                )
                            )
                        ):
                            tr.role = Role.VISITOR
                            tr.reason = "foreground_floor_entry"
                        elif (
                            zones.has_bed
                            and origin_outside_bed
                            and obs.aspect_ratio_wh < 0.85
                            and near_border
                            and far_from_last_patient
                        ):
                            tr.role = Role.VISITOR
                            tr.reason = "border_standing_entry"
                    elif origin_outside_bed:
                        tr.reason = "blocked_no_door_entry"
                    self._tracks[obs.track_id] = tr

                dt = max(0.0, min(1.0, now - tr.last_seen))
                tr.last_seen = now
                tr.last_center = obs.body_center
                if moved_flag:
                    tr.has_moved = True
                if frames_seen >= 4 and out_disp >= 0.045:
                    tr.confirmed_second_track = True

                if homography.is_valid:
                    m3d = homography.measure_pose(
                        obs,
                        bed_polygon=zones.bed_polygon if zones.has_bed else None,
                    )
                    tr.floor_xy_cm = m3d.floor_xy_cm
                    tr.height_cm = m3d.height_cm

                reclining_on_bed = self._is_sleeping_or_reclining_on_bed(obs, tr, zones)
                if reclining_on_bed:
                    tr.was_on_bed = True
                if (reclining_on_bed or obs.track_id >= 9000) and tr.reason == "face_visitor":
                    tr.role = Role.UNKNOWN
                    tr.reason = "cleared_reclining_face_visitor"
                    tr.face_score = 0.0
                tall_floor_stander = self._is_tall_floor_stander(
                    obs,
                    tr,
                    zones,
                    hom_calibrated=homography.calibrated,
                )
                outside_companion_guard = (
                    zones.has_bed
                    and not on_bed
                    and tr.bed_accum_sec < 1.0
                    and not tr.was_on_bed
                    and tr.reason != "manual_override"
                    and (
                        self._door_visit_active
                        or (
                            zones.has_door
                            and self._patient_in_bed_latched
                            and not any_obs_on_bed
                            and (obs.track_id != self._latched_patient_track_id or homography.calibrated)
                        )
                    )
                )
                if (
                    outside_companion_guard
                    and tr.role != Role.PATIENT
                    and (self._door_visit_active or tr.has_moved)
                ):
                    if tr.role != Role.VISITOR:
                        tr.role = Role.VISITOR
                        tr.reason = "door_visit_companion"
                    tr.confirmed_second_track = True
                    tr.bed_accum_sec = 0.0
                    self._activate_door_visit(now)

                # Вход через полигон двери (или её 5%-порог с реальным движением) на первых секундах жизни трека -> железный Role.VISITOR ("door_entry")
                in_door_portal_now = zones.has_door and (
                    in_door
                    or (moved_flag and polygon_distance_norm(obs.body_center, zones.door_polygon) >= -0.05)
                )
                if (
                    in_door_portal_now
                    and not on_bed
                    and tr.role != Role.PATIENT
                    and not tr.was_on_bed
                    and (now - tr.first_seen) <= 6.0
                ):
                    tr.role = Role.VISITOR
                    tr.reason = "door_entry"
                    tr.spawned_in_door = True
                    tr.confirmed_second_track = True
                    tr.bed_accum_sec = 0.0
                    self._activate_door_visit(now)

                # Актуализация воротниково-торсового профиля строго в активной модальности:
                # Днём — 48D RGB/HSV, Ночью ИК — 64D LBP + Sobel + CLAHE (без кросс-смешивания!)
                if frame_bgr is not None:
                    if night_ir:
                        if (
                            tr.appearance_vec_night is None
                            or not tr.appearance_is_night
                            or (now - tr.last_appearance_calc) >= 1.5
                        ):
                            vec_n = compute_night_ir_appearance(frame_bgr, obs)
                            if vec_n is not None:
                                tr.appearance_vec_night = vec_n
                                tr.appearance_vec = vec_n
                                tr.appearance_is_night = True
                                tr.last_appearance_calc = now
                    else:
                        if (
                            tr.appearance_vec_day is None
                            or tr.appearance_is_night
                            or (now - tr.last_appearance_calc) >= 1.5
                        ):
                            vec_d = compute_torso_appearance(frame_bgr, obs)
                            if vec_d is not None:
                                tr.appearance_vec_day = vec_d
                                tr.appearance_vec = vec_d
                                tr.appearance_is_night = False
                                tr.last_appearance_calc = now

                tr_mode_vec = self._get_track_mode_vec(tr, night_ir)
                mode_anchor = self._get_mode_anchor(night_ir)

                strict_outfit_guard = (
                    zones.has_door
                    and not self._door_visit_active
                    and self._first_update_ts is not None
                    and (now - self._first_update_ts) > 15.0
                )

                # Проверка по сохранённой 8-часовой галерее одежды родственников («Свой»):
                # НИКОГДА не применяется к реально лежащему в кровати человеку (reclining_on_bed),
                # а door_visit_active взводится строго только на подтверждённо двигающемся человеке ВНЕ кровати!
                if (
                    not reclining_on_bed
                    and tr_mode_vec is not None
                    and tr.reason
                    not in (
                        "manual_override",
                        "face_patient",
                        "door_entry",
                    )
                ):
                    vis_outfit_sim = self._max_visitor_outfit_sim(tr_mode_vec, night_ir=night_ir)
                    pat_anchor_sim = (
                        cosine_sim(mode_anchor, tr_mode_vec)
                        if mode_anchor is not None
                        else 0.0
                    )
                    min_vis_outfit = 0.86 if strict_outfit_guard else 0.84
                    if (
                        vis_outfit_sim >= min_vis_outfit
                        and vis_outfit_sim >= pat_anchor_sim + 0.06
                        and tr.role != Role.PATIENT
                        and (not strict_outfit_guard or (not on_bed and tr.has_moved))
                    ):
                        tr.role = Role.VISITOR
                        tr.reason = "visitor_outfit_match"
                        tr.confirmed_second_track = True
                        tr.bed_accum_sec = 0.0
                        if not on_bed and tr.has_moved:
                            self._activate_door_visit(now)

                # Перехват перескока track_id у сидящего/стоящего гостя (НИКОГДА не трогает лежащего в кровати!)
                if (
                    not strict_outfit_guard
                    and not reclining_on_bed
                    and tr.role == Role.UNKNOWN
                    and (now - tr.first_seen) <= 2.0
                ):
                    matched_vis_tid = self._match_recent_visitor_continuity(
                        obs=obs,
                        tr=tr,
                        active_ids=active_ids,
                        zones=zones,
                        now=now,
                        night_ir=night_ir,
                    )
                    if matched_vis_tid is not None:
                        tr.role = Role.VISITOR
                        tr.reason = "visitor_id_switch_continuity"
                        tr.confirmed_second_track = True
                        tr.bed_accum_sec = 0.0
                        self._recent_visitors.pop(matched_vis_tid, None)

                # Слой 2: Биометрия лица (не чаще 1 раза в секунду на трек)
                # Перепроверка уже назначенных ролей (раз в 2 с): гость вне кровати может оказаться
                # бабушкой, а «бабушка» вне кровати — родственником. Смена роли только по двум
                # подряд уверенным совпадениям лица, чтобы единичный шум не переворачивал роли.
                face_recheck_role = (
                    tr.role == Role.VISITOR and not on_bed
                ) or (
                    tr.role == Role.PATIENT and not reclining_on_bed and tr.reason != "face_patient"
                )
                role_before_face = tr.role
                if (
                    frame_bgr is not None
                    and self._face_registry is not None
                    and tr.reason not in ("manual_override", "manual_visitor_override")
                    and (
                        (
                            (now - tr.last_face_check) >= 1.0
                            and (tr.role == Role.UNKNOWN or (tr.role == Role.VISITOR and on_bed))
                        )
                        or (face_recheck_role and (now - tr.last_face_check) >= 2.0)
                    )
                ):
                    tr.last_face_check = now
                    other_obs_for_face = [o for o in observations if o.track_id != obs.track_id]
                    latched_bed_box = (
                        self._latched_patient_bed_bbox if self._patient_in_bed_latched else None
                    )
                    try:
                        matched_role, score = self._face_registry.match_person_crop(
                            frame_bgr,
                            obs,
                            other_observations=other_obs_for_face,
                            zones=zones,
                            latched_patient_bed_bbox=latched_bed_box,
                        )
                    except TypeError:
                        matched_role, score = self._face_registry.match_person_crop(frame_bgr, obs)

                    if face_recheck_role:
                        strong_face = score >= max(self._face_recheck_min_score, 0.46)
                        if role_before_face == Role.PATIENT:
                            if matched_role == Role.VISITOR and strong_face:
                                tr.face_visitor_hits += 1
                            elif matched_role == Role.PATIENT:
                                tr.face_visitor_hits = 0
                            if tr.face_visitor_hits < 2:
                                matched_role = Role.UNKNOWN
                            else:
                                tr.role = Role.UNKNOWN
                                tr.bed_accum_sec = 0.0
                                if confirmed_patient_id == obs.track_id:
                                    confirmed_patient_id = None
                                logger.info(
                                    "Лицо «Своего» дважды подтверждено на треке #%d с ролью бабушки (%.2f) — роль снята",
                                    obs.track_id,
                                    score,
                                )
                        else:
                            if matched_role == Role.PATIENT and strong_face:
                                tr.face_patient_hits += 1
                            elif matched_role == Role.VISITOR:
                                tr.face_patient_hits = 0
                            if matched_role == Role.PATIENT and tr.face_patient_hits < 2:
                                matched_role = Role.UNKNOWN
                    face_double_confirmed_patient = (
                        face_recheck_role
                        and role_before_face == Role.VISITOR
                        and tr.face_patient_hits >= 2
                    )
                    if (
                        matched_role == Role.VISITOR
                        and not reclining_on_bed
                        and obs.track_id < 9000
                        and (not strict_outfit_guard or score >= 0.48)
                    ):
                        tr.role = Role.VISITOR
                        tr.reason = "face_visitor"
                        tr.confirmed_second_track = True
                        tr.face_score = score
                        self._activate_door_visit(now)
                    elif (
                        matched_role == Role.PATIENT
                        and not tall_floor_stander
                        and (not outside_companion_guard or face_double_confirmed_patient)
                    ):
                        tr.face_visitor_hits = 0
                        if confirmed_patient_id is not None and confirmed_patient_id != obs.track_id:
                            old_pat = self._tracks.get(confirmed_patient_id)
                            if old_pat is not None:
                                old_pat.role = Role.VISITOR
                                old_pat.reason = "id_switch_demoted_visitor"
                                old_pat.confirmed_second_track = True
                        tr.role = Role.PATIENT
                        tr.reason = "face_patient"
                        tr.face_score = score
                        confirmed_patient_id = obs.track_id
                        self._ever_locked_patient = True

                # Накопление времени РЕАЛЬНОГО лежания в кровати + поддержка самоисцеления sole_bed_sleeper_recovery
                explicit_visitor_lock = (
                    (tr.reason == "face_visitor" and tr.face_score >= 0.48)
                    or (tr.reason == "manual_visitor_override" and (now - tr.manual_assigned_at) <= 28800.0)
                    or tr.reason == "visitor_outfit_match"
                )
                sole_sleeper_eligible = (
                    len(observations) == 1
                    and reclining_on_bed
                    and not explicit_visitor_lock
                )
                if reclining_on_bed and (tr.role != Role.VISITOR or sole_sleeper_eligible):
                    tr.bed_accum_sec += dt
                elif not reclining_on_bed and tr.role == Role.UNKNOWN:
                    tr.bed_accum_sec = max(0.0, tr.bed_accum_sec - dt * 0.5)

            # 3. Защита от прямого перескока ID между бабушкой в кровати и уходящим родственником
            confirmed_patient_id = self._resolve_bed_id_switch(
                observations,
                zones,
                night_ir=night_ir,
            )

            # 4. Восстановление роли бабушки после перекрытия, скидывания одеяла или ошибочного гостя
            mode_anchor = self._get_mode_anchor(night_ir)
            mode_restored_disk = self._is_mode_restored_from_disk(night_ir)
            for obs in observations:
                tr = self._tracks[obs.track_id]
                on_bed = is_person_on_bed(obs, zones.bed_polygon, zones=zones) if zones.has_bed else False
                reclining_on_bed = self._is_sleeping_or_reclining_on_bed(obs, tr, zones)
                near_bed = self._is_near_or_on_bed(obs, zones)
                is_static_coat = self._is_standing_static_posture(obs, tr)
                tall_floor_stander = self._is_tall_floor_stander(
                    obs,
                    tr,
                    zones,
                    hom_calibrated=homography.calibrated,
                )
                explicit_visitor_lock = (
                    (tr.reason == "face_visitor" and tr.face_score >= 0.48)
                    or (tr.reason == "manual_visitor_override" and (now - tr.manual_assigned_at) <= 28800.0)
                    or tr.reason == "visitor_outfit_match"
                )

                # 4.0. Самоисцеление sole_bed_sleeper_recovery: 1 человек в комнате реально спит на матрасе (<= 3.0 с)
                if (
                    len(observations) == 1
                    and reclining_on_bed
                    and not explicit_visitor_lock
                    and tr.role == Role.VISITOR
                    and confirmed_patient_id is None
                ):
                    recovery_needed = min(3.0, max(0.8, self._initial_bed_lock_sec))
                    if tr.bed_accum_sec >= recovery_needed:
                        tr.role = Role.PATIENT
                        tr.reason = "sole_bed_sleeper_recovery"
                        confirmed_patient_id = obs.track_id
                        self._ever_locked_patient = True
                        self._recent_visitors.pop(obs.track_id, None)

                can_rehabilitate = tr.role == Role.UNKNOWN or (
                    tr.role == Role.VISITOR
                    and tr.reason not in _HARD_VISITOR_REASONS
                    and near_bed
                )

                outside_bed_latch_blocked = (
                    zones.has_bed
                    and not on_bed
                    and tr.bed_accum_sec < 1.0
                    and not tr.was_on_bed
                    and tr.reason != "manual_override"
                    and (
                        self._door_visit_active
                        or (
                            self._patient_in_bed_latched
                            and (obs.track_id != self._latched_patient_track_id or homography.calibrated)
                        )
                    )
                )
                if (
                    confirmed_patient_id is None
                    and self._ever_locked_patient
                    and can_rehabilitate
                    and not tr.spawned_in_door
                    and not is_static_coat
                    and not tall_floor_stander
                    and not outside_bed_latch_blocked
                    and not explicit_visitor_lock
                ):
                    recent_loss = (
                        0.0 <= (now - self._last_patient_seen_at) <= 60.0
                        or (
                            self._patient_in_bed_latched
                            and near_bed
                            and not self._door_visit_active
                            and not self._recent_visitors
                        )
                    )
                    dist_to_last = (
                        math.hypot(
                            obs.body_center[0] - self._last_patient_pos[0],
                            obs.body_center[1] - self._last_patient_pos[1],
                        )
                        if self._last_patient_pos is not None
                        else 999.0
                    )

                    angle_reid = obs.torso_angle_from_vertical_deg()
                    standing_visitor_candidate = (
                        self._door_visit_active
                        and not on_bed
                        and not reclining_on_bed
                        and not tr.was_on_bed
                        and angle_reid is not None
                        and angle_reid < 28.0
                        and obs.aspect_ratio_wh < 0.55
                    )
                    eligible_for_reid = (on_bed or tr.has_moved) and not standing_visitor_candidate
                    tr_mode_vec = self._get_track_mode_vec(tr, night_ir)
                    vis_sim = self._max_visitor_outfit_sim(tr_mode_vec, night_ir=night_ir)

                    # 4а. Восстановление по воротниковому якорю текущей модальности (день 48D / ночь 64D)
                    if eligible_for_reid and mode_anchor is not None and tr_mode_vec is not None:
                        sim = cosine_sim(mode_anchor, tr_mode_vec)
                        if (
                            sim >= 0.78
                            and sim >= vis_sim + 0.04
                            and (on_bed or (recent_loss and dist_to_last <= 0.26))
                        ):
                            tr.role = Role.PATIENT
                            tr.reason = (
                                "reid_persisted_anchor"
                                if mode_restored_disk and self._last_patient_seen_at < 0.0
                                else "reid_appearance"
                            )
                            confirmed_patient_id = obs.track_id

                    # 4б. Пространственно-временная преемственность у кровати
                    closer_recent_visitor = any(
                        0.0 <= (now - rv.last_seen) <= 15.0
                        and math.hypot(
                            obs.body_center[0] - rv.last_center[0],
                            obs.body_center[1] - rv.last_center[1],
                        )
                        + 0.03
                        < dist_to_last
                        for rv in self._recent_visitors.values()
                    )
                    max_cont_gap = (
                        86400.0
                        if (
                            self._patient_in_bed_latched
                            and (on_bed or near_bed)
                            and not self._door_visit_active
                            and not self._recent_visitors
                        )
                        else (120.0 if (on_bed and not self._recent_visitors) else 12.0)
                    )
                    max_cont_dist = (
                        0.28
                        if ((on_bed or self._patient_in_bed_latched) and not self._recent_visitors)
                        else 0.22
                    )
                    cont_gap_ok = (
                        0.0 <= (now - self._last_patient_seen_at) <= max_cont_gap
                        or (
                            self._patient_in_bed_latched
                            and (on_bed or near_bed)
                            and not self._door_visit_active
                            and not self._recent_visitors
                            and self._last_patient_seen_at < 0.0
                        )
                    )
                    if (
                        confirmed_patient_id is None
                        and near_bed
                        and not standing_visitor_candidate
                        and (reclining_on_bed or tr.has_moved or on_bed)
                        and (reclining_on_bed or tr.reason != "visitor_id_switch_continuity")
                        and (reclining_on_bed or not closer_recent_visitor)
                        and self._last_patient_pos is not None
                        and cont_gap_ok
                        and dist_to_last <= max_cont_dist
                        and vis_sim < 0.82
                    ):
                        closer_other = False
                        for o_other in observations:
                            if o_other.track_id == obs.track_id:
                                continue
                            tr_o = self._tracks.get(o_other.track_id)
                            if tr_o is not None and tr_o.role == Role.VISITOR:
                                continue
                            d_other = math.hypot(
                                o_other.body_center[0] - self._last_patient_pos[0],
                                o_other.body_center[1] - self._last_patient_pos[1],
                            )
                            if d_other < dist_to_last:
                                closer_other = True
                                break
                        if not closer_other:
                            tr.role = Role.PATIENT
                            tr.reason = "spatiotemporal_bed_continuity"
                            confirmed_patient_id = obs.track_id

                    # 4в. Возврат единственного лежащего в кровати при отсутствии якоря текущей модальности
                    if confirmed_patient_id is None and mode_anchor is None and reclining_on_bed:
                        non_visitor_obs = [
                            o
                            for o in observations
                            if o.track_id != obs.track_id
                            and (self._tracks.get(o.track_id) is None or self._tracks[o.track_id].role != Role.VISITOR)
                        ]
                        if len(non_visitor_obs) == 0 and (
                            not tr.spawned_outside_bed or obs.track_id == self._last_patient_track_id
                        ):
                            tr.role = Role.PATIENT
                            tr.reason = "reid_bed_position"
                            confirmed_patient_id = obs.track_id

                # Слой 1: Bed-Origin Lock для лежащего/полулежащего в кровати + Аксиома 1 («на кровати без двери = Бабушка»)
                if reclining_on_bed and tr.role != Role.VISITOR:
                    if len(observations) == 1 and not explicit_visitor_lock and not tr.spawned_in_door:
                        needed = (
                            self._initial_bed_lock_sec
                            if not self._ever_locked_patient
                            else min(3.0, self._bed_lock_sec)
                        )
                    else:
                        needed = (
                            self._initial_bed_lock_sec
                            if not self._ever_locked_patient and not tr.spawned_outside_bed
                            else self._bed_lock_sec
                        )
                    if tr.bed_accum_sec >= needed and confirmed_patient_id in (None, obs.track_id):
                        tr.role = Role.PATIENT
                        tr.reason = (
                            "initial_bed_occupant"
                            if not self._ever_locked_patient
                            else "bed_sleep_lock"
                        )
                        confirmed_patient_id = obs.track_id
                        self._ever_locked_patient = True
                elif (
                    tr.role == Role.UNKNOWN
                    and not self._door_visit_active
                    and zones.has_bed
                    and on_bed
                    and len(observations) == 1
                    and not is_static_coat
                    and not tall_floor_stander
                    and not tr.spawned_in_door
                    and not explicit_visitor_lock
                    and confirmed_patient_id in (None, obs.track_id)
                ):
                    sit_lock_sec = min(1.0, self._initial_bed_lock_sec)
                    if self._ever_locked_patient or (now - tr.first_seen) >= sit_lock_sec:
                        tr.role = Role.PATIENT
                        tr.reason = "reid_bed_position"
                        confirmed_patient_id = obs.track_id
                        self._ever_locked_patient = True

                # Если полигон кровати ещё не размечен — привязываем только реально двигающегося человека!
                if (
                    not zones.has_bed
                    and len(observations) == 1
                    and tr.role == Role.UNKNOWN
                    and tr.has_moved
                    and confirmed_patient_id is None
                ):
                    tr.role = Role.PATIENT
                    tr.reason = "single_person_no_zones"
                    confirmed_patient_id = obs.track_id

            strict_door_now = (
                zones.has_door
                and not self._door_visit_active
                and self._first_update_ts is not None
                and (now - self._first_update_ts) > 15.0
            )

            # 4.5. Совместный многоперсонный решатель при одновременном присутствии >= 2 людей
            if len(observations) >= 2:
                arb_pid = self._arbitrate_multi_person_roles(
                    observations,
                    zones,
                    now,
                    night_ir=night_ir,
                    allow_non_door_visitor=not strict_door_now,
                )
                if arb_pid is not None:
                    confirmed_patient_id = arb_pid

            # 5. Единственность PATIENT + расчёт Caregiver Co-Presence и строгий гейт записи раздельных якорей
            patient_obs: PoseObservation | None = None
            other_obs: list[PoseObservation] = []
            for obs in observations:
                tr = self._tracks[obs.track_id]
                if tr.role == Role.PATIENT and patient_obs is None:
                    patient_obs = obs
                else:
                    if tr.role == Role.PATIENT:
                        tr.role = Role.VISITOR
                        tr.reason = "duplicate_patient_guard"
                    other_obs.append(obs)

            caregiver_nearby = False
            patient_occluded_by_other = False
            if patient_obs is not None and other_obs:
                px, py = patient_obs.body_center
                reach = max(0.32, patient_obs.bbox_height * 1.6)
                for other in other_obs:
                    if not is_valid_human_pose(other, min_kp_conf=0.30, min_valid_kps=4, min_box_conf=0.48, zones=zones):
                        continue
                    other_tr = self._tracks.get(other.track_id)
                    if (
                        not strict_door_now
                        and other_tr is not None
                        and other_tr.role == Role.UNKNOWN
                        and (other_tr.has_moved or (now - other_tr.first_seen) >= 0.4)
                    ):
                        other_tr.role = Role.VISITOR
                        other_tr.reason = "co_present_with_patient"
                    confirmed_other = other_tr is not None and (
                        other_tr.role == Role.VISITOR
                        or (not strict_door_now and other_tr.confirmed_second_track and other_tr.has_moved)
                    )
                    ox, oy = other.body_center
                    dist = math.hypot(px - ox, py - oy)
                    if confirmed_other and (
                        dist <= reach
                        or is_confident_foreground_seated_person(
                            other,
                            zones.bed_polygon if zones.has_bed else None,
                            zones=zones,
                        )
                    ):
                        caregiver_nearby = True
                    if dist <= 0.18 or bbox_iou(patient_obs.bbox_norm, other.bbox_norm) >= 0.15:
                        patient_occluded_by_other = True
            elif patient_obs is None and other_obs and zones.has_bed:
                # Защита от окклюзии спиной родственника у кровати: подтверждённый VISITOR
                # внутри или ближе 0.18 к контуру кровати означает активный уход рядом с постелью
                for other in other_obs:
                    if not is_valid_human_pose(other, min_kp_conf=0.30, min_valid_kps=4, min_box_conf=0.48, zones=zones):
                        continue
                    other_tr = self._tracks.get(other.track_id)
                    if other_tr is None or other_tr.role != Role.VISITOR:
                        continue
                    near_bed_care = (
                        is_person_on_bed(other, zones.bed_polygon, zones=zones)
                        or polygon_distance_norm(other.body_center, zones.bed_polygon) >= -0.18
                        or (
                            other.mid_hip is not None
                            and polygon_distance_norm(other.mid_hip, zones.bed_polygon) >= -0.18
                        )
                    )
                    if near_bed_care:
                        caregiver_nearby = True
                        break

            # Обновляем позицию и эталон одежды бабушки СТРОГО в активном банке (день 48D ИЛИ ночь 64D)
            if patient_obs is not None:
                pat_tr = self._tracks[patient_obs.track_id]
                self._last_patient_track_id = patient_obs.track_id
                self._last_patient_pos = patient_obs.body_center
                self._last_patient_seen_at = now
                self._ever_locked_patient = True

                pat_on_bed = (
                    is_person_on_bed(patient_obs, zones.bed_polygon, zones=zones)
                    if zones.has_bed
                    else False
                )
                if zones.has_bed:
                    _, bed_max_y = polygon_y_bounds(zones.bed_polygon)
                    legs_hanging_now = are_legs_hanging_outside_bed(
                        patient_obs,
                        zones.bed_polygon,
                        zones=zones,
                    )
                    deep_ankles_on_floor = legs_hanging_now and any(
                        (pt := patient_obs.kp(idx, 0.48)) is not None
                        and pt[1] >= (bed_max_y + 0.03)
                        and not point_in_polygon(pt, zones.bed_polygon)
                        for idx in (15, 16)
                    )
                    if pat_on_bed and not deep_ankles_on_floor:
                        was_latched_before = self._patient_in_bed_latched
                        prev_bed_bbox = self._latched_patient_bed_bbox
                        self._patient_in_bed_latched = True
                        self._latched_patient_track_id = patient_obs.track_id
                        self._latched_bed_wall_ts = self._wall_clock()
                        self._sole_mover_at_door_ts = -1e9
                        if prev_bed_bbox is None or patient_obs.bbox_width >= 0.14:
                            self._latched_patient_bed_bbox = patient_obs.bbox_norm
                        if not was_latched_before or prev_bed_bbox is None:
                            self._save_anchor_state()
                    elif (
                        (not pat_on_bed)
                        and (
                            pat_tr.was_on_bed
                            or pat_tr.bed_accum_sec >= 1.0
                            or pat_tr.reason == "manual_override"
                            or (
                                not self._door_visit_active
                                and not homography.calibrated
                                and patient_obs.track_id == self._latched_patient_track_id
                            )
                        )
                    ) or (
                        deep_ankles_on_floor and not self._door_visit_active and not caregiver_nearby
                    ):
                        if self._patient_in_bed_latched:
                            self._patient_in_bed_latched = False
                            self._latched_bed_wall_ts = 0.0
                            self._save_anchor_state()
                any_box_overlap = any(
                    bbox_iou(patient_obs.bbox_norm, o.bbox_norm) >= 0.08 for o in other_obs
                )
                allow_face_anchor = (
                    pat_tr.reason == "face_patient"
                    and (not zones.has_bed or pat_on_bed)
                    and not any_box_overlap
                )
                trusted_for_anchor = (
                    pat_tr.reason == "manual_override"
                    or allow_face_anchor
                    or (
                        pat_tr.was_on_bed
                        and pat_tr.bed_accum_sec >= min(1.0, self._initial_bed_lock_sec)
                        and (pat_on_bed or pat_tr.has_moved)
                    )
                )

                pat_mode_vec = self._get_track_mode_vec(pat_tr, night_ir)
                cur_mode_anchor = self._get_mode_anchor(night_ir)
                cur_mode_updated = self._get_mode_updated_at(night_ir)
                if (
                    trusted_for_anchor
                    and pat_mode_vec is not None
                    and not patient_occluded_by_other
                    and not any_box_overlap
                    and (cur_mode_anchor is None or (now - cur_mode_updated) >= 20.0)
                ):
                    if (
                        cur_mode_anchor is None
                        or pat_tr.reason == "manual_override"
                        or allow_face_anchor
                    ):
                        self._set_mode_anchor(pat_mode_vec, night_ir=night_ir, now=now)
                        self._save_anchor_state()
                    elif cosine_sim(cur_mode_anchor, pat_mode_vec) >= 0.58:
                        blended = 0.80 * cur_mode_anchor + 0.20 * pat_mode_vec
                        norm = float(np.linalg.norm(blended))
                        if norm > 1e-6:
                            self._set_mode_anchor(blended / norm, night_ir=night_ir, now=now)
                            self._save_anchor_state()

                updated_mode_anchor = self._get_mode_anchor(night_ir)
                if (
                    updated_mode_anchor is not None
                    and pat_on_bed
                    and pat_tr.was_on_bed
                    and pat_tr.bed_accum_sec >= min(1.0, self._initial_bed_lock_sec)
                    and not patient_occluded_by_other
                    and not any_box_overlap
                ):
                    if night_ir:
                        self._clean_bed_anchor_night = updated_mode_anchor.copy()
                    else:
                        self._clean_bed_anchor_day = updated_mode_anchor.copy()

            # Актуализируем краткосрочную память гостей (СТРОГО запрещено записывать лежащих в кровати!)
            for obs in observations:
                tr = self._tracks.get(obs.track_id)
                if tr is None:
                    continue
                reclining_now = self._is_sleeping_or_reclining_on_bed(obs, tr, zones)
                if reclining_now or tr.role == Role.PATIENT:
                    self._recent_visitors.pop(obs.track_id, None)
                elif tr.role == Role.VISITOR:
                    prev_rv = self._recent_visitors.get(obs.track_id)
                    saved_vec = (
                        tr.appearance_vec.copy()
                        if tr.appearance_vec is not None
                        else (prev_rv.appearance_vec if prev_rv is not None else None)
                    )
                    saved_day = (
                        tr.appearance_vec_day.copy()
                        if tr.appearance_vec_day is not None
                        else (prev_rv.appearance_vec_day if prev_rv is not None else None)
                    )
                    saved_night = (
                        tr.appearance_vec_night.copy()
                        if tr.appearance_vec_night is not None
                        else (prev_rv.appearance_vec_night if prev_rv is not None else None)
                    )
                    self._recent_visitors[obs.track_id] = _RecentVisitorMemory(
                        track_id=obs.track_id,
                        last_seen=now,
                        last_center=obs.body_center,
                        bbox_norm=obs.bbox_norm,
                        appearance_vec=saved_vec,
                        appearance_vec_day=saved_day,
                        appearance_vec_night=saved_night,
                        reason=tr.reason,
                    )

            # Контроль редкого ухода единственного человека из комнаты через дверь без визита гостей
            if not self._door_visit_active and zones.has_door:
                if observations:
                    for obs_s in observations:
                        tr_s = self._tracks.get(obs_s.track_id)
                        if tr_s is None or tr_s.spawned_in_door or tr_s.role == Role.VISITOR:
                            continue
                        on_bed_s = is_person_on_bed(obs_s, zones.bed_polygon, zones=zones) if zones.has_bed else False
                        if not on_bed_s and self._is_touching_door(obs_s, zones):
                            self._sole_mover_at_door_ts = now
                            self._sole_mover_last_seen_ts = now
                        elif not on_bed_s and self._sole_mover_at_door_ts > 0.0:
                            self._sole_mover_last_seen_ts = now
                elif (
                    self._patient_in_bed_latched
                    and self._sole_mover_at_door_ts > 0.0
                    and (now - self._sole_mover_last_seen_ts) >= 2.0
                    and 0.0 <= (self._sole_mover_last_seen_ts - self._sole_mover_at_door_ts) <= 3.0
                ):
                    self._patient_in_bed_latched = False
                    self._latched_bed_wall_ts = 0.0
                    self._sole_mover_at_door_ts = -1e9
                    self._save_anchor_state()

            # 6. Закон сохранения Своих через дверной портал (DoorPortalTracker + оптическая занятость стула у двери)
            if zones.has_door:
                d_xs = [float(p[0]) for p in zones.door_polygon]
                d_ys = [float(p[1]) for p in zones.door_polygon]
                door_x1, door_y1, door_x2, door_y2 = min(d_xs), min(d_ys), max(d_xs), max(d_ys)
                door_h = max(0.05, door_y2 - door_y1)
                door_upper_box = (door_x1, door_y1, door_x2, min(1.0, door_y1 + 0.60 * door_h))
            else:
                door_y1 = 0.0
                door_h = 1.0
                door_upper_box = (0.0, 0.0, 0.0, 0.0)

            def _is_standing_door_transit(v_box: tuple[float, float, float, float]) -> bool:
                return (v_box[1] <= door_y1 + 0.54 * door_h) or ((v_box[3] - v_box[1]) >= 0.42 * door_h)

            live_obs_ids = {o.track_id for o in observations}
            non_bed_obs_in_room = [
                o
                for o in valid_obs
                if (
                    o.track_id in live_obs_ids
                    or o.track_id in self._tracks
                    or self._liveness_gate.is_track_moved(o.track_id)
                    or o.liveness_verified
                )
                and (not zones.has_bed or not is_person_on_bed(o, zones.bed_polygon, zones=zones))
            ]
            has_interior_non_bed_person = any(
                not self._is_touching_door(o, zones) for o in non_bed_obs_in_room
            )
            if self._door_visit_active and has_interior_non_bed_person:
                self._visitor_exiting_at_door = False
                self._last_visitor_seen_at = now

            active_visitors = [
                obs
                for obs in observations
                if obs.track_id in self._tracks and self._tracks[obs.track_id].role == Role.VISITOR
            ]
            if not self._door_visit_active and not active_visitors and cur_gray_small is not None and cur_textured:
                # Запоминаем эталон пустой комнаты (пустого стула и двери) ДО начала визита родственника
                self._pre_visit_gray_small = cur_gray_small
                self._pre_visit_frame_mean = float(np.mean(cur_gray_small))

            if active_visitors:
                self._last_visitor_seen_at = now
                primary_vis = max(active_visitors, key=lambda o: o.conf)
                self._latched_visitor_bbox = primary_vis.bbox_norm
                self._visitor_spot_occupied = True
                if cur_gray_small is not None and cur_textured:
                    v_roi = OpticalKeypointStabilizer._extract_blurred_roi(
                        cur_gray_small,
                        primary_vis.bbox_norm,
                    )
                    if v_roi is not None:
                        self._last_visitor_roi_blur = v_roi
                        self._last_visitor_frame_mean = float(np.mean(cur_gray_small))
                        self._last_visitor_roi_frame_id = id(frame_bgr) if frame_bgr is not None else None

                for v_obs in active_visitors:
                    v_tr = self._tracks[v_obs.track_id]
                    touching_door = self._is_touching_door(v_obs, zones)
                    if not touching_door and (
                        not zones.has_door
                        or polygon_distance_norm(v_obs.body_center, zones.door_polygon) < -0.04
                    ):
                        self._visitor_entered_interior = True
                        self._visitor_exiting_at_door = False
                    elif self._door_visit_active and zones.has_door and touching_door:
                        if _is_standing_door_transit(v_obs.bbox_norm):
                            if (
                                (now - self._door_visit_started_at) >= 3.0
                                or (now - v_tr.first_seen) >= 3.0
                            ):
                                self._last_visitor_at_door_ts = now
                                self._visitor_exiting_at_door = True
                        else:
                            # Низкое сидение на стуле у двери не является выходом стоя через проём двери
                            self._visitor_exiting_at_door = False
            elif self._door_visit_active:
                if (
                    cur_gray_small is not None
                    and cur_textured
                    and self._last_visitor_roi_blur is not None
                    and self._latched_visitor_bbox is not None
                ):
                    spot_drift = OpticalKeypointStabilizer.compute_anchor_drift(
                        cur_gray_small,
                        self._last_visitor_roi_blur,
                        self._last_visitor_frame_mean,
                        self._latched_visitor_bbox,
                    )
                    if self._pre_visit_gray_small is not None:
                        pre_roi = OpticalKeypointStabilizer._extract_blurred_roi(
                            self._pre_visit_gray_small,
                            self._latched_visitor_bbox,
                        )
                        pre_drift = (
                            OpticalKeypointStabilizer.compute_anchor_drift(
                                cur_gray_small,
                                pre_roi,
                                self._pre_visit_frame_mean,
                                self._latched_visitor_bbox,
                            )
                            if pre_roi is not None
                            else 1.0
                        )
                        differs_from_empty_spot = pre_drift >= 0.042
                    else:
                        differs_from_empty_spot = (
                            frame_bgr is not None and id(frame_bgr) != self._last_visitor_roi_frame_id
                        )
                    self._visitor_spot_occupied = bool(spot_drift < 0.042 and differs_from_empty_spot)
                else:
                    self._visitor_spot_occupied = False

                if self._visitor_spot_occupied:
                    # Гость физически сидит/стоит на последнем месте (например, на стуле у двери спиной к камере)
                    self._last_visitor_seen_at = now
                else:
                    upper_door_motion = (
                        OpticalKeypointStabilizer.compute_roi_optical_motion(
                            cur_gray_small,
                            self._portal_prev_gray,
                            door_upper_box,
                            door_upper_box,
                        )
                        if (zones.has_door and has_portal_opt_pair)
                        else 0.0
                    )
                    # Если гость на выходе в дверь сменил track_id на 1 кадр (и удерживается StaticObjectGate),
                    # фиксируем проход через дверь стоя в первые 3.5 с после последнего живого кадра гостя
                    if (
                        zones.has_door
                        and (now - self._door_visit_started_at) >= 3.0
                        and not self._visitor_exiting_at_door
                        and self._last_visitor_seen_at > -1e8
                        and 0.0 <= (now - self._last_visitor_seen_at) <= 3.5
                    ):
                        for raw_v in valid_obs:
                            if not zones.has_bed or not is_person_on_bed(raw_v, zones.bed_polygon, zones=zones):
                                if self._is_touching_door(raw_v, zones) and _is_standing_door_transit(raw_v.bbox_norm):
                                    self._last_visitor_at_door_ts = now
                                    self._visitor_exiting_at_door = True
                                    break

                    # Если гость встал со стула / ушёл спиной к камере через верхний дверной проём без скелета YOLO
                    if (
                        zones.has_door
                        and (now - self._door_visit_started_at) >= 3.0
                        and not self._visitor_exiting_at_door
                        and upper_door_motion >= 0.015
                    ):
                        self._last_visitor_at_door_ts = now
                        self._visitor_exiting_at_door = True
                        self._last_visitor_seen_at = now

                    door_busy_now = (
                        zones.has_door
                        and has_portal_opt_pair
                        and (now - self._portal_prev_gray_ts) <= 1.0
                        and self._portal_prev_gray_ts > self._last_visitor_seen_at
                        and upper_door_motion >= 0.020
                    )
                    idle_sec = (
                        (now - self._last_visitor_seen_at)
                        if self._last_visitor_seen_at > -1e8
                        else (now - self._door_visit_started_at)
                    )
                    exited_via_door = (
                        not non_bed_obs_in_room
                        and not self._visitor_spot_occupied
                        and self._visitor_exiting_at_door
                        and self._last_visitor_at_door_ts >= self._door_visit_started_at
                        and idle_sec >= 2.5
                        and not door_busy_now
                    )
                    timed_out = (
                        not non_bed_obs_in_room
                        and not self._visitor_spot_occupied
                        and idle_sec >= self._door_visit_idle_timeout_sec
                    )
                    if exited_via_door or timed_out:
                        self._reset_door_visit_portal()

            self._last_update_now = now
            if other_obs or active_visitors or (patient_obs is None and observations):
                self._other_person_seen_at = now

            if self._door_visit_active:
                caregiver_nearby = True

            if not same_portal_frame:
                if frame_bgr is None:
                    self._portal_prev_gray = None
                    self._portal_prev_gray_ts = -1e9
                    self._portal_prev_textured = None
                    self._portal_last_frame_id = None
                elif cur_gray_small is not None:
                    self._portal_prev_gray = cur_gray_small
                    self._portal_prev_gray_ts = now
                    self._portal_prev_textured = cur_textured
                    self._portal_last_frame_id = id(frame_bgr)
            elif now > self._portal_prev_gray_ts:
                self._portal_prev_gray_ts = now

            active_snapshot = {tid: self._tracks[tid] for tid in active_ids if tid in self._tracks}
            return active_snapshot, caregiver_nearby
