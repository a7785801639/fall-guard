from __future__ import annotations

import asyncio
import contextlib
import ctypes
import inspect
import json
import logging
import math
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .archive import ClipArchiveStore
from .capture import VideoStreamCapture
from .config import Settings
from .detector import PoseTracker, _load_font, render_annotated_frame
from .camera_health import CameraHealthMonitor
from .devices import AlertFeed, DeviceStore
from .feedback import FalseAlertFeedbackStore, is_safe_alert_id
from .geometry import (
    KP_L_ANKLE,
    KP_R_ANKLE,
    PoseObservation,
    are_legs_hanging_outside_bed,
    bbox_iou,
    bbox_polygon_overlap_ratio,
    is_person_on_bed,
    is_pose_touching_door,
    is_standing_in_front_of_bed,
    point_in_polygon,
    sanitize_collapsed_bed_keypoints,
)
from .notifier import TelegramNotifier
from .patient_id import FaceRegistry, PatientIdentifier, Role, TrackIdentity
from .qr_svg import render_qr_svg
from .recorder import ClipRecorder, build_mp4_from_jpegs
from .runtime_prefs import RuntimePrefsStore
from .state_machine import AlertEvent, FallStateMachine, PatientState
from .stream_hub import H264StreamHub
from .timeline import TimelineStore
from .zones import ZonesConfig, ZoneStore, polygon_coverage_ratio

logger = logging.getLogger("fallguard.engine")

HUMAN_STATE_RU: dict[str, str] = {
    "idle": "Спокойно",
    "paused": "Охрана на паузе",
    "in_bed": "В кровати · спокойно",
    "caregiver": "Рядом родственник",
    "help_call": "Зовёт на помощь (поднята рука)",
    "sitting_up": "Садится / свесила ноги",
    "left_bed": "Встала с кровати одна",
    "missing": "Нет в кровати и в кадре",
    "fallen": "Экстренно: падение вне кровати!",
}

TRACK_REASON_RU: dict[str, str] = {
    "initial_bed_occupant": "лежит в кровати",
    "bed_sleep_lock": "лежит в кровати",
    "sole_bed_sleeper_recovery": "лежит в кровати",
    "spatiotemporal_bed_continuity": "на месте в кровати",
    "tall_floor_visitor": "родственник у кровати",
    "door_visit_companion": "рядом в комнате",
    "reid_bed_position": "на месте в кровати",
    "face_patient": "узнана по лицу",
    "face_visitor": "узнан по лицу",
    "reid_appearance": "по одежде",
    "reid_persisted_anchor": "по одежде",
    "visitor_outfit_match": "по одежде",
    "visitor_outfit": "по одежде",
    "door_entry": "вошёл через дверь",
    "door_visit": "вошёл через дверь",
    "door_visitor_continuity": "вошёл через дверь",
    "border_standing_entry": "вошёл в комнату",
    "foreground_floor_entry": "рядом с кроватью",
    "multi_person_bed_arbitration": "на кровати",
    "second_person_while_patient_active": "рядом с бабушкой",
    "co_present_with_patient": "рядом с бабушкой",
    "duplicate_patient_guard": "рядом с бабушкой",
    "visitor_id_switch_continuity": "рядом с кроватью",
    "manual_override": "указана вручную",
    "manual_patient_override": "указана вручную",
    "manual_visitor_override": "указан вручную",
    "demoted_by_manual_override": "указан вручную",
    "id_switch_demoted_visitor": "рядом с бабушкой",
    "id_switch_recovered": "на месте в кровати",
    "promoted_after_visitor_override": "лежит в кровати",
    "multi_person_visitor_score": "рядом с бабушкой",
    "anchor_cleared": "определяется",
    "single_person_no_zones": "один в комнате",
    "bed_covered_persistence": "лежит в кровати · под одеялом",
    "door_visit_seated": "в комнате · вошёл через дверь",
    "door_visit_present": "в комнате · вошёл через дверь",
}

SYNTHETIC_PERSISTENCE_REASONS: frozenset[str] = frozenset(
    {
        "bed_covered_persistence",
        "door_visit_seated",
        "door_visit_present",
    }
)


def translate_role_reason(role_val: str, reason: str) -> str:
    r = (reason or "").strip()
    if r in TRACK_REASON_RU:
        return TRACK_REASON_RU[r]
    rl = r.lower()
    if "manual" in rl:
        return "указана вручную" if role_val == Role.PATIENT.value else "указан вручную"
    if "face" in rl:
        return "узнана по лицу" if role_val == Role.PATIENT.value else "узнан по лицу"
    if "door" in rl:
        return "вошёл через дверь"
    if "border" in rl:
        return "вошёл в комнату"
    if ("outfit" in rl or "appearance" in rl or "anchor" in rl) and "cleared" not in rl:
        return "по одежде"
    if "arbiter" in rl or "co_present" in rl or "second_person" in rl:
        return "лежит в кровати" if role_val == Role.PATIENT.value else "рядом с бабушкой"
    if "bed" in rl or "sleeper" in rl:
        return "лежит в кровати"
    if role_val == Role.PATIENT.value:
        return "лежит в кровати"
    if role_val == Role.VISITOR.value:
        return "в комнате"
    return "определяется"


def _fmt_ru_duration(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    if sec < 60:
        return f"{sec} с"
    mins = sec // 60
    if mins < 60:
        return f"{mins} мин"
    hrs = mins // 60
    rem_m = mins % 60
    return f"{hrs} ч {rem_m} мин" if rem_m > 0 else f"{hrs} ч"


def _is_obs_in_ignore_zone(obs: PoseObservation, poly: list[list[float]]) -> bool:
    if len(poly) < 3:
        return False
    if point_in_polygon(obs.body_center, poly):
        return True
    mh = obs.mid_hip
    if mh is not None and point_in_polygon(mh, poly):
        return True
    x1, y1, x2, y2 = obs.bbox_norm
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1e-4 or bh <= 1e-4:
        return False
    inside = 0
    total = 25
    for ix in range(5):
        px = x1 + bw * (ix + 0.5) / 5.0
        for iy in range(5):
            py = y1 + bh * (iy + 0.5) / 5.0
            if point_in_polygon((px, py), poly):
                inside += 1
    return (inside / float(total)) >= 0.55


def _is_obs_touching_door(obs: PoseObservation, door_poly: list[list[float]]) -> bool:
    if len(door_poly) < 3:
        return False
    for pt in (obs.body_center, obs.mid_shoulder, obs.mid_hip, obs.head_center):
        if pt is not None and point_in_polygon(pt, door_poly):
            return True
    return bbox_polygon_overlap_ratio(obs.bbox_norm, door_poly) >= 0.12


def _sanitize_bed_obs_compat(
    obs: PoseObservation,
    bed_polygon: list[list[float]],
    other_observations: Sequence[PoseObservation] = (),
    zones: ZonesConfig | None = None,
) -> PoseObservation:
    try:
        return sanitize_collapsed_bed_keypoints(  # type: ignore[call-arg]
            obs,
            bed_polygon,
            other_observations=other_observations,
            zones=zones,
        )
    except TypeError:
        return sanitize_collapsed_bed_keypoints(obs, bed_polygon)


def _filter_ignored_observations(
    observations: list[PoseObservation],
    zones: ZonesConfig,
) -> list[PoseObservation]:
    ignore_polys = getattr(zones, "ignore_polygons", None)
    if not ignore_polys:
        return observations
    door_poly = getattr(zones, "door_polygon", None)
    has_door = isinstance(door_poly, list) and len(door_poly) >= 3
    has_bed = zones.has_bed
    safe_bed_ignores = [
        poly
        for poly in ignore_polys
        if not (has_bed and polygon_coverage_ratio(poly, zones.bed_polygon) >= 0.20)
    ]
    out: list[PoseObservation] = []
    for obs in observations:
        if has_door and _is_obs_touching_door(obs, door_poly):
            out.append(obs)
            continue
        active_polys = (
            safe_bed_ignores
            if (has_bed and is_person_on_bed(obs, zones.bed_polygon, zones=zones))
            else ignore_polys
        )
        if not any(_is_obs_in_ignore_zone(obs, poly) for poly in active_polys):
            out.append(obs)
    return out


class _FallbackThumbnailStore:
    """Резервное хранилище мини-снимков хроники, если app.thumbs недоступен."""

    def __init__(self, thumbs_dir: Any) -> None:
        from pathlib import Path

        self._dir = Path(thumbs_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = []
        self._last_periodic_ts = 0.0
        self._last_state: str | None = None
        self._last_record_ts = 0.0

    def maybe_record(
        self,
        frame_bgr: np.ndarray,
        state: str,
        now_ts: float | None = None,
    ) -> dict[str, Any] | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        ts = float(time.time() if now_ts is None else now_ts)
        with self._lock:
            state_changed = self._last_state is not None and state != self._last_state
            due_periodic = self._last_periodic_ts <= 0.0 or (ts - self._last_periodic_ts) >= 60.0
            due_state = state_changed and (ts - self._last_record_ts) >= 5.0
            if not (due_periodic or due_state):
                return None
            small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
            if not ok:
                return None
            entry_id = f"{int(ts * 1000)}_{secrets.token_hex(2)}"
            target = self._dir / f"{entry_id}.jpg"
            target.write_bytes(enc.tobytes())
            entry = {"id": entry_id, "ts": round(ts, 3), "state": str(state)}
            self._items.append(entry)
            self._items.sort(key=lambda x: float(x["ts"]))
            if due_periodic:
                self._last_periodic_ts = ts
            self._last_record_ts = ts
            self._last_state = str(state)
            return dict(entry)

    def list_index(self, since_ts: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(it) for it in self._items if float(it["ts"]) >= float(since_ts)]

    def get_by_id(self, entry_id: str) -> Any:
        if not entry_id or "/" in entry_id or "\\" in entry_id or ".." in entry_id:
            return None
        cand = (self._dir / f"{entry_id}.jpg").resolve()
        if cand.parent != self._dir.resolve() or not cand.is_file():
            return None
        return cand

    def get_nearest(self, target_ts: float, max_delta_sec: float = 900.0) -> Any:
        with self._lock:
            if not self._items:
                return None
            best = min(self._items, key=lambda it: abs(float(it["ts"]) - float(target_ts)))
            if abs(float(best["ts"]) - float(target_ts)) > float(max_delta_sec):
                return None
            best_copy = dict(best)
        path = self.get_by_id(str(best_copy["id"]))
        if path is None:
            return None
        return path, best_copy


def _create_thumbnail_store(thumbs_dir: Any) -> Any:
    try:
        from .thumbs import TimelineThumbnailStore  # type: ignore[import-not-found]

        return TimelineThumbnailStore(thumbs_dir, thumb_size=(640, 360), jpeg_quality=70)
    except Exception:
        return _FallbackThumbnailStore(thumbs_dir)


def _detect_night_ir_frame(frame_bgr: np.ndarray | None, identifier: Any = None) -> bool:
    """
    Определяет монохромный ночной ИК-режим камеры:
    сначала пробует штатную функцию из app.patient_id / app.geometry,
    иначе оценивает среднюю насыщенность HSV mean(S) < 12.0.
    """
    if frame_bgr is not None and frame_bgr.ndim == 3 and frame_bgr.size > 0:
        for mod_name in (".patient_id", ".geometry"):
            try:
                mod = __import__(f"{__package__}{mod_name}", fromlist=["is_night_ir_frame"])
                fn = getattr(mod, "is_night_ir_frame", None)
                if callable(fn):
                    return bool(fn(frame_bgr))
            except Exception:
                pass
        small = (
            cv2.resize(frame_bgr, (128, 72), interpolation=cv2.INTER_AREA)
            if frame_bgr.shape[1] > 160 or frame_bgr.shape[0] > 96
            else frame_bgr
        )
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1])) < 12.0

    if identifier is not None:
        val = getattr(identifier, "is_night_ir", None)
        if isinstance(val, bool):
            return val
    return False


def _extract_anchor_flags(identifier: Any, is_night_ir: bool) -> tuple[bool, bool]:
    """Считывает готовность раздельных якорей внешности (день RGB / ночь ИК) с поддержкой обратной совместимости."""
    day_val: bool | None = None
    night_val: bool | None = None

    for attr in ("anchor_day_ready", "has_day_anchor", "day_anchor_ready"):
        v = getattr(identifier, attr, None)
        if isinstance(v, bool):
            day_val = v
            break
    if day_val is None:
        for vec_attr in ("anchor_vec_day", "_anchor_vec_day"):
            if hasattr(identifier, vec_attr):
                day_val = getattr(identifier, vec_attr) is not None
                break

    for attr in ("anchor_night_ready", "has_night_anchor", "night_anchor_ready"):
        v = getattr(identifier, attr, None)
        if isinstance(v, bool):
            night_val = v
            break
    if night_val is None:
        for vec_attr in ("anchor_vec_night", "_anchor_vec_night"):
            if hasattr(identifier, vec_attr):
                night_val = getattr(identifier, vec_attr) is not None
                break

    if day_val is not None or night_val is not None:
        return bool(day_val), bool(night_val)

    has_legacy = bool(getattr(identifier, "has_appearance_anchor", False))
    if not has_legacy:
        return False, False
    return (False, True) if is_night_ir else (True, False)


def _build_floor_homography_matrix(zones: ZonesConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Строит прямую и обратную матрицы гомографии 4-точечного полигона пола."""
    if not zones.has_floor or len(zones.floor_polygon) != 4:
        return None
    src = np.array([[float(p[0]), float(p[1])] for p in zones.floor_polygon[:4]], dtype=np.float32)
    if float(cv2.contourArea(src * 1000.0)) < 5.0:
        return None
    w_cm = max(20.0, float(zones.floor_width_cm))
    d_cm = max(20.0, float(zones.floor_depth_cm))
    dst = np.array(
        [[0.0, 0.0], [w_cm, 0.0], [w_cm, d_cm], [0.0, d_cm]],
        dtype=np.float32,
    )
    try:
        h_mat = cv2.getPerspectiveTransform(src, dst)
        if abs(float(np.linalg.det(h_mat))) < 1e-9:
            return None
        h_inv = np.linalg.inv(h_mat)
        return h_mat, h_inv
    except Exception:
        return None


def _is_homography_calibrated(zones: ZonesConfig) -> bool:
    try:
        from . import homography as hom_mod  # type: ignore[import-not-found]

        fh_cls = getattr(hom_mod, "FloorHomography", None)
        if fh_cls is not None:
            if hasattr(fh_cls, "from_zones"):
                inst = fh_cls.from_zones(zones)
            else:
                sig = inspect.signature(fh_cls.__init__).parameters
                kw: dict[str, Any] = {}
                if "floor_polygon" in sig:
                    kw["floor_polygon"] = zones.floor_polygon
                elif "polygon" in sig:
                    kw["polygon"] = zones.floor_polygon
                if "floor_width_cm" in sig:
                    kw["floor_width_cm"] = zones.floor_width_cm
                elif "width_cm" in sig:
                    kw["width_cm"] = zones.floor_width_cm
                if "floor_depth_cm" in sig:
                    kw["floor_depth_cm"] = zones.floor_depth_cm
                elif "depth_cm" in sig:
                    kw["depth_cm"] = zones.floor_depth_cm
                if "mattress_height_cm" in sig:
                    kw["mattress_height_cm"] = zones.mattress_height_cm
                inst = fh_cls(**kw)
            for cal_attr in ("is_calibrated", "calibrated", "ready", "is_valid"):
                cal_val = getattr(inst, cal_attr, None)
                if isinstance(cal_val, bool):
                    return cal_val
                if callable(cal_val):
                    return bool(cal_val())
    except Exception:
        pass
    return _build_floor_homography_matrix(zones) is not None


@contextlib.contextmanager
def _win32_highres_timer(period_ms: int = 1) -> Iterator[None]:
    """Включает 1-мс квант планировщика Windows (winmm.timeBeginPeriod) на время цикла инференса."""
    winmm = getattr(ctypes, "windll", None) if os.name == "nt" else None
    enabled = False
    if winmm is not None:
        try:
            if int(winmm.winmm.timeBeginPeriod(int(period_ms))) == 0:
                enabled = True
        except Exception:
            enabled = False
    try:
        yield
    finally:
        if enabled and winmm is not None:
            try:
                winmm.winmm.timeEndPeriod(int(period_ms))
            except Exception:
                pass


def _compute_track_25d_metrics(
    obs: PoseObservation,
    zones: ZonesConfig,
) -> tuple[float | None, list[float] | None]:
    """
    Возвращает (height_cm, floor_xy_cm) для трека по калиброванному полигону пола:
    делегирует расчёт в app.homography.FloorHomography.from_zones(zones).
    Для лежащих в кровати поз штатно возвращает (None, None) без ложного роста и сдвига.
    """
    hom_height: float | None = None
    hom_xy: list[float] | None = None
    try:
        from . import homography as hom_mod  # type: ignore[import-not-found]

        fh_cls = getattr(hom_mod, "FloorHomography", None)
        if fh_cls is not None and hasattr(fh_cls, "from_zones"):
            inst = fh_cls.from_zones(
                zones,
                bed_polygon=zones.bed_polygon if zones.has_bed else None,
            )
            if not bool(getattr(inst, "is_calibrated", True)):
                return None, None
            if bool(getattr(inst, "is_valid", False)):
                res = inst.measure_pose(
                    obs,
                    bed_polygon=zones.bed_polygon if zones.has_bed else None,
                )
                h_val = res.get("height_cm") if isinstance(res, dict) else getattr(res, "height_cm", None)
                xy_val = res.get("floor_xy_cm") if isinstance(res, dict) else getattr(res, "floor_xy_cm", None)
                if h_val is not None:
                    hom_height = round(float(h_val), 1)
                if isinstance(xy_val, (list, tuple)) and len(xy_val) >= 2:
                    hom_xy = [round(float(xy_val[0]), 1), round(float(xy_val[1]), 1)]
                return hom_height, hom_xy
    except Exception:
        pass

    mats = _build_floor_homography_matrix(zones)
    if mats is None:
        return hom_height, hom_xy
    h_mat, h_inv = mats

    ak_pts = [p for p in (obs.kp(KP_L_ANKLE, 0.32), obs.kp(KP_R_ANKLE, 0.32)) if p is not None]
    angle = obs.torso_angle_from_vertical_deg()
    is_horiz = (angle is not None and angle >= 58.0) or obs.aspect_ratio_wh >= 1.15
    if ak_pts and not is_horiz:
        contact_u = sum(p[0] for p in ak_pts) / len(ak_pts)
        contact_v = max(p[1] for p in ak_pts)
    elif is_horiz:
        contact_u, contact_v = obs.body_center
    else:
        contact_u = (obs.bbox_norm[0] + obs.bbox_norm[2]) * 0.5
        contact_v = obs.bbox_norm[3]

    pt_arr = np.array([[[float(contact_u), float(contact_v)]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt_arr, h_mat)[0, 0]
    fx_cm = float(np.clip(proj[0], -500.0, 800.0))
    fy_cm = float(np.clip(proj[1], -500.0, 800.0))
    floor_xy = hom_xy if hom_xy is not None else [round(fx_cm, 1), round(fy_cm, 1)]

    if hom_height is not None:
        return hom_height, floor_xy

    probe_world = np.array(
        [[[fx_cm, fy_cm], [fx_cm + 100.0, fy_cm]]],
        dtype=np.float32,
    )
    back_uv = cv2.perspectiveTransform(probe_world, h_inv)[0]
    du_norm = float(np.hypot(back_uv[1, 0] - back_uv[0, 0], back_uv[1, 1] - back_uv[0, 1]))
    cm_per_norm_v = float(
        np.clip((100.0 / max(0.05, du_norm)) / (16.0 / 9.0), 160.0, 420.0)
    )

    on_bed = zones.has_bed and is_person_on_bed(obs, zones.bed_polygon)
    if on_bed:
        hip_y = obs.mid_hip[1] if obs.mid_hip is not None else obs.body_center[1]
        head_y = obs.head_center[1] if obs.head_center is not None else obs.upper_body_point[1]
        rise_v = max(0.0, hip_y - head_y)
        height_cm = round(float(zones.mattress_height_cm) + rise_v * cm_per_norm_v * 0.85, 1)
    elif is_horiz:
        height_cm = round(float(np.clip(obs.bbox_height * cm_per_norm_v * 0.42, 12.0, 42.0)), 1)
    else:
        top_y = min(
            obs.bbox_norm[1],
            obs.head_center[1] if obs.head_center is not None else obs.bbox_norm[1],
        )
        span_v = max(0.03, contact_v - top_y)
        height_cm = round(float(np.clip(span_v * cm_per_norm_v, 35.0, 215.0)), 1)

    return height_cm, floor_xy


KP_TELEMETRY_MIN_CONF: float = 0.48
OPTICAL_MOTION_STILL_RATIO: float = 0.012
OPTICAL_ANCHOR_DRIFT_RATIO: float = 0.050
MAX_SLEEP_COAST_SEC: float = 3600.0


def _prepare_optical_gray(frame_bgr: np.ndarray | None) -> np.ndarray | None:
    """Готовит уменьшенный сглаженный серый кадр 160x90 для быстрой оценки оптического покоя ROI."""
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        return None
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (3, 3), 0)
    except Exception:
        return None


def _compute_roi_optical_motion(
    cur_gray: np.ndarray | None,
    prev_gray: np.ndarray | None,
    bbox_norm: tuple[float, float, float, float],
    exclude_bboxes: Sequence[tuple[float, float, float, float]] = (),
) -> float | None:
    """Доля пикселей внутри расширенного бокса трека с |I_t - I_{t-1}| > 10 (подавляет ИК-зерно матрицы)."""
    if cur_gray is None or prev_gray is None or cur_gray.shape != prev_gray.shape:
        return None
    h, w = cur_gray.shape[:2]
    x1 = max(0, int(math.floor((float(bbox_norm[0]) - 0.02) * w)))
    y1 = max(0, int(math.floor((float(bbox_norm[1]) - 0.02) * h)))
    x2 = min(w, int(math.ceil((float(bbox_norm[2]) + 0.02) * w)))
    y2 = min(h, int(math.ceil((float(bbox_norm[3]) + 0.02) * h)))
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    diff = cv2.absdiff(cur_gray[y1:y2, x1:x2], prev_gray[y1:y2, x1:x2])
    if exclude_bboxes:
        mask = np.ones(diff.shape, dtype=bool)
        for ex in exclude_bboxes:
            ex1 = max(0, int(math.floor(float(ex[0]) * w)) - x1)
            ey1 = max(0, int(math.floor(float(ex[1]) * h)) - y1)
            ex2 = min(x2 - x1, int(math.ceil(float(ex[2]) * w)) - x1)
            ey2 = min(y2 - y1, int(math.ceil(float(ex[3]) * h)) - y1)
            if ex2 > ex1 and ey2 > ey1:
                mask[ey1:ey2, ex1:ex2] = False
        valid_px = int(np.count_nonzero(mask))
        if valid_px >= 8:
            return float(np.mean(diff[mask] > 10))
    return float(np.mean(diff > 10))


def _compute_roi_anchor_drift(
    cur_gray: np.ndarray | None,
    anchor_gray: np.ndarray | None,
    bbox_norm: tuple[float, float, float, float],
) -> float | None:
    """Доля пикселей ROI с локальным уходом > 14 относительно опорного кадра сна (без учёта плавной яркости)."""
    if cur_gray is None or anchor_gray is None or cur_gray.shape != anchor_gray.shape:
        return None
    h, w = cur_gray.shape[:2]
    x1 = max(0, int(math.floor((float(bbox_norm[0]) - 0.02) * w)))
    y1 = max(0, int(math.floor((float(bbox_norm[1]) - 0.02) * h)))
    x2 = min(w, int(math.ceil((float(bbox_norm[2]) + 0.02) * w)))
    y2 = min(h, int(math.ceil((float(bbox_norm[3]) + 0.02) * h)))
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    delta = cur_gray[y1:y2, x1:x2].astype(np.int16) - anchor_gray[y1:y2, x1:x2].astype(np.int16)
    mean_shift = float(np.mean(delta))
    return float(np.mean(np.abs(delta - mean_shift) > 14.0))


@dataclass
class _TrackTelemetryState:
    """Память стабилизации выходной телеметрии трека для 0.0 px дрожания и удержания во сне."""

    track_id: int
    last_seen_mono: float
    anchor_bbox: tuple[float, float, float, float]
    anchor_kps: list[tuple[float, float, float]]
    active_mask: list[bool]
    frozen_bbox: list[float]
    frozen_kps: list[list[float]]
    frozen_angle_deg: float
    frozen_hide_skeleton: bool
    frozen_height_cm: float | None
    frozen_floor_xy_cm: list[float] | None
    was_still: bool = False
    role_val: str = "unknown"
    reason_code: str = ""
    bed_sec: float = 0.0
    on_bed: bool = False
    occluded_by_other: bool = False
    last_confirmed_mono: float = 0.0
    anchor_opt_gray: np.ndarray | None = field(default=None, repr=False)


class TelemetrySubscription:
    """Потокобезопасная подписка клиента WebSocket /ws/telemetry на тики инференса 12 Гц."""

    def __init__(
        self,
        engine: FallGuardEngine,
        loop: asyncio.AbstractEventLoop,
        maxsize: int = 4,
    ) -> None:
        self._engine = engine
        self.loop = loop
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max(2, maxsize))
        self._closed = False

    async def __aenter__(self) -> TelemetrySubscription:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _enqueue_in_loop(self, payload: dict[str, Any] | None) -> None:
        if self._closed and payload is not None:
            return
        while self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def push_threadsafe(self, payload: dict[str, Any] | None) -> None:
        if self._closed and payload is not None:
            return
        if self.loop.is_closed():
            return
        try:
            self.loop.call_soon_threadsafe(self._enqueue_in_loop, payload)
        except RuntimeError:
            pass

    async def get(self) -> dict[str, Any] | None:
        if self._closed and self.queue.empty():
            return None
        return await self.queue.get()

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter_messages()

    async def _iter_messages(self) -> AsyncIterator[dict[str, Any]]:
        try:
            while not self._closed:
                msg = await self.queue.get()
                if msg is None:
                    break
                yield msg
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._engine.unsubscribe_telemetry(self)
        self.push_threadsafe(None)


class FallGuardEngine:
    """Главный рабочий цикл видеоаналитики, идентификации бабушки, архива и тревог."""

    MAX_EVENTS_LOG: int = 200

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)

        self.zones = ZoneStore(self.settings.data_dir / "zones.json")
        # Храним полные 24 часа на диске, поддерживая отдачу окна 6 ч и 24 ч
        self.timeline = TimelineStore(
            self.settings.data_dir / "timeline.json",
            window_sec=86400.0,
        )
        self.thumbs = _create_thumbnail_store(self.settings.data_dir / "thumbs")
        self.prefs = RuntimePrefsStore(
            self.settings.data_dir / "runtime_prefs.json",
            initial_defaults={
                "sit_confirm_seconds": self.settings.sit_confirm_seconds,
                "leave_confirm_seconds": self.settings.leave_confirm_seconds,
                "fall_confirm_seconds": self.settings.fall_confirm_seconds,
                "missing_confirm_seconds": self.settings.missing_confirm_seconds,
            },
        )
        self.archive = ClipArchiveStore(
            self.settings.data_dir / "clips",
            max_clips=60,
            max_total_mb=250.0,
        )
        self.feedback = FalseAlertFeedbackStore(self.settings.data_dir / "feedback")
        self.devices = DeviceStore(self.settings.data_dir / "devices.json")
        self.camera_health = CameraHealthMonitor(self.settings.data_dir)
        self._sync_camera_health_zones()
        self.alert_feed = AlertFeed()
        self.faces = FaceRegistry(
            faces_dir=self.settings.data_dir / "faces",
            models_dir=self.settings.models_dir,
            threshold=self.settings.face_match_threshold,
        )

        ident_kwargs: dict[str, object] = {
            "bed_lock_seconds": self.settings.bed_lock_seconds,
            "initial_bed_lock_seconds": self.settings.initial_bed_lock_seconds,
            "face_registry": self.faces,
        }
        ident_params = inspect.signature(PatientIdentifier.__init__).parameters
        if "state_path" in ident_params:
            ident_kwargs["state_path"] = self.settings.data_dir / "patient_anchor.json"
        if "door_visit_idle_timeout_sec" in ident_params:
            ident_kwargs["door_visit_idle_timeout_sec"] = float(
                getattr(self.settings, "door_visit_idle_timeout_sec", 14400.0)
            )
        self.identifier = PatientIdentifier(**ident_kwargs)  # type: ignore[arg-type]

        prefs_now = self.prefs.get()
        fsm_kwargs: dict[str, object] = {
            "sit_confirm_seconds": prefs_now["sit_confirm_seconds"],
            "leave_confirm_seconds": prefs_now["leave_confirm_seconds"],
            "fall_confirm_seconds": prefs_now["fall_confirm_seconds"],
            "missing_confirm_seconds": prefs_now["missing_confirm_seconds"],
            "alert_cooldown_seconds": self.settings.alert_cooldown_seconds,
            "suppress_sitting_when_caregiver": self.settings.suppress_sitting_when_caregiver,
        }
        fsm_params = inspect.signature(FallStateMachine.__init__).parameters
        for extra_key in (
            "enable_help_gesture",
            "help_confirm_seconds",
            "prolonged_absence_minutes",
            "immobility_reminder_minutes",
            "night_strict_mode",
            "night_start_hour",
            "night_end_hour",
            "floor_safety_net",
        ):
            if extra_key in fsm_params and extra_key in prefs_now:
                fsm_kwargs[extra_key] = prefs_now[extra_key]
        self.fsm = FallStateMachine(**fsm_kwargs)  # type: ignore[arg-type]
        if hasattr(self.fsm, "update_runtime_prefs"):
            self.fsm.update_runtime_prefs(**prefs_now)  # type: ignore[attr-defined]

        cap_kwargs: dict[str, object] = {
            "rtsp_url": self.settings.rtsp_url if self.settings.camera_configured else "",
            "capture_fps": max(self.settings.capture_fps, round(self.settings.infer_fps * 1.6, 2)),
            "pre_event_seconds": self.settings.pre_event_seconds,
        }
        cap_params = inspect.signature(VideoStreamCapture.__init__).parameters
        if "settings_path" in cap_params:
            cap_kwargs["settings_path"] = self.settings.data_dir / "camera.json"
        if "default_rotate" in cap_params:
            cap_kwargs["default_rotate"] = self.settings.frame_rotate
        if "default_flip_h" in cap_params:
            cap_kwargs["default_flip_h"] = self.settings.frame_flip_h
        if "default_enhance" in cap_params:
            cap_kwargs["default_enhance"] = self.settings.low_light_enhance
        if "hwaccel" in cap_params:
            cap_kwargs["hwaccel"] = self.settings.rtsp_hwaccel
        self.capture = VideoStreamCapture(**cap_kwargs)  # type: ignore[arg-type]

        self.pose = PoseTracker(
            model_name=self.settings.pose_model,
            models_dir=self.settings.models_dir,
            device=self.settings.device,
            conf=self.settings.pose_conf,
        )

        self._lock = threading.RLock()
        self._events_path = self.settings.data_dir / "events_log.json"
        self._events_log: deque[dict[str, Any]] = deque(maxlen=self.MAX_EVENTS_LOG)
        self._load_events_locked()

        venc_pref = getattr(self.settings, "video_encoder", "auto")
        if venc_pref == "auto" and self.settings.rtsp_hwaccel == "off":
            venc_pref = "off"
        self.recorder = ClipRecorder(hwaccel=venc_pref)

        stream_enabled = bool(getattr(self.settings, "stream_h264_enabled", True))
        stream_rtsp = (
            self.settings.rtsp_url
            if (self.settings.camera_configured and stream_enabled)
            else ""
        )
        self.stream_hub = H264StreamHub(rtsp_url=stream_rtsp)

        self._latest_jpeg: bytes | None = None
        self._latest_clean_jpeg: bytes | None = None
        self._latest_clean_frame_id: int = 0
        self._latest_raw_frame: np.ndarray | None = None
        self._latest_observations: list[PoseObservation] = []
        self._latest_identities: dict[int, TrackIdentity] = {}
        self._active_tracks_summary: list[dict] = []
        self._caregiver_nearby = False
        self._is_night_ir = False
        self._actual_fps = 0.0
        self._prev_opt_gray: np.ndarray | None = None
        self._telemetry_latch: dict[int, _TrackTelemetryState] = {}
        self._last_latched_bed_bbox: list[float] | None = None
        self._last_latched_patient_tid: int = 1
        self._last_latched_visitor_tid: int = -2
        self._telemetry_subs: set[TelemetrySubscription] = set()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._camera_was_online = False
        self._camera_offline_alerted = False
        self._prev_fall_unack = False
        self._fallback_cam_state: dict[str, object] = {
            "rotate": self.settings.frame_rotate,
            "flip_h": self.settings.frame_flip_h,
            "enhance": self.settings.low_light_enhance,
            "resolution": [0, 0],
        }

        notif_kwargs: dict[str, object] = {
            "bot_token": self.settings.tg_bot_token if self.settings.telegram_configured else "",
            "chat_ids": self.settings.tg_chat_ids,
            "on_pause_cmd": self.pause,
            "on_resume_cmd": self.resume,
            "get_snapshot_jpeg": self.get_latest_jpeg,
            "is_fall_unacknowledged": lambda: self.fsm.fall_unacknowledged,
            "repeat_interval_sec": self.settings.repeat_fall_alert_seconds,
            "max_repeats": self.settings.max_fall_repeats,
            "subscribers_path": self.settings.data_dir / "subscribers.json",
            "bot_username": self.settings.tg_bot_username,
            "get_status_text": self.get_human_status_line,
            "on_event": self._log_event,
        }
        notif_params = inspect.signature(TelegramNotifier.__init__).parameters
        if "get_report_text" in notif_params:
            notif_kwargs["get_report_text"] = self.get_care_report_text
        if "get_latest_clip" in notif_params:
            notif_kwargs["get_latest_clip"] = self.archive.get_latest_mp4
        if "on_false_alert_cmd" in notif_params:
            notif_kwargs["on_false_alert_cmd"] = self.report_false_alert
        if "webapp_url" in notif_params:
            notif_kwargs["webapp_url"] = self.settings.public_webapp_url
        self.notifier = TelegramNotifier(**notif_kwargs)  # type: ignore[arg-type]

    def _load_events_locked(self) -> None:
        if not self._events_path.exists():
            return
        try:
            raw = json.loads(self._events_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Не удалось прочитать журнал событий %s: %s", self._events_path, exc)
            return
        items = raw.get("events") if isinstance(raw, dict) else (raw if isinstance(raw, list) else None)
        if not isinstance(items, list):
            return
        for item in items[: self.MAX_EVENTS_LOG]:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            lvl = str(item.get("level") or "info")
            cat = str(item.get("category") or ("alert" if lvl in ("warning", "critical") else "system"))
            self._events_log.append(
                {
                    "id": str(item.get("id") or secrets.token_hex(4)),
                    "ts": float(item.get("ts") or 0.0),
                    "date": str(item.get("date") or ""),
                    "time": str(item.get("time") or ""),
                    "level": lvl,
                    "category": cat,
                    "state": item.get("state"),
                    "text": str(item.get("text")),
                    "clip_id": item.get("clip_id"),
                }
            )

    def _save_events_locked(self) -> None:
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_ts": round(time.time(), 3),
                "events": list(self._events_log),
            }
            tmp_path = self._events_path.with_name(
                f"{self._events_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
            )
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._events_path)
        except OSError as exc:
            logger.warning("Не удалось сохранить журнал событий %s: %s", self._events_path, exc)

    def start(self) -> None:
        threading.Thread(target=self.faces.ensure_models, name="face-init", daemon=True).start()
        if self.settings.camera_configured:
            self.capture.start()
            if getattr(self.settings, "stream_h264_enabled", True):
                self.stream_hub.start()
        if self.settings.telegram_configured:
            self.notifier.start_polling()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name="infer-loop", daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.stream_hub.stop()
        self.capture.stop()
        self.notifier.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        with self._lock:
            subs = list(self._telemetry_subs)
            self._telemetry_subs.clear()
        for sub in subs:
            sub.close()
        self.timeline.flush()
        with self._lock:
            self._save_events_locked()

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_latest_clean_jpeg(self) -> bytes | None:
        with self._lock:
            frame = self._latest_raw_frame
            if frame is None:
                return self._latest_jpeg
            fid = id(frame)
            if self._latest_clean_jpeg is not None and self._latest_clean_frame_id == fid:
                return self._latest_clean_jpeg
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            with self._lock:
                return self._latest_jpeg
        clean_bytes = enc.tobytes()
        with self._lock:
            if self._latest_raw_frame is frame:
                self._latest_clean_jpeg = clean_bytes
                self._latest_clean_frame_id = fid
        return clean_bytes

    def subscribe_telemetry(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> TelemetrySubscription:
        ev_loop = loop if loop is not None else asyncio.get_running_loop()
        sub = TelemetrySubscription(self, ev_loop, maxsize=4)
        with self._lock:
            self._telemetry_subs.add(sub)
        return sub

    def unsubscribe_telemetry(self, sub: TelemetrySubscription) -> None:
        with self._lock:
            self._telemetry_subs.discard(sub)

    def publish_telemetry(self) -> None:
        with self._lock:
            if not self._telemetry_subs:
                return
            subs = list(self._telemetry_subs)
        snap = self.get_telemetry_snapshot()
        for sub in subs:
            sub.push_threadsafe(snap)

    def get_human_status_line(self) -> str:
        st = self.fsm.state
        st_val = str(getattr(st, "value", st))
        label = HUMAN_STATE_RU.get(st_val, st_val)
        if st == PatientState.PAUSED:
            rem = int(self.fsm.pause_remaining_sec(time.monotonic()))
            if rem > 0:
                label = f"Охрана на паузе ({rem // 60} мин {rem % 60:02d} с)"
        cam_ok = bool(self.get_camera_transform().get("connected", self.capture.is_connected))
        cam_str = "онлайн" if cam_ok else "ожидание потока"
        return f"Статус: {label} · Камера: {cam_str}"

    def get_prefs(self) -> dict[str, Any]:
        return self.prefs.get()

    def update_prefs(self, patch: dict[str, Any]) -> dict[str, Any]:
        updated = self.prefs.update(patch)
        if hasattr(self.fsm, "update_runtime_prefs"):
            self.fsm.update_runtime_prefs(**updated)  # type: ignore[attr-defined]
        else:
            with self._lock:
                if hasattr(self.fsm, "_sit_confirm"):
                    self.fsm._sit_confirm = float(updated["sit_confirm_seconds"])  # type: ignore[attr-defined]
                if hasattr(self.fsm, "_leave_confirm"):
                    self.fsm._leave_confirm = float(updated["leave_confirm_seconds"])  # type: ignore[attr-defined]
                if hasattr(self.fsm, "_fall_confirm"):
                    self.fsm._fall_confirm = float(updated["fall_confirm_seconds"])  # type: ignore[attr-defined]
                if hasattr(self.fsm, "_missing_confirm"):
                    self.fsm._missing_confirm = float(updated["missing_confirm_seconds"])  # type: ignore[attr-defined]
        preset_ru = {
            "high": "высокая",
            "standard": "стандарт",
            "gentle": "мягкая",
            "custom": "своя",
        }.get(str(updated.get("sensitivity_preset")), "стандарт")
        self._log_event("info", f"Настройки ухода обновлены (чувствительность: {preset_ru})", category="system")
        return updated

    def get_care_diagnostics(self, now_mono: float | None = None) -> dict[str, Any]:
        ts_mono = time.monotonic() if now_mono is None else now_mono
        if hasattr(self.fsm, "get_care_diagnostics"):
            try:
                return dict(self.fsm.get_care_diagnostics(now=ts_mono))  # type: ignore[attr-defined]
            except TypeError:
                return dict(self.fsm.get_care_diagnostics())  # type: ignore[attr-defined]
        prefs_cur = self.prefs.get()
        hour = datetime.now().hour
        ns, ne = int(prefs_cur["night_start_hour"]), int(prefs_cur["night_end_hour"])
        night_win = (hour >= ns or hour < ne) if ns > ne else (ns <= hour < ne)
        night_active = bool(prefs_cur["night_strict_mode"] and night_win)
        return {
            "baseline": getattr(self.fsm, "baseline_in_bed", {"calibrated": False, "samples": 0}),
            "night_active": night_active,
            "effective_sit_confirm_sec": round(
                max(1.0, float(prefs_cur["sit_confirm_seconds"]) * 0.72)
                if night_active
                else float(prefs_cur["sit_confirm_seconds"]),
                2,
            ),
            "effective_leave_confirm_sec": round(
                max(1.1, float(prefs_cur["leave_confirm_seconds"]) * 0.75)
                if night_active
                else float(prefs_cur["leave_confirm_seconds"]),
                2,
            ),
            "immobility_minutes": None,
            "away_from_bed_minutes": None,
            "last_movement_sec_ago": None,
        }

    def recalibrate_baseline(self) -> dict[str, Any]:
        if hasattr(self.fsm, "reset_baseline"):
            self.fsm.reset_baseline()  # type: ignore[attr-defined]
        elif hasattr(self.fsm, "_baseline"):
            self.fsm._baseline.reset()  # type: ignore[attr-defined]
        self._log_event(
            "info",
            "Эталон позы сна сброшен — выполнится новая калибровка в кровати",
            category="system",
        )
        return self.get_care_diagnostics()

    def reset_appearance_anchor(self) -> dict[str, Any]:
        self.clear_patient_anchor()
        self._log_event("info", "Суточный якорь внешности сброшен", category="system")
        return {
            "has_appearance_anchor": bool(getattr(self.identifier, "has_appearance_anchor", False))
        }

    def get_care_report_text(self) -> str:
        stats = self.timeline.summary_stats(window_sec=86400.0)
        diag = self.get_care_diagnostics()
        clips_info = self.archive.storage_stats()
        st_line = self.get_human_status_line()

        lines = [
            "Сводка ухода FallGuard за 24 часа",
            st_line,
            f"• Спокойно в кровати: {_fmt_ru_duration(stats['in_bed_sec'])}",
            f"• Посадок в кровати: {stats['sit_up_count']}",
            f"• Выходов из кровати: {stats['left_bed_count']}",
            f"• Вызовов рукой: {stats['help_call_count']}",
            f"• Падений: {stats['fall_count']}",
            f"• Визитов близких: {stats['caregiver_visits_count']} ({_fmt_ru_duration(stats['caregiver_sec'])})",
        ]
        imm_min = diag.get("immobility_minutes")
        if imm_min is not None:
            lines.append(f"• Без смены позы в кровати: {int(round(float(imm_min)))} мин")
        mov_sec = diag.get("last_movement_sec_ago")
        if mov_sec is not None:
            lines.append(f"• Последнее движение: {_fmt_ru_duration(float(mov_sec))} назад")
        lines.append(
            f"• Записей в видеоархиве: {clips_info['count']} ({clips_info['mp4_count']} MP4, {clips_info['used_mb']} МБ)"
        )
        return "\n".join(lines)

    def get_telegram_qr_svg(self) -> str:
        uname = (self.notifier.bot_username or "NoFallenAi").lstrip("@")
        target_url = f"https://t.me/{uname}?start=watch"
        return render_qr_svg(target_url, border=2)

    def get_camera_transform(self) -> dict[str, object]:
        if hasattr(self.capture, "get_transform"):
            raw = dict(self.capture.get_transform())  # type: ignore[attr-defined]
        else:
            with self._lock:
                raw = dict(self._fallback_cam_state)
        raw.setdefault("rotate", self.settings.frame_rotate)
        raw.setdefault("flip_h", self.settings.frame_flip_h)
        raw.setdefault("enhance", self.settings.low_light_enhance)
        raw.setdefault("resolution", [0, 0])
        fresh_sec = float(getattr(self.capture, "seconds_since_last_frame", 999.0))
        raw["connected"] = bool(self.capture.is_connected or fresh_sec < 15.0)
        return raw

    def set_camera_transform(
        self,
        rotate: int | None = None,
        flip_h: bool | None = None,
        enhance: bool | None = None,
    ) -> dict[str, object]:
        prev_cam = self.get_camera_transform()
        if hasattr(self.capture, "set_transform"):
            res = dict(
                self.capture.set_transform(  # type: ignore[attr-defined]
                    rotate=rotate,
                    flip_h=flip_h,
                    enhance=enhance,
                )
            )
        else:
            with self._lock:
                if rotate is not None:
                    self._fallback_cam_state["rotate"] = rotate
                if flip_h is not None:
                    self._fallback_cam_state["flip_h"] = flip_h
                if enhance is not None:
                    self._fallback_cam_state["enhance"] = enhance
                res = dict(self._fallback_cam_state)
        if (
            res.get("rotate") != prev_cam.get("rotate")
            or res.get("flip_h") != prev_cam.get("flip_h")
        ):
            if hasattr(self.pose, "reset_smoother"):
                self.pose.reset_smoother()  # type: ignore[attr-defined]
            self._reset_telemetry_latch()
            self.camera_health.reset_reference()
        fresh_sec = float(getattr(self.capture, "seconds_since_last_frame", 999.0))
        res["connected"] = bool(self.capture.is_connected or fresh_sec < 15.0)
        rot_lbl = f"{res.get('rotate', 0)}°"
        flip_lbl = "зеркало вкл" if res.get("flip_h") else "зеркало выкл"
        enh_lbl = "контраст вкл" if res.get("enhance") else "контраст выкл"
        self._log_event("info", f"Камера: {rot_lbl}, {flip_lbl}, {enh_lbl}", category="system")
        self.publish_telemetry()
        return res

    def _reset_telemetry_latch(self, track_id: int | None = None) -> None:
        with self._lock:
            if track_id is None:
                self._telemetry_latch.clear()
                self._prev_opt_gray = None
            else:
                self._telemetry_latch.pop(track_id, None)

    def _sync_camera_health_zones(self) -> None:
        if hasattr(self.camera_health, "set_zones"):
            self.camera_health.set_zones(self.zones.get())

    def on_zones_updated(self) -> None:
        # Разметку перерисовали — обновляем маску неподвижного фона и сбрасываем эталон
        self._sync_camera_health_zones()
        self.camera_health.reset_reference()
        with self._lock:
            self._reset_telemetry_latch()
            self._sync_tracks_summary_locked()
        self.publish_telemetry()

    def clear_patient_anchor(self) -> None:
        self._reset_telemetry_latch()
        with self._lock:
            self._last_latched_bed_bbox = None
        if hasattr(self.identifier, "clear_anchor"):
            self.identifier.clear_anchor()  # type: ignore[attr-defined]
        else:
            anchor_file = self.settings.data_dir / "patient_anchor.json"
            if anchor_file.exists():
                try:
                    anchor_file.unlink()
                except OSError:
                    pass
        with self._lock:
            self._sync_tracks_summary_locked()
        self.publish_telemetry()

    IGNORE_CONFLICT_DETAIL: str = (
        "Нельзя добавить игнор-зону поверх кровати или двери — используйте кнопку «Свой»"
    )

    def ignore_track(self, track_id: int) -> tuple[bool, ZonesConfig]:
        bbox: tuple[float, float, float, float] | None = None
        obs_found: PoseObservation | None = None
        role_found: Role | None = None
        with self._lock:
            for obs in self._latest_observations:
                if obs.track_id == track_id:
                    obs_found = obs
                    bbox = obs.bbox_norm
                    break
            ident_lat = self._latest_identities.get(track_id)
            if ident_lat is not None:
                role_found = ident_lat.role
            for tr_item in self._active_tracks_summary:
                if tr_item.get("track_id") == track_id:
                    if role_found is None and str(tr_item.get("role") or "") == Role.PATIENT.value:
                        role_found = Role.PATIENT
                    if bbox is None and isinstance(tr_item.get("bbox"), list):
                        b = tr_item["bbox"]
                        if len(b) == 4:
                            bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                    break

        ident_tracks = getattr(self.identifier, "_tracks", None)
        if role_found is None and isinstance(ident_tracks, dict) and track_id in ident_tracks:
            role_found = ident_tracks[track_id].role

        cur_zones = self.zones.get()
        if bbox is None:
            return False, cur_zones

        pad = 0.02
        x1 = max(0.0, round(bbox[0] - pad, 5))
        y1 = max(0.0, round(bbox[1] - pad, 5))
        x2 = min(1.0, round(bbox[2] + pad, 5))
        y2 = min(1.0, round(bbox[3] + pad, 5))
        cand_bbox = (x1, y1, x2, y2)
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        if role_found == Role.PATIENT:
            raise ValueError(self.IGNORE_CONFLICT_DETAIL)
        if cur_zones.has_bed:
            on_bed_flag = obs_found is not None and is_person_on_bed(
                obs_found, cur_zones.bed_polygon, zones=cur_zones
            )
            box_in_bed = bbox_polygon_overlap_ratio(cand_bbox, cur_zones.bed_polygon, grid_n=12)
            bed_cov = polygon_coverage_ratio(poly, cur_zones.bed_polygon, grid_n=24)
            if on_bed_flag or box_in_bed >= 0.12 or bed_cov >= 0.12:
                raise ValueError(self.IGNORE_CONFLICT_DETAIL)
        if cur_zones.has_door:
            box_in_door = bbox_polygon_overlap_ratio(cand_bbox, cur_zones.door_polygon, grid_n=12)
            door_cov = polygon_coverage_ratio(poly, cur_zones.door_polygon, grid_n=24)
            if box_in_door >= 0.25 or door_cov >= 0.25:
                raise ValueError(self.IGNORE_CONFLICT_DETAIL)

        updated_zones = self.zones.add_ignore_polygon(poly)
        if hasattr(self.identifier, "rollback_or_clear_polluted_anchor"):
            with contextlib.suppress(Exception):
                self.identifier.rollback_or_clear_polluted_anchor(track_id)  # type: ignore[attr-defined]
        if isinstance(ident_tracks, dict):
            ident_lock = getattr(self.identifier, "_lock", self._lock)
            with ident_lock:
                ident_tracks.pop(track_id, None)

        with self._lock:
            self._telemetry_latch.pop(track_id, None)
            self._latest_identities.pop(track_id, None)
            self._latest_observations = [
                o for o in self._latest_observations if o.track_id != track_id
            ]
            self._active_tracks_summary = [
                t for t in self._active_tracks_summary if t.get("track_id") != track_id
            ]
            self._sync_tracks_summary_locked()

        if not self._is_patient_in_bed_latched():
            if hasattr(self.fsm, "reset_to_idle"):
                self.fsm.reset_to_idle()  # type: ignore[attr-defined]
            elif self.fsm.state not in (PatientState.PAUSED, PatientState.IN_BED):
                rem = self.fsm.pause_remaining_sec(time.monotonic())
                if rem <= 0:
                    self.fsm.acknowledge_and_pause(time.monotonic(), 0.0)

        if hasattr(self.pose, "reset_smoother"):
            self.pose.reset_smoother()  # type: ignore[attr-defined]

        self._log_event("info", "Добавлена зона игнорирования предмета", category="system")
        self.publish_telemetry()
        return True, updated_zones

    def clear_ignore_zones(self) -> ZonesConfig:
        updated = self.zones.clear_ignore()
        self._reset_telemetry_latch()
        self._log_event("info", "Зоны игнорирования очищены", category="system")
        self.publish_telemetry()
        return updated

    def _fallback_force_assign_visitor(self, track_id: int) -> bool:
        ident_lock = getattr(self.identifier, "_lock", self._lock)
        tracks_map = getattr(self.identifier, "_tracks", None)
        if not isinstance(tracks_map, dict):
            return False
        with ident_lock:
            target = tracks_map.get(track_id)
            if target is None:
                return False
            target.role = Role.VISITOR
            target.reason = "manual_visitor_override"
            target.bed_accum_sec = 0.0
            if getattr(self.identifier, "_last_patient_track_id", None) == track_id:
                self.identifier._last_patient_track_id = None  # type: ignore[attr-defined]
            return True

    def _resolve_detector_still_hint(
        self,
        obs: PoseObservation,
        obs_clean: PoseObservation,
        optical_motion_ratio: float | None,
    ) -> bool | None:
        if bool(getattr(obs, "_kp_stabilized", False)):
            return bool(getattr(obs, "is_still", False))
        if bool(getattr(obs_clean, "_kp_stabilized", False)):
            return bool(getattr(obs_clean, "is_still", False))
        if bool(getattr(obs, "is_still", False)) or bool(getattr(obs_clean, "is_still", False)):
            return True
        for opt_cand in (
            getattr(obs, "optical_motion_ratio", None),
            getattr(obs_clean, "optical_motion_ratio", None),
        ):
            if isinstance(opt_cand, (int, float)):
                return bool(float(opt_cand) < OPTICAL_MOTION_STILL_RATIO)
        if optical_motion_ratio is not None:
            return bool(optical_motion_ratio < OPTICAL_MOTION_STILL_RATIO)
        return None

    def _build_track_ui_item(
        self,
        obs: PoseObservation,
        ident: TrackIdentity | None,
        zones: ZonesConfig,
        state: PatientState,
        optical_motion_ratio: float | None = None,
        cur_opt_gray: np.ndarray | None = None,
        now_mono: float | None = None,
        other_observations: Sequence[PoseObservation] = (),
    ) -> dict[str, Any]:
        ts_mono = time.monotonic() if now_mono is None else now_mono
        obs_clean = (
            _sanitize_bed_obs_compat(
                obs,
                zones.bed_polygon,
                other_observations=other_observations,
                zones=zones,
            )
            if zones.has_bed
            else obs
        )
        role_val = ident.role.value if ident else Role.UNKNOWN.value
        reason_code = ident.reason if ident else ""
        bed_sec = round(ident.bed_accum_sec, 1) if ident else 0.0

        # Отсекаем шумовые под-одеяльные точки (conf < 0.48 -> [0.0, 0.0, 0.0]) ДО округления
        raw_bbox = (
            float(obs_clean.bbox_norm[0]),
            float(obs_clean.bbox_norm[1]),
            float(obs_clean.bbox_norm[2]),
            float(obs_clean.bbox_norm[3]),
        )
        raw_kps: list[tuple[float, float, float]] = []
        active_mask: list[bool] = []
        valid_draw_pts: list[tuple[float, float]] = []
        for k in range(17):
            if k < len(obs_clean.keypoints_norm):
                kx, ky, kc = (
                    float(obs_clean.keypoints_norm[k][0]),
                    float(obs_clean.keypoints_norm[k][1]),
                    float(obs_clean.keypoints_norm[k][2]),
                )
                if kc >= KP_TELEMETRY_MIN_CONF and not (kx <= 0.0 and ky <= 0.0):
                    raw_kps.append((kx, ky, kc))
                    active_mask.append(True)
                    valid_draw_pts.append((kx, ky))
                else:
                    raw_kps.append((0.0, 0.0, 0.0))
                    active_mask.append(False)
            else:
                raw_kps.append((0.0, 0.0, 0.0))
                active_mask.append(False)

        if len(valid_draw_pts) >= 2:
            span_x = max(pt[0] for pt in valid_draw_pts) - min(pt[0] for pt in valid_draw_pts)
            span_y = max(pt[1] for pt in valid_draw_pts) - min(pt[1] for pt in valid_draw_pts)
            skeleton_span = max(span_x, span_y)
        else:
            skeleton_span = 0.0

        has_confident_hip = active_mask[11] or active_mask[12]
        on_bed_now = bool(
            (zones.has_bed and is_person_on_bed(obs_clean, zones.bed_polygon, zones=zones))
            or reason_code in ("bed_sleep_lock", "sole_bed_sleeper_recovery")
        )
        calm_state = state in (
            PatientState.IN_BED,
            PatientState.CAREGIVER_ASSIST,
            PatientState.PAUSED,
            PatientState.IDLE,
        )
        hide_skeleton = bool(
            skeleton_span < 0.11
            or (
                role_val == Role.PATIENT.value
                and calm_state
                and (on_bed_now or state == PatientState.IN_BED)
                and not has_confident_hip
            )
        )
        height_cm, floor_xy_cm = _compute_track_25d_metrics(obs_clean, zones)
        angle_deg = round(obs_clean.torso_angle_from_vertical_deg() or 0.0, 1)
        in_bed_sleep = state == PatientState.IN_BED or reason_code in (
            "bed_sleep_lock",
            "sole_bed_sleeper_recovery",
        )
        active_motion_state = state in (
            PatientState.SITTING_UP,
            PatientState.LEFT_BED,
            PatientState.FALLEN,
            PatientState.HELP_CALL,
        )
        still_hint = self._resolve_detector_still_hint(obs, obs_clean, optical_motion_ratio)

        with self._lock:
            stale_ids = [
                tid
                for tid, st_rec in self._telemetry_latch.items()
                if (ts_mono - st_rec.last_seen_mono)
                > (MAX_SLEEP_COAST_SEC if (st_rec.role_val == Role.PATIENT.value and st_rec.on_bed) else 4.0)
            ]
            for tid in stale_ids:
                self._telemetry_latch.pop(tid, None)

            entry = self._telemetry_latch.get(obs_clean.track_id)
            if (
                entry is None
                and in_bed_sleep
                and on_bed_now
                and role_val == Role.PATIENT.value
                and not active_motion_state
                and still_hint is not False
            ):
                # Бесшовная склейка замороженных координат сна при перескоке track_id (ByteTrack <-> ROI)
                for prev_tid, prev_rec in list(self._telemetry_latch.items()):
                    if prev_rec.role_val == Role.PATIENT.value and prev_rec.on_bed:
                        if max(abs(raw_bbox[i] - prev_rec.anchor_bbox[i]) for i in range(4)) <= 0.045:
                            self._telemetry_latch.pop(prev_tid, None)
                            prev_rec.track_id = obs_clean.track_id
                            self._telemetry_latch[obs_clean.track_id] = prev_rec
                            entry = prev_rec
                            break

            occluded_now = bool(
                on_bed_now
                and other_observations
                and any(
                    bbox_iou(raw_bbox, o.bbox_norm) >= 0.08
                    or (entry is not None and bbox_iou(entry.anchor_bbox, o.bbox_norm) >= 0.08)
                    for o in other_observations
                )
            )

            if entry is not None:
                entry.last_seen_mono = ts_mono
                entry.last_confirmed_mono = ts_mono
                entry.role_val = role_val
                entry.reason_code = reason_code
                entry.bed_sec = bed_sec
                entry.on_bed = on_bed_now

                bbox_delta = max(abs(raw_bbox[i] - entry.anchor_bbox[i]) for i in range(4))
                occlusion_shift = (occluded_now != entry.occluded_by_other) or (
                    occluded_now and bbox_delta > 0.020
                )
                entry.occluded_by_other = occluded_now
                kp_deltas: list[float] = []
                for k in range(17):
                    if entry.active_mask[k] and raw_kps[k][2] >= 0.50:
                        kp_deltas.append(
                            math.hypot(
                                raw_kps[k][0] - entry.anchor_kps[k][0],
                                raw_kps[k][1] - entry.anchor_kps[k][1],
                            )
                        )
                common_kp_delta = max(kp_deltas, default=0.0)
                emerge_thresh = 0.60 if in_bed_sleep else 0.56
                sudden_limb_emergence = any(
                    (not entry.active_mask[k]) and raw_kps[k][2] >= emerge_thresh
                    for k in range(11)
                )

                if active_motion_state:
                    is_still = False
                elif still_hint is False:
                    is_still = False
                elif still_hint is True:
                    is_still = not (
                        bbox_delta > 0.042
                        or common_kp_delta > 0.045
                        or sudden_limb_emergence
                    )
                else:
                    deadzone_b = 0.026 if in_bed_sleep else 0.016
                    deadzone_k = 0.028 if in_bed_sleep else 0.018
                    is_still = not (
                        bbox_delta > deadzone_b
                        or common_kp_delta > deadzone_k
                        or sudden_limb_emergence
                    )

                if is_still and (occlusion_shift or (hide_skeleton != entry.frozen_hide_skeleton)):
                    entry.anchor_bbox = raw_bbox
                    entry.frozen_bbox = [round(v, 4) for v in raw_bbox]
                    entry.anchor_kps = list(raw_kps)
                    entry.active_mask = list(active_mask)
                    entry.frozen_kps = [
                        [round(kx, 4), round(ky, 4), round(kc, 3)]
                        if kc >= KP_TELEMETRY_MIN_CONF
                        else [0.0, 0.0, 0.0]
                        for (kx, ky, kc) in raw_kps
                    ]
                    entry.frozen_hide_skeleton = hide_skeleton

                if is_still:
                    entry.was_still = True
                    if entry.anchor_opt_gray is None and cur_opt_gray is not None:
                        entry.anchor_opt_gray = cur_opt_gray
                    if role_val == Role.PATIENT.value and on_bed_now:
                        self._last_latched_bed_bbox = list(entry.frozen_bbox)
                        self._last_latched_patient_tid = int(obs_clean.track_id)
                    elif role_val == Role.VISITOR.value:
                        self._last_latched_visitor_tid = int(obs_clean.track_id)
                    return {
                        "track_id": obs_clean.track_id,
                        "role": role_val,
                        "reason": reason_code,
                        "reason_ru": translate_role_reason(role_val, reason_code),
                        "bed_sec": bed_sec,
                        "angle_deg": entry.frozen_angle_deg,
                        "is_still": True,
                        "bbox": list(entry.frozen_bbox),
                        "keypoints": [list(pt) for pt in entry.frozen_kps],
                        "hide_skeleton": entry.frozen_hide_skeleton,
                        "height_cm": entry.frozen_height_cm,
                        "floor_xy_cm": (
                            list(entry.frozen_floor_xy_cm)
                            if entry.frozen_floor_xy_cm is not None
                            else None
                        ),
                    }
            else:
                is_still = bool(
                    not active_motion_state
                    and (still_hint is True or (in_bed_sleep and bed_sec >= 1.5))
                )

            # Округление координат до 4 знаков строго ПОСЛЕ фильтрации и стабилизации
            rounded_bbox = [round(v, 4) for v in raw_bbox]
            rounded_kps = [
                [round(kx, 4), round(ky, 4), round(kc, 3)] if kc >= KP_TELEMETRY_MIN_CONF else [0.0, 0.0, 0.0]
                for (kx, ky, kc) in raw_kps
            ]
            if role_val == Role.PATIENT.value and on_bed_now:
                self._last_latched_bed_bbox = list(rounded_bbox)
                self._last_latched_patient_tid = int(obs_clean.track_id)
            elif role_val == Role.VISITOR.value:
                self._last_latched_visitor_tid = int(obs_clean.track_id)
            self._telemetry_latch[obs_clean.track_id] = _TrackTelemetryState(
                track_id=obs_clean.track_id,
                last_seen_mono=ts_mono,
                anchor_bbox=raw_bbox,
                anchor_kps=list(raw_kps),
                active_mask=list(active_mask),
                frozen_bbox=list(rounded_bbox),
                frozen_kps=[list(pt) for pt in rounded_kps],
                frozen_angle_deg=angle_deg,
                frozen_hide_skeleton=hide_skeleton,
                frozen_height_cm=height_cm,
                frozen_floor_xy_cm=list(floor_xy_cm) if floor_xy_cm is not None else None,
                was_still=is_still,
                role_val=role_val,
                reason_code=reason_code,
                bed_sec=bed_sec,
                on_bed=on_bed_now,
                occluded_by_other=occluded_now,
                last_confirmed_mono=ts_mono,
                anchor_opt_gray=cur_opt_gray,
            )

        return {
            "track_id": obs_clean.track_id,
            "role": role_val,
            "reason": reason_code,
            "reason_ru": translate_role_reason(role_val, reason_code),
            "bed_sec": bed_sec,
            "angle_deg": angle_deg,
            "is_still": is_still,
            "bbox": rounded_bbox,
            "keypoints": rounded_kps,
            "hide_skeleton": hide_skeleton,
            "height_cm": height_cm,
            "floor_xy_cm": floor_xy_cm,
        }

    def _is_patient_in_bed_latched(self) -> bool:
        return bool(getattr(self.identifier, "patient_in_bed_latched", False))

    def _resolve_latched_bed_bbox(self, zones: ZonesConfig) -> list[float] | None:
        raw = getattr(self.identifier, "latched_patient_bed_bbox", None)
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return [round(float(v), 4) for v in raw]
        if self._last_latched_bed_bbox is not None and len(self._last_latched_bed_bbox) == 4:
            return list(self._last_latched_bed_bbox)
        if zones.has_bed and len(zones.bed_polygon) >= 3:
            xs = [float(p[0]) for p in zones.bed_polygon]
            ys = [float(p[1]) for p in zones.bed_polygon]
            return [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]
        return None

    def _resolve_latched_patient_tid(self) -> int:
        raw_tid = getattr(self.identifier, "latched_patient_track_id", None)
        if isinstance(raw_tid, int) and raw_tid > 0:
            return int(raw_tid)
        if self._last_latched_patient_tid > 0:
            return int(self._last_latched_patient_tid)
        return 1

    def _resolve_latched_visitor_bbox(self) -> list[float] | None:
        raw = getattr(self.identifier, "latched_visitor_bbox", None)
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return [round(float(v), 4) for v in raw]
        return None

    def _resolve_latched_visitor_tid(self) -> int:
        raw_tid = getattr(self.identifier, "latched_visitor_track_id", None)
        if isinstance(raw_tid, int) and raw_tid != 0:
            return int(raw_tid)
        return -2

    def _apply_presence_persistence_locked(
        self,
        tracks_ui: list[dict[str, Any]],
        state_now: PatientState,
        zones: ZonesConfig,
    ) -> list[dict[str, Any]]:
        """
        Обеспечивает постоянство карточек «Бабушка» и «Свой» в «В кадре» и /ws/telemetry:
        1) При patient_in_bed_latched == True (вне LEFT_BED / MISSING / FALLEN), если нет живого или
           оптически защёлкнутого трека бабушки (например, после переворота под одеялом / паузы),
           возвращает спокойную карточку bed_covered_persistence без устаревших костей.
        2) При door_visit_active == True, если нет живого трека гостя, возвращает door_visit_seated
           (со стулом latched_visitor_bbox при visitor_spot_occupied) или door_visit_present (track_id: -2).
        """
        out = [
            dict(t)
            for t in tracks_ui
            if str(t.get("reason") or "") not in SYNTHETIC_PERSISTENCE_REASONS
        ]
        patient_latched = self._is_patient_in_bed_latched()
        allow_bed_persistence = bool(
            patient_latched
            and state_now
            not in (PatientState.LEFT_BED, PatientState.MISSING, PatientState.FALLEN)
        )
        if allow_bed_persistence and not any(
            t.get("role") == Role.PATIENT.value for t in out
        ):
            bed_bbox = self._resolve_latched_bed_bbox(zones)
            pat_tid = self._resolve_latched_patient_tid()
            bed_card: dict[str, Any] = {
                "track_id": pat_tid,
                "role": Role.PATIENT.value,
                "reason": "bed_covered_persistence",
                "reason_ru": translate_role_reason(
                    Role.PATIENT.value, "bed_covered_persistence"
                ),
                "bed_sec": 0.0,
                "angle_deg": 90.0,
                "is_still": True,
                "bbox": bed_bbox,
                "keypoints": [],
                "hide_skeleton": True,
                "height_cm": None,
                "floor_xy_cm": None,
                "virtual_presence": bed_bbox is None,
            }
            out = [bed_card, *out]

        door_visit_active = bool(getattr(self.identifier, "door_visit_active", False))
        if door_visit_active and not any(
            t.get("role") == Role.VISITOR.value for t in out
        ):
            spot_occupied = bool(getattr(self.identifier, "visitor_spot_occupied", False))
            vis_bbox = self._resolve_latched_visitor_bbox()
            if spot_occupied and vis_bbox is not None:
                overlaps_patient_card = any(
                    t.get("role") == Role.PATIENT.value
                    and isinstance(t.get("bbox"), (list, tuple))
                    and len(t["bbox"]) == 4
                    and bbox_iou(
                        (
                            float(vis_bbox[0]),
                            float(vis_bbox[1]),
                            float(vis_bbox[2]),
                            float(vis_bbox[3]),
                        ),
                        (
                            float(t["bbox"][0]),
                            float(t["bbox"][1]),
                            float(t["bbox"][2]),
                            float(t["bbox"][3]),
                        ),
                    )
                    >= 0.20
                    for t in out
                )
                if overlaps_patient_card:
                    spot_occupied = False
            if spot_occupied and vis_bbox is not None:
                vis_card: dict[str, Any] = {
                    "track_id": self._resolve_latched_visitor_tid(),
                    "role": Role.VISITOR.value,
                    "reason": "door_visit_seated",
                    "reason_ru": translate_role_reason(
                        Role.VISITOR.value, "door_visit_seated"
                    ),
                    "bed_sec": 0.0,
                    "angle_deg": 0.0,
                    "is_still": True,
                    "bbox": vis_bbox,
                    "keypoints": [],
                    "hide_skeleton": True,
                    "height_cm": None,
                    "floor_xy_cm": None,
                    "virtual_presence": False,
                }
            else:
                vis_card = {
                    "track_id": -2,
                    "role": Role.VISITOR.value,
                    "reason": "door_visit_present",
                    "reason_ru": translate_role_reason(
                        Role.VISITOR.value, "door_visit_present"
                    ),
                    "bed_sec": 0.0,
                    "angle_deg": 0.0,
                    "is_still": True,
                    "bbox": None,
                    "keypoints": [],
                    "hide_skeleton": True,
                    "height_cm": None,
                    "floor_xy_cm": None,
                    "virtual_presence": True,
                }
            out.append(vis_card)

        return out

    def _build_tracks_ui_for_frame(
        self,
        live_observations: list[PoseObservation],
        identities: dict[int, TrackIdentity],
        zones: ZonesConfig,
        state_now: PatientState,
        cur_opt_gray: np.ndarray | None = None,
        prev_opt_gray: np.ndarray | None = None,
        now_mono: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Формирует список tracks для /api/status и /ws/telemetry:
        удерживает неподвижный спальный трек из _telemetry_latch при optical_motion < 0.012 и anchor_drift < 0.050,
        а также гарантирует постоянство карточек Бабушки в кровати и Своих в комнате.
        """
        ts_mono = time.monotonic() if now_mono is None else now_mono
        tracks_ui = [
            self._build_track_ui_item(
                obs,
                identities.get(obs.track_id),
                zones,
                state_now,
                optical_motion_ratio=_compute_roi_optical_motion(
                    cur_opt_gray,
                    prev_opt_gray,
                    obs.bbox_norm,
                    exclude_bboxes=[
                        o.bbox_norm for o in live_observations if o.track_id != obs.track_id
                    ],
                ),
                cur_opt_gray=cur_opt_gray,
                now_mono=ts_mono,
                other_observations=[o for o in live_observations if o.track_id != obs.track_id],
            )
            for obs in live_observations
        ]

        with self._lock:
            patient_latched = self._is_patient_in_bed_latched()
            can_coast_sleep = (
                state_now not in (PatientState.LEFT_BED, PatientState.MISSING, PatientState.FALLEN)
                and (state_now == PatientState.IN_BED or patient_latched)
                and not any(t.get("role") == Role.PATIENT.value for t in tracks_ui)
            )
            if can_coast_sleep:
                best_entry: _TrackTelemetryState | None = None
                for rec in self._telemetry_latch.values():
                    if rec.role_val != Role.PATIENT.value or not rec.on_bed:
                        continue
                    if (ts_mono - rec.last_confirmed_mono) > MAX_SLEEP_COAST_SEC:
                        continue
                    if best_entry is None or rec.last_confirmed_mono > best_entry.last_confirmed_mono:
                        best_entry = rec

                if best_entry is not None:
                    step_motion = _compute_roi_optical_motion(
                        cur_opt_gray,
                        prev_opt_gray,
                        best_entry.anchor_bbox,
                    )
                    anchor_drift = _compute_roi_anchor_drift(
                        cur_opt_gray,
                        best_entry.anchor_opt_gray,
                        best_entry.anchor_bbox,
                    )
                    if (step_motion is not None and step_motion >= OPTICAL_MOTION_STILL_RATIO) or (
                        anchor_drift is not None and anchor_drift >= OPTICAL_ANCHOR_DRIFT_RATIO
                    ):
                        # Скелет в старой позе устарел (одеяло сдвинулось / переворот) — сбрасываем кости,
                        # при patient_in_bed_latched далее подставится карточка bed_covered_persistence.
                        self._telemetry_latch.pop(best_entry.track_id, None)
                    else:
                        best_entry.last_seen_mono = ts_mono
                        best_entry.was_still = True
                        if best_entry.anchor_opt_gray is None and cur_opt_gray is not None:
                            best_entry.anchor_opt_gray = cur_opt_gray
                        coasted_bed_sec = round(
                            best_entry.bed_sec + max(0.0, ts_mono - best_entry.last_confirmed_mono),
                            1,
                        )
                        coasted_item = {
                            "track_id": best_entry.track_id,
                            "role": best_entry.role_val,
                            "reason": best_entry.reason_code,
                            "reason_ru": translate_role_reason(best_entry.role_val, best_entry.reason_code),
                            "bed_sec": coasted_bed_sec,
                            "angle_deg": best_entry.frozen_angle_deg,
                            "is_still": True,
                            "bbox": list(best_entry.frozen_bbox),
                            "keypoints": [list(pt) for pt in best_entry.frozen_kps],
                            "hide_skeleton": best_entry.frozen_hide_skeleton,
                            "height_cm": best_entry.frozen_height_cm,
                            "floor_xy_cm": (
                                list(best_entry.frozen_floor_xy_cm)
                                if best_entry.frozen_floor_xy_cm is not None
                                else None
                            ),
                        }
                        tracks_ui = [coasted_item, *tracks_ui]

            return self._apply_presence_persistence_locked(tracks_ui, state_now, zones)

    def _sync_tracks_summary_locked(self) -> None:
        ident_tracks = getattr(self.identifier, "_tracks", None)
        if isinstance(ident_tracks, dict):
            for tid in list(self._latest_identities.keys()):
                if tid in ident_tracks:
                    self._latest_identities[tid] = ident_tracks[tid]
            for obs in self._latest_observations:
                if obs.track_id in ident_tracks:
                    self._latest_identities[obs.track_id] = ident_tracks[obs.track_id]

        zones_now = self.zones.get()
        state_now = self.fsm.state
        if self._latest_observations:
            self._active_tracks_summary = self._build_tracks_ui_for_frame(
                live_observations=self._latest_observations,
                identities=self._latest_identities,
                zones=zones_now,
                state_now=state_now,
                cur_opt_gray=self._prev_opt_gray,
                prev_opt_gray=self._prev_opt_gray,
            )
        else:
            existing_non_synthetic = [
                dict(it)
                for it in self._active_tracks_summary
                if str(it.get("reason") or "") not in SYNTHETIC_PERSISTENCE_REASONS
            ]
            if isinstance(ident_tracks, dict) and existing_non_synthetic:
                for item in existing_non_synthetic:
                    tid = item.get("track_id")
                    if isinstance(tid, int) and tid in ident_tracks:
                        ident_obj = ident_tracks[tid]
                        item["role"] = ident_obj.role.value
                        item["reason"] = ident_obj.reason
                        item["reason_ru"] = translate_role_reason(ident_obj.role.value, ident_obj.reason)
                        item["bed_sec"] = round(ident_obj.bed_accum_sec, 1)
                self._active_tracks_summary = self._apply_presence_persistence_locked(
                    existing_non_synthetic,
                    state_now,
                    zones_now,
                )
            else:
                self._active_tracks_summary = self._build_tracks_ui_for_frame(
                    live_observations=[],
                    identities={},
                    zones=zones_now,
                    state_now=state_now,
                    cur_opt_gray=self._prev_opt_gray,
                    prev_opt_gray=self._prev_opt_gray,
                )

    def assign_track_role(self, track_id: int, role: Role) -> dict[str, Any]:
        if role not in (Role.PATIENT, Role.VISITOR):
            return {"ok": False, "detail": "Роль должна быть patient или visitor"}

        prev_role: Role | None = None
        with self._lock:
            prev_ident = self._latest_identities.get(track_id)
            if prev_ident is not None:
                prev_role = prev_ident.role
        if prev_role is None:
            ident_tracks = getattr(self.identifier, "_tracks", None)
            if isinstance(ident_tracks, dict) and track_id in ident_tracks:
                prev_role = ident_tracks[track_id].role

        if role == Role.PATIENT:
            ok = bool(self.identifier.force_assign_patient(track_id))
            if ok and hasattr(self.identifier, "clear_door_visit"):
                self.identifier.clear_door_visit()  # type: ignore[attr-defined]
        else:
            if hasattr(self.identifier, "force_assign_visitor"):
                ok = bool(self.identifier.force_assign_visitor(track_id))  # type: ignore[attr-defined]
            else:
                ok = self._fallback_force_assign_visitor(track_id)

        if not ok:
            return {"ok": False, "detail": "Трек не найден в текущем кадре"}

        with self._lock:
            self._sync_tracks_summary_locked()
            if role == Role.VISITOR:
                self._caregiver_nearby = True
            any_patient_left = any(
                ident.role == Role.PATIENT for ident in self._latest_identities.values()
            )
        if not any_patient_left:
            ident_tracks = getattr(self.identifier, "_tracks", None)
            if isinstance(ident_tracks, dict):
                any_patient_left = any(tr.role == Role.PATIENT for tr in ident_tracks.values())

        if role == Role.VISITOR:
            alarm_states = (
                PatientState.SITTING_UP,
                PatientState.LEFT_BED,
                PatientState.MISSING,
                PatientState.FALLEN,
                PatientState.HELP_CALL,
            )
            caused_by_this_track = prev_role == Role.PATIENT or not any_patient_left
            if caused_by_this_track and (
                self.fsm.state in alarm_states or self.fsm.fall_unacknowledged
            ):
                if hasattr(self.fsm, "reset_to_idle"):
                    self.fsm.reset_to_idle()  # type: ignore[attr-defined]

        ok_face, face_info = self.enroll_face_from_camera(role, track_id=track_id)
        role_label = "Бабушка" if role == Role.PATIENT else "Свой"
        detail_suffix = "одежда и лицо сохранены" if ok_face else "одежда запомнена"
        msg = f"Закреплено: {role_label} ({detail_suffix})"
        self._log_event("info", msg, category="system")
        self.publish_telemetry()

        return {
            "ok": True,
            "track_id": track_id,
            "role": role.value,
            "auto_face_saved": bool(ok_face),
            "face_filename": face_info if ok_face else None,
            "face_detail": None if ok_face else face_info,
            "state": self.fsm.state.value,
            "message": msg,
        }

    def enroll_face_from_camera(self, role: Role, track_id: int | None = None) -> tuple[bool, str]:
        frame, _ = self.capture.get_latest()
        with self._lock:
            if frame is None and self._latest_raw_frame is not None:
                frame = self._latest_raw_frame.copy()
            observations = list(self._latest_observations)
            identities = dict(self._latest_identities)

        if frame is None:
            return False, "Камера ещё не передала живой видеокадр"

        target_obs: PoseObservation | None = None
        if track_id is not None:
            for obs in observations:
                if obs.track_id == track_id:
                    target_obs = obs
                    break
            if target_obs is None and len(observations) > 1:
                return False, "Силуэт трека вне текущего кадра для съёмки лица"
        else:
            for obs in observations:
                ident = identities.get(obs.track_id)
                if ident and ident.role == role:
                    target_obs = obs
                    break
            if target_obs is None and role == Role.VISITOR:
                for obs in observations:
                    ident = identities.get(obs.track_id)
                    if not ident or ident.role != Role.PATIENT:
                        target_obs = obs
                        break
            if target_obs is None and len(observations) == 1:
                target_obs = observations[0]

        # При явном указании track_id среди нескольких людей ограничиваем поиск кропом вокруг этого трека,
        # чтобы не захватить чужое лицо второго человека на другом краю комнаты, если целевой стоит спиной.
        enroll_frame = frame
        enroll_obs = target_obs
        if track_id is not None and target_obs is not None and len(observations) > 1:
            h, w = frame.shape[:2]
            bx1, by1, bx2, by2 = target_obs.bbox_norm
            bw = max(0.05, bx2 - bx1)
            bh = max(0.05, by2 - by1)
            cx1 = int(max(0, (bx1 - bw * 0.24) * w))
            cx2 = int(min(w, (bx2 + bw * 0.24) * w))
            cy1 = int(max(0, (by1 - bh * 0.20) * h))
            cy2 = int(min(h, (by2 + bh * 0.16) * h))
            if (cx2 - cx1) >= 40 and (cy2 - cy1) >= 40:
                enroll_frame = frame[cy1:cy2, cx1:cx2]
                enroll_obs = None

        if hasattr(self.faces, "enroll_from_frame"):
            enroll_fn = getattr(self.faces, "enroll_from_frame")
            ok, info = enroll_fn(role, enroll_frame, obs=enroll_obs)
        else:
            ok, info = self._fallback_enroll_from_frame(role, enroll_frame, enroll_obs)

        if ok:
            role_ru = "бабушка" if role == Role.PATIENT else "свой"
            self._log_event("info", f"Сохранён эталон лица ({role_ru})", category="system")
        return ok, info

    def _fallback_enroll_from_frame(
        self,
        role: Role,
        frame_bgr: np.ndarray,
        obs: PoseObservation | None,
    ) -> tuple[bool, str]:
        candidates: list[np.ndarray] = []
        if obs is not None:
            h, w = frame_bgr.shape[:2]
            bx1, by1, bx2, by2 = obs.bbox_norm
            pad_x = (bx2 - bx1) * 0.22
            pad_y = (by2 - by1) * 0.16
            x1 = int(max(0, (bx1 - pad_x) * w))
            x2 = int(min(w, (bx2 + pad_x) * w))
            y1 = int(max(0, (by1 - pad_y) * h))
            y2 = int(min(h, (by1 + (by2 - by1) * 0.72) * h))
            if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                candidates.append(frame_bgr[y1:y2, x1:x2])
        candidates.append(frame_bgr)

        last_err = "Лицо на кадре не обнаружено — повернитесь к камере"
        for crop in candidates:
            ok_enc, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if not ok_enc:
                continue
            ok_add, info = self.faces.add_photo(role, f"cam_{role.value}.jpg", enc.tobytes())
            if ok_add:
                return True, info
            last_err = info
        return False, last_err

    def list_subscribers_normalized(self) -> list[dict[str, Any]]:
        raw_subs = self.notifier.list_subscribers()
        out: list[dict[str, Any]] = []
        for s in raw_subs:
            if isinstance(s, dict):
                out.append({**s, "enabled": bool(s.get("enabled", True))})
        return out

    def toggle_subscriber_enabled(
        self,
        chat_id: str,
        enabled: bool | None = None,
    ) -> dict[str, Any] | None:
        cid = str(chat_id).strip()
        if not cid:
            return None

        subs_before = self.list_subscribers_normalized()
        existing = next((s for s in subs_before if str(s.get("chat_id", "")) == cid), None)
        target_enabled = (
            bool(enabled)
            if enabled is not None
            else (not bool(existing.get("enabled", True)) if existing is not None else False)
        )

        if hasattr(self.notifier, "set_subscriber_enabled"):
            updated = self.notifier.set_subscriber_enabled(cid, target_enabled)  # type: ignore[attr-defined]
            if updated is None:
                return None
            return {**dict(updated), "enabled": bool(updated.get("enabled", target_enabled))}

        if existing is None:
            return None
        subs_map = getattr(self.notifier, "_subscribers", None)
        notif_lock = getattr(self.notifier, "_lock", self._lock)
        if isinstance(subs_map, dict):
            with notif_lock:
                rec = subs_map.get(cid)
                if not isinstance(rec, dict):
                    rec = dict(existing)
                    subs_map[cid] = rec
                rec["enabled"] = target_enabled
                if hasattr(self.notifier, "_save_subscribers_locked"):
                    self.notifier._save_subscribers_locked()  # type: ignore[attr-defined]
                return {**dict(rec), "enabled": target_enabled}
        return None

    def _resolve_stream_resolution(self, cam: dict[str, object]) -> list[int]:
        hub_res = self.stream_hub.resolution
        if isinstance(hub_res, list) and len(hub_res) == 2 and int(hub_res[0]) > 0 and int(hub_res[1]) > 0:
            return [int(hub_res[0]), int(hub_res[1])]
        cam_res = cam.get("resolution")
        if isinstance(cam_res, list) and len(cam_res) == 2 and int(cam_res[0]) > 0 and int(cam_res[1]) > 0:
            return [int(cam_res[0]), int(cam_res[1])]
        with self._lock:
            if self._latest_raw_frame is not None and self._latest_raw_frame.ndim >= 2:
                h, w = self._latest_raw_frame.shape[:2]
                return [int(w), int(h)]
        return [0, 0]

    def get_telemetry_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        z = self.zones.get()
        cam = self.get_camera_transform()
        decoder_backend = str(getattr(self.capture, "decoder_backend", "CPU (FFmpeg)"))
        encoder_backend = self.recorder.encoder_backend
        door_visit_active = bool(getattr(self.identifier, "door_visit_active", False))
        patient_in_bed_latched = self._is_patient_in_bed_latched()
        homography_cal = _is_homography_calibrated(z)
        stream_mode = "ws_fmp4" if self.stream_hub.is_ready else "mjpeg"
        stream_clients = int(self.stream_hub.stream_clients)
        resolution = self._resolve_stream_resolution(cam)

        with self._lock:
            is_night_ir = _detect_night_ir_frame(self._latest_raw_frame, self.identifier)
            self._is_night_ir = is_night_ir
            anchor_day_ready, anchor_night_ready = _extract_anchor_flags(self.identifier, is_night_ir)
            tracks_snapshot = self._apply_presence_persistence_locked(
                self._active_tracks_summary,
                self.fsm.state,
                z,
            )
            caregiver_flag = bool(self._caregiver_nearby or door_visit_active)
            fps_val = round(self._actual_fps, 1)

        return {
            "type": "telemetry",
            "ts": round(time.time(), 3),
            "state": self.fsm.state.value,
            "status_reason": getattr(self.fsm, "status_reason", ""),
            "fall_unacknowledged": self.fsm.fall_unacknowledged,
            "pause_remaining_sec": round(self.fsm.pause_remaining_sec(now), 1),
            "caregiver_nearby": caregiver_flag,
            "door_visit_active": door_visit_active,
            "patient_in_bed_latched": patient_in_bed_latched,
            "is_night_ir": is_night_ir,
            "anchor_day_ready": anchor_day_ready,
            "anchor_night_ready": anchor_night_ready,
            "homography_calibrated": homography_cal,
            "infer_fps": fps_val,
            "runtime": self.pose.runtime_label,
            "decoder_backend": decoder_backend,
            "encoder_backend": encoder_backend,
            "stream_mode": stream_mode,
            "stream_clients": stream_clients,
            "resolution": resolution,
            "camera_issue": self.camera_health.issue,
            "camera_health": self.camera_health.snapshot(),
            "camera_rotate": int(cam.get("rotate", self.settings.frame_rotate)),  # type: ignore[arg-type]
            "camera_flip_h": bool(cam.get("flip_h", self.settings.frame_flip_h)),
            "tracks": tracks_snapshot,
        }

    def get_status(self, timeline_window_sec: float = 21600.0) -> dict[str, Any]:
        now = time.monotonic()
        z = self.zones.get()
        uname = self.notifier.bot_username
        subs = self.list_subscribers_normalized()
        cam = self.get_camera_transform()
        prefs_cur = self.prefs.get()
        diag = self.get_care_diagnostics(now_mono=now)
        summary_24h = self.timeline.summary_stats(window_sec=86400.0)
        clips_stats = self.archive.storage_stats()
        feedback_stats = self.feedback.stats()
        recent_clips = self.archive.list_clips(limit=40)
        tl_dict = self.timeline.to_dict(view_window_sec=timeline_window_sec)
        thumbs_list = (
            self.thumbs.list_index(since_ts=float(tl_dict["start_ts"]))
            if self.thumbs is not None
            else []
        )
        decoder_backend = str(getattr(self.capture, "decoder_backend", "CPU (FFmpeg)"))
        encoder_backend = self.recorder.encoder_backend
        door_visit_active = bool(getattr(self.identifier, "door_visit_active", False))
        patient_in_bed_latched = self._is_patient_in_bed_latched()
        homography_cal = _is_homography_calibrated(z)
        stream_mode = "ws_fmp4" if self.stream_hub.is_ready else "mjpeg"
        stream_clients = int(self.stream_hub.stream_clients)

        with self._lock:
            is_night_ir = _detect_night_ir_frame(self._latest_raw_frame, self.identifier)
            self._is_night_ir = is_night_ir
            anchor_day_ready, anchor_night_ready = _extract_anchor_flags(self.identifier, is_night_ir)
            has_any_anchor = bool(
                getattr(self.identifier, "has_appearance_anchor", False)
                or anchor_day_ready
                or anchor_night_ready
            )
            tracks_snapshot = self._apply_presence_persistence_locked(
                self._active_tracks_summary,
                self.fsm.state,
                z,
            )
            return {
                "state": self.fsm.state.value,
                "status_reason": getattr(self.fsm, "status_reason", ""),
                "fall_unacknowledged": self.fsm.fall_unacknowledged,
                "pause_remaining_sec": round(self.fsm.pause_remaining_sec(now), 1),
                "camera_configured": self.settings.camera_configured,
                "camera_connected": bool(cam.get("connected", self.capture.is_connected)),
                "camera_error": self.capture.last_error,
                "camera_issue": self.camera_health.issue,
                "camera_health": self.camera_health.snapshot(),
                "camera": cam,
                "decoder_backend": decoder_backend,
                "encoder_backend": encoder_backend,
                "last_encode_ms": round(self.recorder.last_encode_ms, 2),
                "stream_mode": stream_mode,
                "stream_clients": stream_clients,
                "telegram_configured": self.settings.telegram_configured,
                "telegram_bot_username": uname,
                "telegram_bot_url": f"https://t.me/{uname}",
                "telegram_subscribers": subs,
                "telegram_recipients_count": len(self.notifier.get_recipient_chat_ids()),
                "devices": self.devices.list(),
                "public_webapp_url": self.settings.public_webapp_url,
                "missing_env_keys": self.settings.missing_required_keys(),
                "runtime": self.pose.runtime_label,
                "infer_fps": round(self._actual_fps, 1),
                "bed_zone_ready": z.has_bed,
                "door_zone_ready": z.has_door,
                "floor_zone_ready": z.has_floor,
                "homography_calibrated": homography_cal,
                "ignore_zones_count": len(z.ignore_polygons),
                "zones": z.to_dict(),
                "caregiver_nearby": bool(self._caregiver_nearby or door_visit_active),
                "door_visit_active": door_visit_active,
                "patient_in_bed_latched": patient_in_bed_latched,
                "is_night_ir": is_night_ir,
                "has_appearance_anchor": has_any_anchor,
                "anchor_day_ready": anchor_day_ready,
                "anchor_night_ready": anchor_night_ready,
                "tracks": tracks_snapshot,
                "events": list(self._events_log),
                "faces": self.faces.list_photos(),
                "timeline": tl_dict,
                "thumbs": thumbs_list,
                "prefs": prefs_cur,
                "care": {
                    "diagnostics": diag,
                    "summary_24h": summary_24h,
                },
                "clips_stats": clips_stats,
                "clip_pre_sec": float(self.settings.pre_event_seconds),
                # Часы сервера: клиенты считают «сколько прошло» от них, а не от своих (часы телефона могут уйти)
                "server_ts": round(time.time(), 3),
                "feedback_stats": feedback_stats,
                "recent_clips": recent_clips,
            }

    def clear_door_visit(self) -> dict[str, Any]:
        if hasattr(self.identifier, "clear_door_visit"):
            self.identifier.clear_door_visit()  # type: ignore[attr-defined]
        if bool(getattr(self.identifier, "door_visit_active", False)):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(self.identifier, "door_visit_active", False)
        with self._lock:
            for tid in [
                k for k, rec in self._telemetry_latch.items() if rec.role_val == Role.VISITOR.value
            ]:
                self._telemetry_latch.pop(tid, None)
            has_live_visitor = any(
                ident.role == Role.VISITOR for ident in self._latest_identities.values()
            )
            if not has_live_visitor:
                self._caregiver_nearby = False
            self._sync_tracks_summary_locked()
        self._log_event("info", "Визит гостя завершён вручную («Вышел»)", category="system")
        self.publish_telemetry()
        return {
            "ok": True,
            "door_visit_active": bool(getattr(self.identifier, "door_visit_active", False)),
        }

    def clear_bed_presence(self) -> dict[str, Any]:
        if hasattr(self.identifier, "clear_bed_presence"):
            self.identifier.clear_bed_presence()  # type: ignore[attr-defined]
        else:
            for priv_attr, priv_val in (
                ("_patient_in_bed_latched", False),
                ("_latched_patient_bed_bbox", None),
                ("_ever_locked_patient", False),
            ):
                if hasattr(self.identifier, priv_attr):
                    with contextlib.suppress(AttributeError, TypeError):
                        setattr(self.identifier, priv_attr, priv_val)
        if bool(getattr(self.identifier, "patient_in_bed_latched", False)):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(self.identifier, "patient_in_bed_latched", False)

        with self._lock:
            self._last_latched_bed_bbox = None
            for tid in [
                k for k, rec in self._telemetry_latch.items() if rec.role_val == Role.PATIENT.value
            ]:
                self._telemetry_latch.pop(tid, None)
            has_live_patient = any(
                obs.track_id in self._latest_identities
                and self._latest_identities[obs.track_id].role == Role.PATIENT
                for obs in self._latest_observations
            )
            if not has_live_patient and hasattr(self.fsm, "reset_to_idle"):
                self.fsm.reset_to_idle()  # type: ignore[attr-defined]
            self._sync_tracks_summary_locked()

        self._log_event(
            "info",
            "Отметка нахождения в кровати сброшена вручную («Не в кровати»)",
            category="system",
        )
        self.publish_telemetry()
        return {
            "ok": True,
            "patient_in_bed_latched": bool(self._is_patient_in_bed_latched()),
        }

    def pause(self, seconds: float) -> None:
        self.fsm.acknowledge_and_pause(time.monotonic(), seconds)
        with self._lock:
            self._sync_tracks_summary_locked()
        cam_ok = bool(self.get_camera_transform().get("connected", self.capture.is_connected))
        self.timeline.record(self.fsm.state.value, cam_ok)
        self._log_event("info", f"Пауза охраны ({int(seconds // 60)} мин)", category="system")
        self.publish_telemetry()

    def resume(self) -> None:
        if hasattr(self.identifier, "clear_door_visit"):
            self.identifier.clear_door_visit()  # type: ignore[attr-defined]
        self.fsm.resume()
        with self._lock:
            has_live_visitor = any(
                ident.role == Role.VISITOR for ident in self._latest_identities.values()
            )
            if not has_live_visitor and not bool(getattr(self.identifier, "door_visit_active", False)):
                self._caregiver_nearby = False
            self._sync_tracks_summary_locked()
        cam_ok = bool(self.get_camera_transform().get("connected", self.capture.is_connected))
        self.timeline.record(self.fsm.state.value, cam_ok)
        self._log_event("info", "Охрана включена", category="system")
        self.publish_telemetry()

    def log_event(
        self,
        level: str,
        text: str,
        category: str | None = None,
        clip_id: str | None = None,
        state: str | None = None,
    ) -> None:
        """Публичная запись в журнал событий (для веб-слоя)."""
        self._log_event(level, text, category=category, clip_id=clip_id, state=state)

    def _log_event(
        self,
        level: str,
        text: str,
        category: str | None = None,
        clip_id: str | None = None,
        state: str | None = None,
    ) -> None:
        now_epoch = time.time()
        dt = datetime.fromtimestamp(now_epoch)
        cat = category or ("alert" if level in ("warning", "critical") else "system")
        entry: dict[str, Any] = {
            "id": secrets.token_hex(4),
            "ts": round(now_epoch, 3),
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "level": level,
            "category": cat,
            "state": state,
            "text": text,
            "clip_id": clip_id,
        }
        with self._lock:
            self._events_log.appendleft(entry)
            self._save_events_locked()

    def _build_placeholder_jpeg(self, message: str) -> bytes:
        img = np.zeros((540, 960, 3), dtype=np.uint8)
        img[:] = (16, 13, 11)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        font = _load_font(19)
        draw.text((40, 255), message, fill=(170, 180, 195), font=font)
        out_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return enc.tobytes() if ok else b""

    @staticmethod
    def _reconstruct_obs_from_track_dict(item: dict[str, Any]) -> PoseObservation | None:
        if not isinstance(item, dict):
            return None
        reason = str(item.get("reason") or "")
        if reason in SYNTHETIC_PERSISTENCE_REASONS or bool(item.get("virtual_presence")):
            return None
        raw_bbox = item.get("bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return None
        try:
            bx1, by1, bx2, by2 = (
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
            raw_tid = item.get("track_id", 0)
            tid = int(raw_tid) if isinstance(raw_tid, (int, float)) else 0
        except (TypeError, ValueError):
            return None
        if (bx2 - bx1) <= 0.01 or (by2 - by1) <= 0.01:
            return None

        raw_kps = item.get("keypoints")
        if not isinstance(raw_kps, (list, tuple)) or len(raw_kps) == 0:
            return None
        kps: list[tuple[float, float, float]] = []
        max_conf = 0.0
        for idx in range(17):
            if idx < len(raw_kps) and isinstance(raw_kps[idx], (list, tuple)) and len(raw_kps[idx]) >= 3:
                try:
                    kx = float(raw_kps[idx][0])
                    ky = float(raw_kps[idx][1])
                    kc = float(raw_kps[idx][2])
                except (TypeError, ValueError):
                    kx, ky, kc = 0.0, 0.0, 0.0
                max_conf = max(max_conf, kc)
                kps.append((kx, ky, kc))
            else:
                kps.append((0.0, 0.0, 0.0))
        return PoseObservation(
            track_id=tid,
            bbox_norm=(bx1, by1, bx2, by2),
            keypoints_norm=tuple(kps),
            conf=max(0.5, min(1.0, max_conf)),
        )

    @staticmethod
    def _safe_latched_bed_bbox_for_enroll(
        candidate_bbox: tuple[float, float, float, float] | None,
        culprit_obs: PoseObservation,
        zones: ZonesConfig,
    ) -> tuple[float, float, float, float] | None:
        if candidate_bbox is None or len(candidate_bbox) != 4:
            return None
        if bbox_iou(candidate_bbox, culprit_obs.bbox_norm) >= 0.45:
            return None
        if zones.has_bed and bbox_polygon_overlap_ratio(candidate_bbox, zones.bed_polygon) < 0.15:
            return None
        return candidate_bbox

    def report_false_alert(
        self,
        alert_id: str,
        reporter_chat_id: str = "",
        reporter_name: str = "Пользователь",
    ) -> dict[str, Any]:
        """
        Сохраняет диагностический пакет ложного срабатывания в FalseAlertFeedbackStore,
        помечает запись в ClipArchiveStore и выполняет самокоррекцию живого движка без 15-минутной паузы.
        """
        clean_id = str(alert_id or "").strip()
        if not is_safe_alert_id(clean_id):
            raise ValueError("Недопустимый идентификатор события")
        clean_reporter = str(reporter_name or "").strip() or "Пользователь"

        recent_info = self.feedback.get_recent_alert(clean_id)
        fallback_clip = self.archive.get_clip(clean_id)
        fallback_jpg: bytes | None = None
        mp4_bytes: bytes | None = None
        if recent_info is None or not recent_info.get("has_annotated_jpg"):
            jpg_path = self.archive.resolve_media_path(clean_id, "jpg")
            if jpg_path is not None:
                with contextlib.suppress(OSError):
                    fallback_jpg = jpg_path.read_bytes()
        if recent_info is None or not recent_info.get("has_mp4"):
            mp4_path = self.archive.resolve_media_path(clean_id, "mp4")
            if mp4_path is not None:
                with contextlib.suppress(OSError):
                    mp4_bytes = mp4_path.read_bytes()

        zones_now = self.zones.get()
        live_context: dict[str, Any] = {
            "state": self.fsm.state.value,
            "telemetry_snapshot": self.get_telemetry_snapshot(),
            "baseline_in_bed": dict(getattr(self.fsm, "baseline_in_bed", {})),
            "zones": zones_now.to_dict(),
            "was_bed_latched": self._is_patient_in_bed_latched(),
        }
        fb_entry = self.feedback.mark_false_alert(
            alert_id=clean_id,
            reporter_chat_id=reporter_chat_id,
            reporter_name=clean_reporter,
            fallback_clip=fallback_clip,
            mp4_bytes=mp4_bytes,
            fallback_jpg=fallback_jpg,
            live_context=live_context,
        )
        clip_entry = self.archive.mark_false_alarm(clean_id, clean_reporter)

        alert_state = str(
            (recent_info.get("state") if recent_info else None)
            or (fallback_clip.get("state") if fallback_clip else None)
            or fb_entry.get("state")
            or self.fsm.state.value
        ).strip().lower()
        raw_pat_tid = (
            recent_info.get("patient_track_id")
            if recent_info is not None
            else fb_entry.get("patient_track_id")
        )
        alert_patient_tid = int(raw_pat_tid) if isinstance(raw_pat_tid, int) else None
        raw_ev_ts = fb_entry.get("event_ts") or (fallback_clip.get("ts") if fallback_clip else None)
        event_wall_ts = (
            float(raw_ev_ts)
            if isinstance(raw_ev_ts, (int, float)) and float(raw_ev_ts) > 0.0
            else time.time()
        )
        is_recent_alert = (abs(time.time() - event_wall_ts) <= 900.0) or (
            self.fsm.state
            in (
                PatientState.SITTING_UP,
                PatientState.LEFT_BED,
                PatientState.MISSING,
                PatientState.FALLEN,
                PatientState.HELP_CALL,
            )
            or self.fsm.fall_unacknowledged
        )

        # Авто-пополнение галереи «Свой» (data/faces/visitor/) по сырому кадру тревоги
        visitor_face_saved = False
        visitor_face_file: str = ""
        visitor_face_status = "not_attempted"

        raw_alert_jpeg, alert_ctx = self.feedback.get_alert_raw_jpeg_and_context(clean_id)
        if raw_alert_jpeg and hasattr(self.faces, "auto_enroll_visitor_from_false_alert"):
            arr_alert = np.frombuffer(raw_alert_jpeg, dtype=np.uint8)
            alert_frame_bgr = cv2.imdecode(arr_alert, cv2.IMREAD_COLOR)
            snap_dict = alert_ctx.get("telemetry_snapshot")
            snap_tracks = snap_dict.get("tracks") if isinstance(snap_dict, dict) else None
            if alert_frame_bgr is not None and isinstance(snap_tracks, list) and snap_tracks:
                ctx_zones_raw = alert_ctx.get("zones")
                zones_alert = zones_now
                if isinstance(ctx_zones_raw, dict) and isinstance(ctx_zones_raw.get("bed_polygon"), list):
                    with contextlib.suppress(Exception):
                        zones_alert = ZonesConfig(
                            bed_polygon=list(ctx_zones_raw.get("bed_polygon") or zones_now.bed_polygon),
                            door_polygon=list(ctx_zones_raw.get("door_polygon") or zones_now.door_polygon),
                            ignore_polygons=list(
                                ctx_zones_raw.get("ignore_polygons") or zones_now.ignore_polygons
                            ),
                            floor_polygon=list(ctx_zones_raw.get("floor_polygon") or zones_now.floor_polygon),
                            floor_width_cm=float(
                                ctx_zones_raw.get("floor_width_cm") or zones_now.floor_width_cm
                            ),
                            floor_depth_cm=float(
                                ctx_zones_raw.get("floor_depth_cm") or zones_now.floor_depth_cm
                            ),
                            mattress_height_cm=float(
                                ctx_zones_raw.get("mattress_height_cm") or zones_now.mattress_height_cm
                            ),
                        )
                reconstructed: list[tuple[dict[str, Any], PoseObservation]] = []
                persisted_bed_bbox: tuple[float, float, float, float] | None = None
                for t_item in snap_tracks:
                    if not isinstance(t_item, dict):
                        continue
                    if (
                        str(t_item.get("reason") or "") == "bed_covered_persistence"
                        and isinstance(t_item.get("bbox"), (list, tuple))
                        and len(t_item["bbox"]) == 4
                    ):
                        with contextlib.suppress(TypeError, ValueError):
                            b = t_item["bbox"]
                            persisted_bed_bbox = (
                                float(b[0]),
                                float(b[1]),
                                float(b[2]),
                                float(b[3]),
                            )
                    obs_rec = self._reconstruct_obs_from_track_dict(t_item)
                    if obs_rec is not None:
                        reconstructed.append((t_item, obs_rec))

                culprit_obs: PoseObservation | None = None
                if alert_patient_tid is not None:
                    culprit_obs = next(
                        (o for _, o in reconstructed if o.track_id == alert_patient_tid),
                        None,
                    )
                if culprit_obs is None:
                    culprit_obs = next(
                        (
                            o
                            for t_item, o in reconstructed
                            if str(t_item.get("role") or "") == Role.PATIENT.value
                        ),
                        None,
                    )
                if culprit_obs is not None:
                    other_alert_obs = [
                        o for _, o in reconstructed if o.track_id != culprit_obs.track_id
                    ]
                    raw_id_bed = getattr(self.identifier, "latched_patient_bed_bbox", None)
                    cand_bed_box: tuple[float, float, float, float] | None = persisted_bed_bbox
                    if cand_bed_box is None and isinstance(raw_id_bed, (list, tuple)) and len(raw_id_bed) == 4:
                        cand_bed_box = (
                            float(raw_id_bed[0]),
                            float(raw_id_bed[1]),
                            float(raw_id_bed[2]),
                            float(raw_id_bed[3]),
                        )
                    safe_bed_box = self._safe_latched_bed_bbox_for_enroll(
                        cand_bed_box,
                        culprit_obs,
                        zones_alert,
                    )
                    visitor_face_saved, visitor_face_file, visitor_face_status = (
                        self.faces.auto_enroll_visitor_from_false_alert(
                            frame_bgr=alert_frame_bgr,
                            obs=culprit_obs,
                            zones=zones_alert,
                            other_observations=other_alert_obs,
                            latched_patient_bed_bbox=safe_bed_box,
                            alert_id=clean_id,
                        )
                    )

        reassigned_tid: int | None = None
        if is_recent_alert:
            with self._lock:
                obs_snapshot = list(self._latest_observations)
                ident_init = dict(self._latest_identities)
                live_raw_frame = self._latest_raw_frame

            ident_tracks_init = getattr(self.identifier, "_tracks", None)
            if isinstance(ident_tracks_init, dict):
                for tid_k, tr_v in ident_tracks_init.items():
                    ident_init[tid_k] = tr_v

            cand_tid = alert_patient_tid
            if cand_tid is None:
                for o in obs_snapshot:
                    tr_o = ident_init.get(o.track_id)
                    if tr_o is not None and tr_o.role == Role.PATIENT:
                        cand_tid = o.track_id
                        break

            live_target_obs = (
                next((o for o in obs_snapshot if o.track_id == cand_tid), None)
                if cand_tid is not None
                else None
            )
            if (
                not visitor_face_saved
                and visitor_face_status != "already_known"
                and live_raw_frame is not None
                and live_target_obs is not None
                and hasattr(self.faces, "auto_enroll_visitor_from_false_alert")
            ):
                other_live_obs = [
                    o for o in obs_snapshot if o.track_id != live_target_obs.track_id
                ]
                raw_live_bed = getattr(self.identifier, "latched_patient_bed_bbox", None)
                cand_live_bed = (
                    (
                        float(raw_live_bed[0]),
                        float(raw_live_bed[1]),
                        float(raw_live_bed[2]),
                        float(raw_live_bed[3]),
                    )
                    if isinstance(raw_live_bed, (list, tuple)) and len(raw_live_bed) == 4
                    else None
                )
                safe_live_bed = self._safe_latched_bed_bbox_for_enroll(
                    cand_live_bed,
                    live_target_obs,
                    zones_now,
                )
                fb_saved, fb_file, fb_status = self.faces.auto_enroll_visitor_from_false_alert(
                    frame_bgr=live_raw_frame,
                    obs=live_target_obs,
                    zones=zones_now,
                    other_observations=other_live_obs,
                    latched_patient_bed_bbox=safe_live_bed,
                    alert_id=clean_id,
                )
                if fb_saved or fb_status == "already_known" or visitor_face_status == "not_attempted":
                    visitor_face_saved = fb_saved
                    visitor_face_file = fb_file
                    visitor_face_status = fb_status

            # 1) Самоисцеление при ложной тревоге («Ты ошибся») для всех состояний (включая sitting_up и fallen):
            # если виновный трек находится вне кровати ИЛИ стоит/сидит у кровати без истории сна (< 3 с),
            # переводим его в Role.VISITOR, откатываем загрязнённый якорь и повышаем реального человека на кровати!
            # Исключение: единственная подтверждённая бабушка, сидящая на кровати (не разжалуем в «Своего»).
            was_bed_latched_ctx = bool(
                (alert_ctx or {}).get("was_bed_latched", False)
                if isinstance(alert_ctx, dict)
                else False
            )
            should_demote = False
            target_obs: PoseObservation | None = None
            target_tr: TrackIdentity | None = None
            if cand_tid is not None and zones_now.has_door:
                target_obs = next(
                    (o for o in obs_snapshot if o.track_id == cand_tid),
                    None,
                )
                target_tr = ident_init.get(cand_tid)
                if target_obs is not None:
                    on_bed_now = (
                        is_person_on_bed(target_obs, zones_now.bed_polygon, zones=zones_now)
                        if zones_now.has_bed
                        else False
                    )
                    if not on_bed_now:
                        should_demote = True
                    else:
                        if (
                            target_tr is not None
                            and hasattr(self.identifier, "_is_sleeping_or_reclining_on_bed")
                        ):
                            reclining_now = bool(
                                self.identifier._is_sleeping_or_reclining_on_bed(  # type: ignore[attr-defined]
                                    target_obs,
                                    target_tr,
                                    zones_now,
                                )
                            )
                        else:
                            ang_now = target_obs.torso_angle_from_vertical_deg()
                            reclining_now = (ang_now is None or ang_now >= 28.0) and not (
                                zones_now.has_bed
                                and are_legs_hanging_outside_bed(
                                    target_obs,
                                    zones_now.bed_polygon,
                                    zones=zones_now,
                                )
                            )
                        bed_accum = (
                            float(getattr(target_tr, "bed_accum_sec", 0.0))
                            if target_tr is not None
                            else 0.0
                        )
                        if not reclining_now and bed_accum < 3.0:
                            should_demote = True

                    # Защита от разжалования единственной бабушки в «Своего» при ложной тревоге вставания:
                    # если в кадре 1 человек с подтверждённой личностью / историей кровати (не стоящий у двери/перед кроватью)
                    if should_demote and len(obs_snapshot) == 1 and target_tr is not None:
                        tr_reason = str(getattr(target_tr, "reason", "") or "")
                        tr_was_on_bed = bool(getattr(target_tr, "was_on_bed", False))
                        tr_bed_accum = float(getattr(target_tr, "bed_accum_sec", 0.0))
                        touching_door_now = is_pose_touching_door(
                            target_obs,
                            zones_now.door_polygon,
                            min_overlap=0.15,
                        )
                        standing_front_now = (
                            zones_now.has_bed
                            and is_standing_in_front_of_bed(
                                target_obs,
                                zones_now.bed_polygon,
                                zones=zones_now,
                            )
                        )
                        tall_stander_now = bool(
                            hasattr(self.identifier, "_is_tall_floor_stander")
                            and self.identifier._is_tall_floor_stander(  # type: ignore[attr-defined]
                                target_obs,
                                target_tr,
                                zones_now,
                            )
                        )
                        confirmed_sole_grandma = (
                            on_bed_now
                            and (
                                tr_reason == "face_patient"
                                or tr_was_on_bed
                                or tr_bed_accum >= 1.0
                                or not was_bed_latched_ctx
                            )
                        ) or (
                            not was_bed_latched_ctx
                            and (tr_reason == "face_patient" or tr_was_on_bed or tr_bed_accum >= 3.0)
                        )
                        if (
                            confirmed_sole_grandma
                            and not touching_door_now
                            and not standing_front_now
                            and not tall_stander_now
                        ):
                            should_demote = False
                elif target_tr is not None and not bool(getattr(target_tr, "was_on_bed", False)):
                    if float(getattr(target_tr, "bed_accum_sec", 0.0)) < 1.0:
                        should_demote = True

                if should_demote:
                    if hasattr(self.identifier, "rollback_or_clear_polluted_anchor"):
                        with contextlib.suppress(Exception):
                            self.identifier.rollback_or_clear_polluted_anchor(cand_tid)  # type: ignore[attr-defined]
                    if hasattr(self.identifier, "force_assign_visitor"):
                        ok_vis = bool(self.identifier.force_assign_visitor(cand_tid))  # type: ignore[attr-defined]
                    else:
                        ok_vis = self._fallback_force_assign_visitor(cand_tid)
                    if ok_vis:
                        reassigned_tid = cand_tid
                        with self._lock:
                            self._caregiver_nearby = True
                            ident_tracks = getattr(self.identifier, "_tracks", None)
                            if isinstance(ident_tracks, dict):
                                for tid_k, tr_v in ident_tracks.items():
                                    if tid_k in self._latest_identities or tid_k == cand_tid:
                                        self._latest_identities[tid_k] = tr_v

            # 2) Если нет реального трека PATIENT вне кровати — восстанавливаем защёлку присутствия в кровати
            if zones_now.has_bed and alert_state in (
                "left_bed",
                "missing",
                "sitting_up",
                "fallen",
                "help_call",
            ):
                with self._lock:
                    ident_snapshot = dict(self._latest_identities)
                ident_tracks = getattr(self.identifier, "_tracks", None)
                if isinstance(ident_tracks, dict):
                    for tid_k, tr_v in ident_tracks.items():
                        ident_snapshot[tid_k] = tr_v
                has_real_patient_outside = any(
                    ident_snapshot.get(o.track_id) is not None
                    and ident_snapshot[o.track_id].role == Role.PATIENT
                    and not is_person_on_bed(o, zones_now.bed_polygon, zones=zones_now)
                    for o in obs_snapshot
                )
                if not has_real_patient_outside:
                    if hasattr(self.identifier, "restore_bed_presence"):
                        self.identifier.restore_bed_presence()  # type: ignore[attr-defined]
                    else:
                        for attr_name in ("_patient_in_bed_latched", "patient_in_bed_latched"):
                            if hasattr(self.identifier, attr_name):
                                with contextlib.suppress(AttributeError, TypeError):
                                    setattr(self.identifier, attr_name, True)

            # 3) Снимаем ложную тревогу в FSM без 15-минутной паузы
            bed_latched_now = self._is_patient_in_bed_latched()
            if hasattr(self.fsm, "dismiss_false_alert"):
                if (
                    not should_demote
                    and target_obs is not None
                    and alert_state in ("sitting_up", "left_bed")
                ):
                    try:
                        self.fsm.dismiss_false_alert(  # type: ignore[attr-defined]
                            bed_latched_now,
                            seated_patient_obs=target_obs,
                            zones=zones_now,
                        )
                    except TypeError:
                        self.fsm.dismiss_false_alert(bed_latched_now)  # type: ignore[attr-defined]
                else:
                    self.fsm.dismiss_false_alert(bed_latched_now)  # type: ignore[attr-defined]
            elif hasattr(self.fsm, "reset_to_idle"):
                self.fsm.reset_to_idle()  # type: ignore[attr-defined]
                if bed_latched_now and self.fsm.state != PatientState.PAUSED:
                    with contextlib.suppress(AttributeError, TypeError):
                        self.fsm._state = PatientState.IN_BED

        # 4) Чистим ложный пик в хронике, пишем событие и публикуем телеметрию
        caregiver_now = bool(
            self._caregiver_nearby or getattr(self.identifier, "door_visit_active", False)
        )
        tl_fallback = (
            "caregiver"
            if (is_recent_alert and (reassigned_tid is not None or caregiver_now))
            else "in_bed"
        )
        tl_cleaned = self.timeline.dismiss_false_alarm_around(
            event_wall_ts,
            fallback_state=tl_fallback,
        )
        with self._lock:
            self._sync_tracks_summary_locked()
        face_log_note = (
            f", эталон лица «Своего» сохранён ({visitor_face_file})"
            if visitor_face_saved and visitor_face_file
            else ""
        )
        self._log_event(
            "info",
            f"Отмечена ложная тревога ({clean_reporter}) — пакет {clean_id} сохранён{face_log_note}",
            category="system",
            clip_id=clean_id,
        )
        self.publish_telemetry()

        return {
            "ok": True,
            "alert_id": clean_id,
            "feedback": fb_entry,
            "clip": clip_entry,
            "feedback_stats": self.feedback.stats(),
            "state": self.fsm.state.value,
            "patient_in_bed_latched": self._is_patient_in_bed_latched(),
            "caregiver_nearby": bool(
                self._caregiver_nearby or getattr(self.identifier, "door_visit_active", False)
            ),
            "timeline_cleaned": bool(tl_cleaned),
            "reassigned_visitor_track_id": reassigned_tid,
            "visitor_face_saved": bool(visitor_face_saved),
            "visitor_face_file": visitor_face_file or None,
            "visitor_face_status": str(visitor_face_status or "not_attempted"),
        }

    def _check_alarm_resolved_by_family(self) -> None:
        """Тревога была, кнопку никто не нажимал, а в кадре появился свой — тревога снята сама."""
        cur = bool(self.fsm.fall_unacknowledged)
        prev, self._prev_fall_unack = self._prev_fall_unack, cur
        if prev and not cur and self.fsm.state == PatientState.CAREGIVER_ASSIST:
            note = "Рядом свой — тревога снята"
            self._log_event("info", note, category="alert", state="caregiver")
            threading.Thread(
                target=self.notifier.mark_latest_resolved,
                args=(note,),
                name="alert-resolved",
                daemon=True,
            ).start()

    def _handle_camera_health(self, frame: np.ndarray, night_ir: bool) -> None:
        try:
            self._sync_camera_health_zones()
            changes = self.camera_health.update(frame, time.time(), night_ir)
        except Exception as exc:
            logger.debug("Сбой контроля камеры: %s", exc)
            return
        for ch in changes:
            if ch.active:
                title = "Изображение с камеры застыло" if ch.kind == "frozen" else "Камеру сдвинули"
                self._log_event("warning", f"{title}: {ch.detail}", category="system")
                if self.settings.notify_camera_offline:
                    self.notifier.send_text(f"⚠️ FallGuard: {title.lower()}. {ch.detail}")
            else:
                title = "Изображение с камеры снова идёт" if ch.kind == "frozen" else "Вид с камеры снова совпадает с разметкой"
                self._log_event("info", title, category="system")
                if self.settings.notify_camera_offline:
                    self.notifier.send_text(f"✅ FallGuard: {title.lower()}.")

    def _dispatch_alert_async(self, event: AlertEvent, alert_jpeg: bytes) -> None:
        clip_id: str | None = None
        if alert_jpeg:
            try:
                saved = self.archive.save_event_snapshot(event, alert_jpeg)
                clip_id = str(saved.get("id") or "") or None
            except Exception as exc:
                logger.warning("Ошибка сохранения снимка тревоги в архив: %s", exc)

        alert_id = clip_id or f"alert_{int(time.time() * 1000)}_{secrets.token_hex(2)}"

        raw_jpeg = b""
        with self._lock:
            raw_frame = self._latest_raw_frame
        if raw_frame is not None and raw_frame.size > 0:
            ok_raw, enc_raw = cv2.imencode(".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok_raw:
                raw_jpeg = enc_raw.tobytes()

        alert_context: dict[str, Any] = {
            "event_wall_ts": round(time.time(), 3),
            "telemetry_snapshot": self.get_telemetry_snapshot(),
            "baseline_in_bed": dict(getattr(self.fsm, "baseline_in_bed", {})),
            "zones": self.zones.get().to_dict(),
            "was_bed_latched": self._is_patient_in_bed_latched(),
        }
        self.feedback.register_recent_alert(
            alert_id=alert_id,
            event=event,
            annotated_jpeg=alert_jpeg,
            raw_jpeg=raw_jpeg,
            context=alert_context,
        )

        ev_state = str(getattr(event.state, "value", event.state))
        self.alert_feed.push(
            alert_id=alert_id,
            state=ev_state,
            severity=event.severity.value,
            title=event.title,
            description=event.description,
            has_image=bool(clip_id),
        )
        self._log_event(
            event.severity.value,
            f"{event.title} — {event.description}",
            category="alert",
            clip_id=clip_id,
            state=ev_state,
        )

        def _task() -> None:
            photo_params = inspect.signature(self.notifier.send_alert_photo).parameters
            if "alert_id" in photo_params:
                msg_ids = self.notifier.send_alert_photo(event, alert_jpeg, alert_id=alert_id)
            else:
                msg_ids = self.notifier.send_alert_photo(event, alert_jpeg)  # type: ignore[call-arg]
            if self._stop_event.wait(self.settings.post_event_seconds):
                return
            min_ts = event.timestamp - self.settings.pre_event_seconds
            frames = self.capture.snapshot_ring_since(min_ts)
            from .recorder import build_mp4_from_jpegs as _orig_build_mp4

            if build_mp4_from_jpegs is not _orig_build_mp4:
                mp4_bytes = build_mp4_from_jpegs(frames, fps=self.settings.infer_fps)
            else:
                mp4_bytes = self.recorder.encode(frames, fps=self.settings.infer_fps)
            if mp4_bytes:
                if clip_id:
                    try:
                        self.archive.attach_mp4(clip_id, mp4_bytes)
                    except Exception as exc:
                        logger.warning("Ошибка сохранения MP4 в архив (%s): %s", clip_id, exc)
                try:
                    self.feedback.attach_mp4_if_flagged(alert_id, mp4_bytes)
                except Exception as exc:
                    logger.warning("Ошибка доклейки MP4 в пакет разбора (%s): %s", alert_id, exc)
                if self.notifier.enabled:
                    vid_params = inspect.signature(self.notifier.send_followup_video).parameters
                    if "alert_id" in vid_params:
                        self.notifier.send_followup_video(
                            mp4_bytes,
                            msg_ids,
                            event.title,
                            alert_id=alert_id,
                        )
                    else:
                        self.notifier.send_followup_video(mp4_bytes, msg_ids, event.title)  # type: ignore[call-arg]

        threading.Thread(target=_task, name="alert-dispatch", daemon=True).start()

    def _run_loop(self) -> None:
        try:
            self.pose.load()
        except Exception as exc:
            logger.error("Ошибка загрузки модели YOLO-pose: %s", exc)
            self._log_event("critical", f"Ошибка загрузки модели: {exc}", category="system")

        target_period = 1.0 / max(1.0, self.settings.infer_fps)
        last_processed_frame_ts = 0.0
        last_tick_mono = 0.0
        ema_fps = 0.0

        with _win32_highres_timer(1):
            next_tick = time.monotonic() + target_period
            while not self._stop_event.is_set():
                loop_t0 = time.monotonic()

                if not self.settings.camera_configured:
                    self.timeline.record(self.fsm.state.value, False)
                    jpeg = self._build_placeholder_jpeg("Укажите RTSP_URL и TG_BOT_TOKEN в файле .env")
                    with self._lock:
                        self._latest_jpeg = jpeg
                    self._stop_event.wait(1.0)
                    next_tick = time.monotonic() + target_period
                    last_tick_mono = 0.0
                    continue

                frame, frame_ts = self.capture.get_latest()
                if frame is None or frame_ts <= 0.0:
                    self.timeline.record(self.fsm.state.value, False)
                    err_hint = self.capture.last_error or "Подключение к камере…"
                    jpeg = self._build_placeholder_jpeg(err_hint)
                    with self._lock:
                        self._latest_jpeg = jpeg
                    self._stop_event.wait(0.5)
                    next_tick = time.monotonic() + target_period
                    last_tick_mono = 0.0
                    continue

                age = loop_t0 - frame_ts
                if age > 20.0:
                    self.timeline.record(self.fsm.state.value, False)
                    if self._camera_was_online and not self._camera_offline_alerted:
                        self._camera_offline_alerted = True
                        self._log_event("warning", "Потерян сигнал камеры (>20 с)", category="system")
                        if self.settings.notify_camera_offline:
                            self.notifier.send_text("⚠️ FallGuard: потерян видеосигнал с IP-камеры (>20 с)!")
                    self._stop_event.wait(0.5)
                    next_tick = time.monotonic() + target_period
                    last_tick_mono = 0.0
                    continue
                else:
                    if self._camera_offline_alerted:
                        self._camera_offline_alerted = False
                        self._log_event("info", "Сигнал камеры восстановлен", category="system")
                        if self.settings.notify_camera_offline:
                            self.notifier.send_text("✅ FallGuard: связь с IP-камерой восстановлена.")
                    self._camera_was_online = True

                if frame_ts == last_processed_frame_ts:
                    self._stop_event.wait(0.002)
                    continue
                last_processed_frame_ts = frame_ts

                zones = self.zones.get()
                try:
                    try:
                        raw_observations = self.pose.infer(frame, zones=zones)
                    except TypeError:
                        raw_observations = self.pose.infer(frame)
                except Exception as exc:
                    logger.error("Сбой инференса кадра: %s", exc)
                    raw_observations = []

                observations = _filter_ignored_observations(raw_observations, zones)

                now = time.monotonic()
                identities, caregiver_nearby = self.identifier.update(
                    observations=observations,
                    zones=zones,
                    now=now,
                    frame_bgr=frame,
                )
                caregiver_nearby = bool(
                    caregiver_nearby or getattr(self.identifier, "door_visit_active", False)
                )
                # Оставляем для FSM, видео и UI только живые треки, принятые identifier
                live_observations = [obs for obs in observations if obs.track_id in identities]

                patient_in_bed_latched = self._is_patient_in_bed_latched()
                step_kwargs: dict[str, Any] = {
                    "observations": live_observations,
                    "identities": identities,
                    "caregiver_nearby": caregiver_nearby,
                    "zones": zones,
                    "now": now,
                }
                step_params = inspect.signature(self.fsm.step).parameters
                if "patient_in_bed_latched" in step_params:
                    step_kwargs["patient_in_bed_latched"] = patient_in_bed_latched
                if "caregiver_visible" in step_params:
                    step_kwargs["caregiver_visible"] = bool(
                        getattr(self.identifier, "caregiver_visible", caregiver_nearby)
                    )
                alert_event = self.fsm.step(**step_kwargs)
                self._check_alarm_resolved_by_family()
                self.timeline.record(self.fsm.state.value, True)
                if self.thumbs is not None:
                    try:
                        self.thumbs.maybe_record(frame, self.fsm.state.value)
                    except Exception as exc:
                        logger.debug("Сбой записи мини-превью хроники: %s", exc)

                annotated = render_annotated_frame(
                    frame_bgr=frame,
                    observations=live_observations,
                    identities=identities,
                    zones=zones,
                    state=self.fsm.state,
                    pause_remaining_sec=self.fsm.pause_remaining_sec(now),
                    status_reason=getattr(self.fsm, "status_reason", ""),
                )
                self.capture.push_annotated_to_ring(annotated, now)

                ok, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                jpeg_bytes = enc.tobytes() if ok else b""

                state_now = self.fsm.state
                cur_opt_gray = _prepare_optical_gray(frame)
                with self._lock:
                    prev_opt_gray = self._prev_opt_gray
                    self._prev_opt_gray = cur_opt_gray

                tracks_ui = self._build_tracks_ui_for_frame(
                    live_observations=live_observations,
                    identities=identities,
                    zones=zones,
                    state_now=state_now,
                    cur_opt_gray=cur_opt_gray,
                    prev_opt_gray=prev_opt_gray,
                    now_mono=now,
                )
                night_ir_now = _detect_night_ir_frame(frame, self.identifier)
                self._handle_camera_health(frame, night_ir_now)

                done_mono = time.monotonic()
                if last_tick_mono > 0.0:
                    frame_period = max(1e-3, done_mono - last_tick_mono)
                    inst_fps = min(self.settings.infer_fps * 1.25, 1.0 / frame_period)
                    ema_fps = inst_fps if ema_fps <= 0.0 else (0.8 * ema_fps + 0.2 * inst_fps)
                else:
                    dt_work = max(1e-3, done_mono - loop_t0)
                    ema_fps = min(self.settings.infer_fps, 1.0 / max(target_period, dt_work))
                last_tick_mono = done_mono

                with self._lock:
                    self._latest_raw_frame = frame
                    self._latest_observations = list(live_observations)
                    self._latest_identities = dict(identities)
                    if jpeg_bytes:
                        self._latest_jpeg = jpeg_bytes
                    self._active_tracks_summary = tracks_ui
                    self._caregiver_nearby = caregiver_nearby
                    self._is_night_ir = night_ir_now
                    self._actual_fps = ema_fps

                self.publish_telemetry()

                if alert_event is not None and jpeg_bytes:
                    self._dispatch_alert_async(alert_event, jpeg_bytes)

                now_end = time.monotonic()
                sleep_rem = next_tick - now_end
                if sleep_rem > 0.001:
                    self._stop_event.wait(sleep_rem)
                next_tick += target_period
                after_sleep = time.monotonic()
                if after_sleep > next_tick:
                    next_tick = after_sleep + target_period
