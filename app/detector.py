from __future__ import annotations

import os

# Исключаем активный spin-wait потоков OpenMP/MKL/BLAS между кадрами на многоядерном CPU
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    cv2.setNumThreads(2)
except Exception:
    pass

from .geometry import (
    KeypointSmoother,
    OpticalKeypointStabilizer,
    PoseObservation,
    StaticObjectGate,
    are_legs_hanging_outside_bed,
    bbox_iou,
    bbox_polygon_overlap_ratio,
    filter_valid_poses,
    is_blanket_sleeper_in_bed,
    is_person_on_bed,
    is_pose_in_ignore_zones,
    is_valid_human_pose,
    point_in_polygon,
    polygon_y_bounds,
    sanitize_collapsed_bed_keypoints,
)
from .patient_id import Role, TrackIdentity
from .state_machine import PatientState
from .zones import ZonesConfig

logger = logging.getLogger("fallguard.detector")

# Связи скелета COCO-17
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _configure_passive_onnxruntime_cpu(intra_threads: int = 4) -> None:
    """
    Отключает агрессивный busy-spin в ONNX Runtime (session.intra_op.allow_spinning = 0),
    удерживающий ядра CPU в холостом ожидании между кадрами при 5 FPS.
    """
    try:
        import onnxruntime as ort

        if getattr(ort, "_fallguard_passive_patched", False):
            return
        orig_cls = ort.InferenceSession

        class _PassiveInferenceSession(orig_cls):  # type: ignore[misc, valid-type]
            def __init__(
                self,
                path_or_bytes: Any,
                sess_options: Any = None,
                providers: Any = None,
                **kwargs: Any,
            ) -> None:
                so = sess_options if sess_options is not None else ort.SessionOptions()
                so.intra_op_num_threads = intra_threads
                so.inter_op_num_threads = 1
                try:
                    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
                except Exception:
                    pass
                super().__init__(path_or_bytes, sess_options=so, providers=providers, **kwargs)

        ort.InferenceSession = _PassiveInferenceSession  # type: ignore[misc]
        setattr(ort, "_fallguard_passive_patched", True)
    except Exception as exc:
        logger.debug("Не удалось применить патч SessionOptions ONNX Runtime: %s", exc)


@dataclass
class _AnonTrackRecord:
    """Лёгкая пространственная память для боксов без ByteTrack ID (при низком conf под одеялом)."""

    track_id: int
    bbox_norm: tuple[float, float, float, float]
    center: tuple[float, float]
    last_ts: float


class PoseTracker:
    """
    Детектор скелета YOLO11-pose + ByteTrack + KeypointSmoother + StaticObjectGate.
    Отсекает неподвижную одежду на стене, вешалки и зоны игнорирования предметов.
    """

    def __init__(
        self,
        model_name: str,
        models_dir: Path,
        device: str = "auto",
        conf: float = 0.48,
        smooth_keypoints: bool = True,
        roi_probe_interval: float = 1.6,
    ) -> None:
        self._model_name = model_name
        self._models_dir = models_dir
        self._device_pref = device
        self._conf = conf
        self._smooth_keypoints = smooth_keypoints
        self._roi_probe_interval = max(0.0, float(roi_probe_interval))
        self._smoother = KeypointSmoother()
        self._kp_stabilizer = OpticalKeypointStabilizer()
        self._liveness_gate = StaticObjectGate(min_displacement=0.045)
        self._anon_tracks: dict[int, _AnonTrackRecord] = {}
        self._next_anon_id: int = 10_001
        self._anon_ttl: float = 5.0
        self._last_bed_track_id: int = 9001
        self._last_bed_bbox: tuple[float, float, float, float] | None = None
        self._last_bed_seen_ts: float = 0.0
        self._bed_envelope_bbox: tuple[float, float, float, float] | None = None
        self._bed_envelope_ts: float = 0.0
        self._last_roi_probe_ts: float = 0.0
        self._cached_roi_obs: PoseObservation | None = None
        self._cached_roi_ts: float = 0.0
        self._preferred_roi_idx: int = 0
        self._sub_roi_cursor: int = 0
        self._last_fg_probe_ts: float = 0.0
        self._cached_fg_obs: list[PoseObservation] = []
        self._cached_fg_ts: float = 0.0
        self._model = None
        self._roi_model = None
        self._loaded_model_path: str | None = None
        self._loaded_is_onnx: bool = False
        self._loaded_is_engine: bool = False
        self._runtime_label = "uninitialized"

    @property
    def runtime_label(self) -> str:
        return self._runtime_label

    def _stabilize_track_keypoints(
        self,
        observations: Sequence[PoseObservation],
        frame_bgr: np.ndarray | None = None,
        zones: ZonesConfig | None = None,
        now: float | None = None,
    ) -> list[PoseObservation]:
        """
        Оптически управляемая стабилизация ключевых точек и бокса по кадру (Волна 6.1):
        - При optical_motion_ratio < 0.012 намертво замораживает скелет (0.0 дрожания ИК-шума, is_still = True).
        - При реальном движении мгновенно раскрывает скоростной фильтр без задержки для тревог (is_still = False).
        """
        return self._kp_stabilizer.stabilize(
            observations,
            frame_bgr=frame_bgr,
            zones=zones,
            now=now,
        )

    def reset_smoother(self) -> None:
        self._smoother.reset()
        self._kp_stabilizer.reset()
        _RENDER_KP_STABILIZER.reset()
        self._liveness_gate.reset()
        self._anon_tracks.clear()
        self._next_anon_id = 10_001
        self._last_bed_track_id = 9001
        self._last_bed_bbox = None
        self._last_bed_seen_ts = 0.0
        self._bed_envelope_bbox = None
        self._bed_envelope_ts = 0.0
        self._last_roi_probe_ts = 0.0
        self._cached_roi_obs = None
        self._cached_roi_ts = 0.0
        self._preferred_roi_idx = 0
        self._sub_roi_cursor = 0
        self._last_fg_probe_ts = 0.0
        self._cached_fg_obs = []
        self._cached_fg_ts = 0.0

    def load(self) -> None:
        from ultralytics import YOLO
        import torch

        self._models_dir.mkdir(parents=True, exist_ok=True)
        use_cuda = False
        if self._device_pref in ("cuda", "auto"):
            use_cuda = bool(torch.cuda.is_available())

        raw_name = Path(self._model_name).name
        stem_name = Path(raw_name).stem
        is_explicit_engine = raw_name.lower().endswith(".engine")
        engine_path = (
            self._models_dir / raw_name
            if is_explicit_engine
            else (self._models_dir / f"{stem_name}.engine")
        )
        pt_path = self._models_dir / f"{stem_name}.pt"

        if use_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            if engine_path.exists():
                try:
                    trt_model = YOLO(str(engine_path), task="pose")
                    # Проверочный прогрев TensorRT-контекста на статическом входе 640x640
                    trt_model.predict(
                        source=np.zeros((640, 640, 3), dtype=np.uint8),
                        imgsz=640,
                        device=0,
                        rect=False,
                        verbose=False,
                    )
                    self._loaded_model_path = str(engine_path)
                    self._loaded_is_onnx = False
                    self._loaded_is_engine = True
                    self._model = trt_model
                    self._roi_model = None
                    self._runtime_label = f"TensorRT FP16 ({gpu_name})"
                    logger.info("PoseTracker запущен на TensorRT: %s", self._runtime_label)
                    return
                except Exception as exc:
                    logger.warning(
                        "Не удалось активировать TensorRT engine %s (%s) — безопасный откат на PyTorch .pt",
                        engine_path.name,
                        exc,
                    )
                    self._loaded_is_engine = False

        if not pt_path.exists():
            source_spec = f"{stem_name}.pt" if is_explicit_engine else self._model_name
            base_model = YOLO(source_spec)
            src_ckpt = Path(getattr(base_model, "ckpt_path", source_spec))
            if src_ckpt.exists() and src_ckpt.resolve() != pt_path.resolve():
                shutil.copy2(src_ckpt, pt_path)

        if use_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            self._loaded_model_path = str(pt_path)
            self._loaded_is_onnx = False
            self._loaded_is_engine = False
            self._model = YOLO(self._loaded_model_path)
            self._roi_model = None
            self._runtime_label = f"CUDA ({gpu_name})"
            logger.info("PoseTracker запущен на GPU: %s", self._runtime_label)
            return

        try:
            torch.set_num_threads(2)
        except Exception:
            pass
        _configure_passive_onnxruntime_cpu(intra_threads=4)

        onnx_path = pt_path.with_suffix(".onnx")
        if not onnx_path.exists():
            logger.info("Экспорт %s -> %s под ONNX Runtime CPU...", pt_path.name, onnx_path.name)
            pt_model = YOLO(str(pt_path))
            exported = pt_model.export(format="onnx", imgsz=640, simplify=False, dynamic=False)
            exp_path = Path(str(exported))
            if exp_path.exists() and exp_path.resolve() != onnx_path.resolve():
                shutil.move(str(exp_path), str(onnx_path))

        self._loaded_model_path = str(onnx_path)
        self._loaded_is_onnx = True
        self._loaded_is_engine = False
        self._model = YOLO(self._loaded_model_path, task="pose")
        self._roi_model = None
        self._runtime_label = "ONNX Runtime (CPU AVX2, passive)"
        logger.info("PoseTracker запущен: %s", self._runtime_label)

    def _get_roi_model(self) -> Any:
        """
        Отдельный ленивый экземпляр YOLO для stateless predict() по кропу кровати:
        гарантирует, что чередование track() и predict() не пересоздаёт predictor
        и не сбрасывает внутреннее состояние ByteTrack основного потока.
        """
        if self._roi_model is not None:
            return self._roi_model
        if self._loaded_model_path is None:
            return self._model
        from ultralytics import YOLO

        if self._loaded_is_onnx or self._loaded_is_engine:
            self._roi_model = YOLO(self._loaded_model_path, task="pose")
        else:
            self._roi_model = YOLO(self._loaded_model_path)
        return self._roi_model

    @staticmethod
    def _bed_roi_windows(zones: ZonesConfig, w: int, h: int) -> list[tuple[int, int, int, int]]:
        """
        Формирует неискажённые пропорциональные окна кропа кровати:
        0 — вся кровать с верхним контекстом;
        1 — правое изголовье крупным планом (~1:1 для мелкой головы под пёстрым одеялом);
        2 — левое изголовье крупным планом.
        """
        xs = [float(p[0]) for p in zones.bed_polygon]
        ys = [float(p[1]) for p in zones.bed_polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bw = max(0.05, max_x - min_x)

        specs = (
            (min_x - 0.05, max_x + 0.05, min_y - 0.22, max_y + 0.02),
            (min_x + 0.32 * bw, max_x + 0.03, min_y - 0.05, max_y + 0.02),
            (min_x - 0.03, max_x - 0.32 * bw, min_y - 0.05, max_y + 0.02),
        )
        windows: list[tuple[int, int, int, int]] = []
        for x1f, x2f, y1f, y2f in specs:
            rx1 = int(max(0, round(x1f * w)))
            rx2 = int(min(w, round(x2f * w)))
            ry1 = int(max(0, round(y1f * h)))
            ry2 = int(min(h, round(min(1.0, y2f) * h)))
            if (rx2 - rx1) >= 64 and (ry2 - ry1) >= 64:
                windows.append((rx1, ry1, rx2, ry2))
        return windows

    @staticmethod
    def _is_reclining_in_bed(obs: PoseObservation, bed_polygon: Sequence[Sequence[float]]) -> bool:
        """
        Проверяет, является ли полнокадровый трек реально лежащим в кровати/изголовье человеком,
        а не вертикально стоящим или прямо сидящим на краю кровати родственником (Свой).
        Поддерживает продольный/косой ракурс сна вдоль матраса (aspect_ratio_wh = 0.55..0.85 внутри bed_polygon).
        """
        if not is_person_on_bed(obs, bed_polygon):
            return False
        if are_legs_hanging_outside_bed(obs, bed_polygon):
            return False
        bed_min_y, bed_max_y = polygon_y_bounds(bed_polygon)
        contained_on_mattress = (
            bbox_polygon_overlap_ratio(obs.bbox_norm, bed_polygon) >= 0.68
            and obs.bbox_norm[1] >= (bed_min_y - 0.03)
            and obs.bbox_norm[3] <= (bed_max_y + 0.025)
            and obs.bbox_height <= 0.34
        )
        angle = obs.torso_angle_from_vertical_deg()
        if not contained_on_mattress:
            if angle is not None and angle < 45.0 and obs.bbox_height >= 0.32 and obs.aspect_ratio_wh < 0.85:
                return False
        elif angle is not None and angle < 25.0 and obs.torso_vertical_rise() >= 0.14:
            return False
        if angle is not None and angle < 20.0 and obs.bbox_height >= 0.20 and obs.aspect_ratio_wh < 0.35:
            return False
        if obs.bbox_height >= 0.44 and obs.aspect_ratio_wh < 0.80:
            return False
        return True

    @staticmethod
    def _is_duplicate_of_existing(
        cand: PoseObservation,
        existing_obs: Sequence[PoseObservation],
    ) -> bool:
        """
        Отсекает дублирование уже найденного полнокадрового трека:
        считается дублем при IoU >= 0.30 либо расстоянии между головами/плечами <= 0.08.
        """
        c_top = cand.head_center or cand.upper_body_point
        cx1, cy1, cx2, cy2 = cand.bbox_norm
        cand_area = max(1e-6, max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1))
        for ex in existing_obs:
            if bbox_iou(cand.bbox_norm, ex.bbox_norm) >= 0.30:
                return True
            ex1, ey1, ex2, ey2 = ex.bbox_norm
            inter_w = max(0.0, min(cx2, ex2) - max(cx1, ex1))
            inter_h = max(0.0, min(cy2, ey2) - max(cy1, ey1))
            if (inter_w * inter_h) / cand_area >= 0.45:
                return True
            e_top = ex.head_center or ex.upper_body_point
            dx = c_top[0] - e_top[0]
            dy = c_top[1] - e_top[1]
            if (dx * dx + dy * dy) <= (0.08 * 0.08):
                return True
        return False

    @staticmethod
    def _boxes_mergeable_on_mattress(
        cand_box: tuple[float, float, float, float],
        ref_box: tuple[float, float, float, float],
    ) -> bool:
        """Проверяет, принадлежат ли два горизонтальных матрасных бокса одному спящему человеку."""
        if bbox_iou(cand_box, ref_box) >= 0.15:
            return True
        cx1, cy1, cx2, cy2 = cand_box
        rx1, ry1, rx2, ry2 = ref_box
        cw = max(1e-6, cx2 - cx1)
        ch = max(1e-6, cy2 - cy1)
        rw = max(1e-6, rx2 - rx1)
        rh = max(1e-6, ry2 - ry1)
        min_area = max(1e-6, min(cw * ch, rw * rh))
        inter_w = max(0.0, min(cx2, rx2) - max(cx1, rx1))
        inter_h = max(0.0, min(cy2, ry2) - max(cy1, ry1))
        if (inter_w * inter_h) / min_area >= 0.30:
            return True
        ccx, ccy = (cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5
        rcx, rcy = (rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5
        if math.hypot(ccx - rcx, ccy - rcy) <= 0.20:
            return True
        if (inter_w / max(1e-6, min(cw, rw))) >= 0.50 and abs(ccy - rcy) <= 0.18:
            return True
        return False

    @classmethod
    def _is_bed_sleeper_duplicate(
        cls,
        cand: PoseObservation,
        stabilized_bed_box: tuple[float, float, float, float],
        raw_bed_box: tuple[float, float, float, float],
        bed_polygon: Sequence[Sequence[float]],
    ) -> bool:
        """
        Отсекает паразитные вторые боксы на одной спящей под одеялом бабушке при низком пороге conf=0.07
        (дублирующий полнокадровый бокс или расколотый фрагмент одеяла/ног на односпальном матрасе),
        НЕ затрагивая реальных стоящих перед кроватью или сидящих со свешенными ногами родственников.
        """
        if len(bed_polygon) < 3:
            return False
        if are_legs_hanging_outside_bed(cand, bed_polygon):
            return False
        _, bed_max_y = polygon_y_bounds(bed_polygon)
        if cand.bbox_norm[3] > (bed_max_y + 0.04) or cand.bbox_height > 0.34:
            return False

        is_reclining_cand = cls._is_reclining_in_bed(cand, bed_polygon)
        is_blanket_fold_cand = (
            not is_reclining_cand
            and cand.bbox_norm[3] <= (bed_max_y + 0.03)
            and cand.bbox_height <= 0.28
            and cand.aspect_ratio_wh >= 1.15
            and bbox_polygon_overlap_ratio(cand.bbox_norm, bed_polygon) >= 0.65
        )
        if not (is_reclining_cand or is_blanket_fold_cand):
            return False

        return cls._boxes_mergeable_on_mattress(
            cand.bbox_norm, stabilized_bed_box
        ) or cls._boxes_mergeable_on_mattress(cand.bbox_norm, raw_bed_box)

    def _allocate_bed_roi_track_id(self, existing_ids: set[int]) -> int:
        tid = self._last_bed_track_id
        if tid in existing_ids:
            tid = 9001
            while tid in existing_ids:
                tid += 1
            self._last_bed_track_id = tid
        return tid

    def _resolve_anon_track_id(
        self,
        bbox_norm: tuple[float, float, float, float],
        used_ids: set[int],
        now: float,
    ) -> int:
        """
        Выдаёт стабильный монотонный track_id (>= 10_001) боксу, которому ByteTrack не присвоил ID,
        сопоставляя его по IoU >= 0.30 или смещению центра <= 0.12 за последние 5.0 с.
        """
        cx = (bbox_norm[0] + bbox_norm[2]) * 0.5
        cy = (bbox_norm[1] + bbox_norm[3]) * 0.5

        best_tid: int | None = None
        best_score = -1.0
        for tid, rec in self._anon_tracks.items():
            if tid in used_ids:
                continue
            age = now - rec.last_ts
            if age < 0.0 or age > self._anon_ttl:
                continue
            iou = bbox_iou(bbox_norm, rec.bbox_norm)
            dist = math.hypot(cx - rec.center[0], cy - rec.center[1])
            if iou < 0.30 and dist > 0.12:
                continue
            score = iou + max(0.0, 0.12 - dist) * 2.0
            if score > best_score:
                best_score = score
                best_tid = tid

        if best_tid is not None:
            return best_tid

        while self._next_anon_id in used_ids or self._next_anon_id in self._anon_tracks:
            self._next_anon_id += 1
        allocated = self._next_anon_id
        self._next_anon_id += 1
        return allocated

    @staticmethod
    def _clip_bed_bbox_by_occluders(
        box_norm: tuple[float, float, float, float],
        obs: PoseObservation,
        occluders: Sequence[PoseObservation],
    ) -> tuple[float, float, float, float]:
        """
        Поджимает видимую спальную рамку со стороны перекрывающего второго человека (IoU >= 0.10),
        не затрагивая внутренний долгосрочный эталон _bed_envelope_bbox.
        """
        if not occluders:
            return box_norm
        x1, y1, x2, y2 = box_norm
        if (x2 - x1) <= 0.145:
            return box_norm
        anchor = obs.head_center or obs.mid_shoulder
        if anchor is None:
            return box_norm
        vis_kps = [obs.kp(k, 0.30) for k in range(17)]
        vis_xs = [pt[0] for pt in vis_kps if pt is not None]
        min_kx = min(vis_xs) if vis_xs else anchor[0]
        max_kx = max(vis_xs) if vis_xs else anchor[0]

        for occ in occluders:
            if occ.track_id == obs.track_id:
                continue
            if bbox_iou((x1, y1, x2, y2), occ.bbox_norm) < 0.10:
                continue
            occ_cx = 0.5 * (occ.bbox_norm[0] + occ.bbox_norm[2])
            if anchor[0] > occ_cx and min_kx >= (occ_cx - 0.025) and (x2 - x1) > 0.14:
                x1 = min(x2 - 0.14, max(x1, occ.bbox_norm[2] - 0.015))
            elif anchor[0] < occ_cx and max_kx <= (occ_cx + 0.025) and (x2 - x1) > 0.14:
                x2 = max(x1 + 0.14, min(x2, occ.bbox_norm[0] + 0.015))
        return (x1, y1, x2, y2)

    def _stabilize_bed_sleeper_bbox(
        self,
        obs: PoseObservation,
        bed_polygon: Sequence[Sequence[float]],
        now: float,
        occluders: Sequence[PoseObservation] = (),
    ) -> PoseObservation:
        """
        Удерживает полноразмерную спокойную рамку спящей в кровати бабушки (_bed_envelope_bbox):
        когда одеяло закрывает корпус и YOLO сжимает бокс до одной головы/воротника (bw < 0.16 или bh < 0.11
        или площадь < 52% эталона сна), рамка монолитно держит контур спящего человека без прыжков,
        но аккуратно поджимается по границе перекрывающего родственника (occluders).
        """
        if len(bed_polygon) < 3 or not self._is_reclining_in_bed(obs, bed_polygon):
            return obs

        b_xs = [float(p[0]) for p in bed_polygon]
        b_ys = [float(p[1]) for p in bed_polygon]
        b_min_x, b_max_x = min(b_xs), max(b_xs)
        b_min_y, b_max_y = min(b_ys), max(b_ys)
        bed_w = max(0.12, b_max_x - b_min_x)
        bed_h = max(0.12, b_max_y - b_min_y)

        bw = obs.bbox_width
        bh = obs.bbox_height
        area = bw * bh
        stale_env = self._bed_envelope_bbox is None or (now - self._bed_envelope_ts) > 1800.0
        any_box_overlap = any(
            occ.track_id != obs.track_id
            and (
                bbox_iou(obs.bbox_norm, occ.bbox_norm) >= 0.08
                or (
                    self._bed_envelope_bbox is not None
                    and bbox_iou(self._bed_envelope_bbox, occ.bbox_norm) >= 0.10
                )
            )
            for occ in occluders
        )

        if bw >= 0.16 and bh >= 0.11 and (stale_env or not any_box_overlap):
            if stale_env:
                self._bed_envelope_bbox = obs.bbox_norm
            else:
                assert self._bed_envelope_bbox is not None
                env_area = max(
                    1e-6,
                    (self._bed_envelope_bbox[2] - self._bed_envelope_bbox[0])
                    * (self._bed_envelope_bbox[3] - self._bed_envelope_bbox[1]),
                )
                if area >= 0.52 * env_area:
                    a_env = 0.20
                    self._bed_envelope_bbox = (
                        self._bed_envelope_bbox[0] * (1.0 - a_env) + obs.bbox_norm[0] * a_env,
                        self._bed_envelope_bbox[1] * (1.0 - a_env) + obs.bbox_norm[1] * a_env,
                        self._bed_envelope_bbox[2] * (1.0 - a_env) + obs.bbox_norm[2] * a_env,
                        self._bed_envelope_bbox[3] * (1.0 - a_env) + obs.bbox_norm[3] * a_env,
                    )
            self._bed_envelope_ts = now
        elif stale_env:
            # Старт сразу с глубоко укрытой головы: строим естественную горизонтальную рамку сна вдоль кровати
            anchor_pt = obs.head_center or obs.mid_shoulder or obs.body_center
            target_w = min(bed_w * 0.62, max(0.25, bw))
            target_h = min(bed_h * 0.56, max(0.16, bh))
            if (b_max_x - anchor_pt[0]) <= (anchor_pt[0] - b_min_x):
                ex2 = min(b_max_x, max(obs.bbox_norm[2], anchor_pt[0] + 0.04))
                ex1 = max(b_min_x, ex2 - target_w)
            else:
                ex1 = max(b_min_x, min(obs.bbox_norm[0], anchor_pt[0] - 0.04))
                ex2 = min(b_max_x, ex1 + target_w)
            ey1 = max(max(0.0, b_min_y - 0.02), min(obs.bbox_norm[1], anchor_pt[1] - target_h * 0.36))
            ey2 = min(b_max_y, max(obs.bbox_norm[3], ey1 + target_h))
            self._bed_envelope_bbox = (ex1, ey1, ex2, ey2)
            self._bed_envelope_ts = now

        env = self._bed_envelope_bbox
        if env is None:
            return obs

        if any_box_overlap:
            self._bed_envelope_ts = now
            vis_box = self._clip_bed_bbox_by_occluders(env, obs, occluders)
            return PoseObservation(
                track_id=obs.track_id,
                bbox_norm=vis_box,
                conf=obs.conf,
                keypoints_norm=obs.keypoints_norm,
                liveness_verified=obs.liveness_verified,
                spawn_center_hint=obs.spawn_center_hint,
                confirmed_frames=obs.confirmed_frames,
                outside_bed_disp=obs.outside_bed_disp,
                is_still=obs.is_still,
                _kp_stabilized=obs._kp_stabilized,
            )

        env_area = max(1e-6, (env[2] - env[0]) * (env[3] - env[1]))
        is_shrunk = bw < 0.16 or bh < 0.11 or area < (0.52 * env_area)
        if is_shrunk:
            merged = (
                min(env[0], obs.bbox_norm[0]),
                min(env[1], obs.bbox_norm[1]),
                max(env[2], obs.bbox_norm[2]),
                max(env[3], obs.bbox_norm[3]),
            )
            alpha = 0.15
            stable_box = (
                env[0] * (1.0 - alpha) + merged[0] * alpha,
                env[1] * (1.0 - alpha) + merged[1] * alpha,
                env[2] * (1.0 - alpha) + merged[2] * alpha,
                env[3] * (1.0 - alpha) + merged[3] * alpha,
            )
            self._bed_envelope_bbox = stable_box
            self._bed_envelope_ts = now
            vis_box = self._clip_bed_bbox_by_occluders(stable_box, obs, occluders)
            return PoseObservation(
                track_id=obs.track_id,
                bbox_norm=vis_box,
                conf=obs.conf,
                keypoints_norm=obs.keypoints_norm,
                liveness_verified=obs.liveness_verified,
                spawn_center_hint=obs.spawn_center_hint,
                confirmed_frames=obs.confirmed_frames,
                outside_bed_disp=obs.outside_bed_disp,
                is_still=obs.is_still,
                _kp_stabilized=obs._kp_stabilized,
            )

        alpha_steady = 0.15
        steady_box = (
            env[0] * (1.0 - alpha_steady) + obs.bbox_norm[0] * alpha_steady,
            env[1] * (1.0 - alpha_steady) + obs.bbox_norm[1] * alpha_steady,
            env[2] * (1.0 - alpha_steady) + obs.bbox_norm[2] * alpha_steady,
            env[3] * (1.0 - alpha_steady) + obs.bbox_norm[3] * alpha_steady,
        )
        self._bed_envelope_bbox = steady_box
        self._bed_envelope_ts = now
        vis_box = self._clip_bed_bbox_by_occluders(steady_box, obs, occluders)
        return PoseObservation(
            track_id=obs.track_id,
            bbox_norm=vis_box,
            conf=obs.conf,
            keypoints_norm=obs.keypoints_norm,
            liveness_verified=obs.liveness_verified,
            spawn_center_hint=obs.spawn_center_hint,
            confirmed_frames=obs.confirmed_frames,
            outside_bed_disp=obs.outside_bed_disp,
            is_still=obs.is_still,
            _kp_stabilized=obs._kp_stabilized,
        )

    def _infer_single_roi_window(
        self,
        roi_model: Any,
        frame_bgr: np.ndarray,
        win: tuple[int, int, int, int],
        zones: ZonesConfig,
        device_arg: int | str,
        eff_conf: float,
        existing_obs: Sequence[PoseObservation],
        candidate_track_id: int,
    ) -> PoseObservation | None:
        h, w = frame_bgr.shape[:2]
        rx1, ry1, rx2, ry2 = win
        bed_crop = np.ascontiguousarray(frame_bgr[ry1:ry2, rx1:rx2])
        trt_kw = {"rect": False} if self._loaded_is_engine else {}
        roi_results = roi_model.predict(
            source=bed_crop,
            conf=0.07,
            imgsz=640,
            device=device_arg,
            verbose=False,
            **trt_kw,
        )
        if not roi_results:
            return None

        r0 = roi_results[0]
        r_boxes = r0.boxes
        r_kps = r0.keypoints
        if r_boxes is None or r_kps is None or len(r_boxes) == 0:
            return None

        r_xyxy = r_boxes.xyxy.cpu().numpy()
        r_confs = r_boxes.conf.cpu().numpy()
        r_kp_xy = r_kps.xy.cpu().numpy()
        r_kp_conf = (
            r_kps.conf.cpu().numpy()
            if r_kps.conf is not None
            else np.ones((len(r_boxes), r_kp_xy.shape[1]), dtype=np.float32)
        )

        inv_w = 1.0 / float(w)
        inv_h = 1.0 / float(h)
        best_cand: PoseObservation | None = None

        for i in range(len(r_boxes)):
            lx1, ly1, lx2, ly2 = r_xyxy[i]
            bbox_norm = (
                max(0.0, min(1.0, float(rx1 + lx1) * inv_w)),
                max(0.0, min(1.0, float(ry1 + ly1) * inv_h)),
                max(0.0, min(1.0, float(rx1 + lx2) * inv_w)),
                max(0.0, min(1.0, float(ry1 + ly2) * inv_h)),
            )
            pts_list: list[tuple[float, float, float]] = []
            for k in range(min(17, r_kp_xy.shape[1])):
                lx, ly = r_kp_xy[i, k]
                kc = float(r_kp_conf[i, k])
                if kc <= 0.0 and lx == 0.0 and ly == 0.0:
                    pts_list.append((0.0, 0.0, 0.0))
                else:
                    pts_list.append(
                        (
                            max(0.0, min(1.0, float(rx1 + lx) * inv_w)),
                            max(0.0, min(1.0, float(ry1 + ly) * inv_h)),
                            kc,
                        )
                    )
            while len(pts_list) < 17:
                pts_list.append((0.0, 0.0, 0.0))

            cand = sanitize_collapsed_bed_keypoints(
                PoseObservation(
                    track_id=candidate_track_id,
                    bbox_norm=bbox_norm,
                    conf=float(r_confs[i]),
                    keypoints_norm=tuple(pts_list),
                ),
                zones.bed_polygon,
                other_observations=existing_obs,
                zones=zones,
            )
            hc = cand.head_center
            ms = cand.mid_shoulder
            mh = cand.mid_hip
            if ms is None or not point_in_polygon(ms, zones.bed_polygon):
                continue
            if hc is not None:
                if not point_in_polygon(hc, zones.bed_polygon):
                    continue
            elif mh is None or not point_in_polygon(mh, zones.bed_polygon):
                continue
            if not self._is_reclining_in_bed(cand, zones.bed_polygon):
                continue
            if not is_valid_human_pose(
                cand,
                min_kp_conf=0.30,
                min_valid_kps=4,
                min_box_conf=eff_conf,
                zones=zones,
            ):
                continue
            if self._is_duplicate_of_existing(cand, existing_obs):
                continue
            if best_cand is None or cand.conf > best_cand.conf:
                best_cand = cand

        return best_cand

    def _infer_bed_roi_fallback(
        self,
        frame_bgr: np.ndarray,
        zones: ZonesConfig,
        device_arg: int | str,
        eff_conf: float,
        existing_obs: Sequence[PoseObservation],
        exhaustive: bool = False,
        candidate_track_id: int | None = None,
    ) -> PoseObservation | None:
        """
        Прицельный фолбэк по кропу области кровати в родном разрешении Full HD без слепящего окна:
        вызывается когда среди полнокадровых треков нет лежащего в кровати человека
        (в том числе если на краю кровати сидит/стоит родственник).
        """
        roi_model = self._get_roi_model()
        assert roi_model is not None
        h, w = frame_bgr.shape[:2]
        windows = self._bed_roi_windows(zones, w, h)
        if not windows:
            return None

        target_tid = candidate_track_id if candidate_track_id is not None else self._last_bed_track_id
        start_idx = self._preferred_roi_idx % len(windows)
        if len(windows) == 1:
            order = [0]
        elif exhaustive or self._cached_roi_obs is not None:
            order = [start_idx] + [idx for idx in range(len(windows)) if idx != start_idx]
        else:
            # WIN 0 проверяется на КАЖДОМ ROI-проходе без пропуска по 3 циклам;
            # если WIN 0 пуст — в том же проходе проверяется одно из подокна изголовья (1 / 2)
            if start_idx != 0:
                order = [start_idx, 0]
            else:
                sub_idx = 1 + (self._sub_roi_cursor % (len(windows) - 1))
                self._sub_roi_cursor += 1
                order = [0, sub_idx]

        for idx in order:
            cand = self._infer_single_roi_window(
                roi_model=roi_model,
                frame_bgr=frame_bgr,
                win=windows[idx],
                zones=zones,
                device_arg=device_arg,
                eff_conf=eff_conf,
                existing_obs=existing_obs,
                candidate_track_id=target_tid,
            )
            if cand is not None:
                self._preferred_roi_idx = idx
                return cand
        return None

    def _infer_foreground_roi(
        self,
        frame_bgr: np.ndarray,
        zones: ZonesConfig,
        device_arg: int | str,
        eff_conf: float,
        existing_obs: Sequence[PoseObservation],
        now: float,
        has_edge_hint: bool = False,
    ) -> list[PoseObservation]:
        """
        Прицельный проход по ближнему переднему плану пола перед кроватью (y=0.45..1.00):
        вытаскивает человека, сидящего/стоящего спиной у нижнего среза кадра (где на полном кадре
        ByteTrack обрезает укороченный снизу силуэт).
        """
        roi_model = self._get_roi_model()
        if roi_model is None:
            return []
        h, w = frame_bgr.shape[:2]
        xs = [float(p[0]) for p in zones.bed_polygon]
        ys = [float(p[1]) for p in zones.bed_polygon]
        if max(ys) >= 0.92:
            return []
        min_x, max_x = min(xs), max(xs)
        rx1 = int(max(0, round(max(0.0, min_x - 0.15) * w)))
        rx2 = int(min(w, round(min(1.0, max_x + 0.02) * w)))
        ry1 = int(max(0, round(0.45 * h)))
        ry2 = h
        if (rx2 - rx1) < 64 or (ry2 - ry1) < 64:
            return []

        crop = np.ascontiguousarray(frame_bgr[ry1:ry2, rx1:rx2])
        if not has_edge_hint and int(np.ptp(crop[::16, ::16])) < 8:
            return []
        trt_kw = {"rect": False} if self._loaded_is_engine else {}
        res_list = roi_model.predict(
            source=crop,
            conf=0.35,
            imgsz=640,
            device=device_arg,
            verbose=False,
            **trt_kw,
        )
        if not res_list:
            return []
        r0 = res_list[0]
        if r0.boxes is None or r0.keypoints is None or len(r0.boxes) == 0:
            return []

        r_xyxy = r0.boxes.xyxy.cpu().numpy()
        r_confs = r0.boxes.conf.cpu().numpy()
        r_kp_xy = r0.keypoints.xy.cpu().numpy()
        r_kp_conf = (
            r0.keypoints.conf.cpu().numpy()
            if r0.keypoints.conf is not None
            else np.ones((len(r0.boxes), r_kp_xy.shape[1]), dtype=np.float32)
        )

        inv_w = 1.0 / float(w)
        inv_h = 1.0 / float(h)
        used_ids = {o.track_id for o in existing_obs}
        found: list[PoseObservation] = []
        all_known = list(existing_obs)

        for i in range(len(r0.boxes)):
            box_c = float(r_confs[i])
            if box_c < max(eff_conf, 0.55):
                continue
            cx1, cy1, cx2, cy2 = r_xyxy[i]
            bbox_norm = (
                max(0.0, min(1.0, float(rx1 + cx1) * inv_w)),
                max(0.0, min(1.0, float(ry1 + cy1) * inv_h)),
                max(0.0, min(1.0, float(rx1 + cx2) * inv_w)),
                max(0.0, min(1.0, float(ry1 + cy2) * inv_h)),
            )
            if bbox_norm[3] < 0.88 or bbox_norm[1] < 0.48:
                continue

            pts_list: list[tuple[float, float, float]] = []
            for k in range(min(17, r_kp_xy.shape[1])):
                lx, ly = r_kp_xy[i, k]
                kc = float(r_kp_conf[i, k])
                if kc <= 0.0 or (lx <= 0.0 and ly <= 0.0):
                    pts_list.append((0.0, 0.0, 0.0))
                else:
                    pts_list.append(
                        (
                            max(0.0, min(1.0, float(rx1 + lx) * inv_w)),
                            max(0.0, min(1.0, float(ry1 + ly) * inv_h)),
                            kc,
                        )
                    )
            while len(pts_list) < 17:
                pts_list.append((0.0, 0.0, 0.0))

            probe_cand = PoseObservation(
                track_id=0,
                bbox_norm=bbox_norm,
                conf=box_c,
                keypoints_norm=tuple(pts_list),
            )
            if is_person_on_bed(probe_cand, zones.bed_polygon):
                continue
            if not is_valid_human_pose(
                probe_cand,
                min_kp_conf=0.30,
                min_valid_kps=4,
                min_box_conf=eff_conf,
                zones=zones,
            ):
                continue
            if self._is_duplicate_of_existing(probe_cand, all_known):
                continue
            tid = self._resolve_anon_track_id(bbox_norm, used_ids, now)
            cand = PoseObservation(
                track_id=tid,
                bbox_norm=bbox_norm,
                conf=box_c,
                keypoints_norm=probe_cand.keypoints_norm,
            )
            used_ids.add(tid)
            found.append(cand)
            all_known.append(cand)

        return found

    def infer(
        self,
        frame_bgr: np.ndarray,
        zones: ZonesConfig | None = None,
    ) -> list[PoseObservation]:
        if self._model is None:
            self.load()
        assert self._model is not None

        h, w = frame_bgr.shape[:2]
        if h < 16 or w < 16:
            return []

        device_arg: int | str = (
            0
            if (self._loaded_is_engine or self._runtime_label.startswith("CUDA"))
            else "cpu"
        )
        eff_conf = max(0.30, float(self._conf))
        has_bed = zones is not None and zones.has_bed
        query_conf = min(eff_conf, 0.07) if has_bed else eff_conf
        trt_kw = {"rect": False} if self._loaded_is_engine else {}
        results = self._model.track(
            source=frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            conf=query_conf,
            imgsz=640,
            device=device_arg,
            verbose=False,
            **trt_kw,
        )

        now = time.monotonic()
        stale_anon = [
            tid for tid, rec in self._anon_tracks.items() if (now - rec.last_ts) > self._anon_ttl
        ]
        for tid in stale_anon:
            del self._anon_tracks[tid]

        has_foreground_edge_hint = False
        observations: list[PoseObservation] = []
        if results:
            res = results[0]
            boxes = res.boxes
            kps = res.keypoints
            if boxes is not None and kps is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                ids = boxes.id.int().cpu().numpy() if boxes.id is not None else None
                kp_xy = kps.xy.cpu().numpy()
                kp_conf = (
                    kps.conf.cpu().numpy()
                    if kps.conf is not None
                    else np.ones((len(boxes), kp_xy.shape[1]), dtype=np.float32)
                )

                inv_w = 1.0 / float(w)
                inv_h = 1.0 / float(h)
                min_raw_c = 0.072 if has_bed else 0.17
                used_ids: set[int] = (
                    {int(ids[j]) for j in range(len(boxes)) if float(confs[j]) >= min_raw_c}
                    if ids is not None
                    else set()
                )

                for i in range(len(boxes)):
                    raw_c = float(confs[i])
                    x1, y1, x2, y2 = xyxy[i]
                    bbox_norm = (
                        max(0.0, min(1.0, float(x1) * inv_w)),
                        max(0.0, min(1.0, float(y1) * inv_h)),
                        max(0.0, min(1.0, float(x2) * inv_w)),
                        max(0.0, min(1.0, float(y2) * inv_h)),
                    )
                    if raw_c < 0.17:
                        if (
                            has_bed
                            and raw_c >= 0.07
                            and bbox_norm[3] >= 0.90
                            and bbox_norm[1] >= 0.48
                            and (bbox_norm[2] - bbox_norm[0]) >= 0.10
                        ):
                            has_foreground_edge_hint = True
                        cx_b = (bbox_norm[0] + bbox_norm[2]) * 0.5
                        cy_b = (bbox_norm[1] + bbox_norm[3]) * 0.5
                        in_bed_box = (
                            has_bed
                            and zones is not None
                            and raw_c >= 0.072
                            and point_in_polygon((cx_b, cy_b), zones.bed_polygon)
                        )
                        if not in_bed_box:
                            continue

                    if ids is not None:
                        tid = int(ids[i])
                    else:
                        tid = self._resolve_anon_track_id(bbox_norm, used_ids, now)
                        used_ids.add(tid)

                    pts_list: list[tuple[float, float, float]] = []
                    for k in range(min(17, kp_xy.shape[1])):
                        kx, ky = kp_xy[i, k]
                        kc = float(kp_conf[i, k])
                        pts_list.append((float(kx) * inv_w, float(ky) * inv_h, kc))
                    while len(pts_list) < 17:
                        pts_list.append((0.0, 0.0, 0.0))

                    raw_obs = PoseObservation(
                        track_id=tid,
                        bbox_norm=bbox_norm,
                        conf=raw_c,
                        keypoints_norm=tuple(pts_list),
                    )
                    observations.append(raw_obs)

        if has_bed and zones is not None and observations:
            observations = [
                sanitize_collapsed_bed_keypoints(
                    o,
                    zones.bed_polygon,
                    other_observations=observations,
                    zones=zones,
                )
                for o in observations
            ]

        observations = filter_valid_poses(
            observations,
            min_kp_conf=0.30,
            min_valid_kps=4,
            min_box_conf=eff_conf,
            zones=zones,
        )

        if has_bed and zones is not None:
            reclining_tracks = [o for o in observations if self._is_reclining_in_bed(o, zones.bed_polygon)]
            if reclining_tracks:
                max_bed_conf = max(o.conf for o in reclining_tracks)
                pinned_matches = [
                    o
                    for o in reclining_tracks
                    if o.track_id == self._last_bed_track_id and o.conf >= max_bed_conf * 0.72
                ]
                best_bed = (
                    max(pinned_matches, key=lambda o: o.conf)
                    if pinned_matches
                    else max(reclining_tracks, key=lambda o: o.conf)
                )
                raw_best_box = best_bed.bbox_norm
                bed_occluders = [
                    o
                    for o in observations
                    if o is not best_bed
                    and not self._is_bed_sleeper_duplicate(
                        o, raw_best_box, raw_best_box, zones.bed_polygon
                    )
                ]
                other_ids = {o.track_id for o in bed_occluders}
                bed_cx = (raw_best_box[0] + raw_best_box[2]) * 0.5
                bed_cy = (raw_best_box[1] + raw_best_box[3]) * 0.5
                can_pin_bed_id = (
                    self._last_bed_seen_ts > 0.0
                    and (now - self._last_bed_seen_ts) <= 120.0
                    and self._last_bed_bbox is not None
                    and self._last_bed_track_id not in other_ids
                    and (
                        bbox_iou(raw_best_box, self._last_bed_bbox) >= 0.20
                        or math.hypot(
                            bed_cx - (self._last_bed_bbox[0] + self._last_bed_bbox[2]) * 0.5,
                            bed_cy - (self._last_bed_bbox[1] + self._last_bed_bbox[3]) * 0.5,
                        )
                        <= 0.22
                    )
                )
                target_bed_tid = (
                    self._last_bed_track_id
                    if (can_pin_bed_id and best_bed.track_id != self._last_bed_track_id)
                    else best_bed.track_id
                )
                sanitized_best_bed = sanitize_collapsed_bed_keypoints(
                    best_bed,
                    zones.bed_polygon,
                    other_observations=bed_occluders,
                    zones=zones,
                )
                stabilized_bed = self._stabilize_bed_sleeper_bbox(
                    PoseObservation(
                        track_id=target_bed_tid,
                        bbox_norm=raw_best_box,
                        conf=sanitized_best_bed.conf,
                        keypoints_norm=sanitized_best_bed.keypoints_norm,
                        liveness_verified=sanitized_best_bed.liveness_verified,
                        spawn_center_hint=sanitized_best_bed.spawn_center_hint,
                        confirmed_frames=sanitized_best_bed.confirmed_frames,
                        outside_bed_disp=sanitized_best_bed.outside_bed_disp,
                    ),
                    zones.bed_polygon,
                    now,
                    occluders=bed_occluders,
                )
                env_ref_box = self._bed_envelope_bbox or stabilized_bed.bbox_norm
                deduped_obs: list[PoseObservation] = []
                for o in observations:
                    if o is best_bed:
                        deduped_obs.append(stabilized_bed)
                    elif not self._is_bed_sleeper_duplicate(
                        o, env_ref_box, raw_best_box, zones.bed_polygon
                    ):
                        deduped_obs.append(o)
                observations = deduped_obs
                best_bed = stabilized_bed

                self._last_bed_track_id = best_bed.track_id
                self._last_bed_bbox = self._bed_envelope_bbox or best_bed.bbox_norm
                self._last_bed_seen_ts = now
                self._cached_roi_obs = best_bed
                self._cached_roi_ts = now
            else:
                existing_ids = {o.track_id for o in observations}
                roi_tid = self._allocate_bed_roi_track_id(existing_ids)
                someone_else_present = len(observations) > 0
                recently_in_bed = self._last_bed_seen_ts > 0.0 and (now - self._last_bed_seen_ts) <= 30.0
                active_interval = min(0.50, self._roi_probe_interval)
                needed_interval = (
                    active_interval
                    if (self._cached_roi_obs is not None or someone_else_present or recently_in_bed)
                    else self._roi_probe_interval
                )
                first_probe = self._last_roi_probe_ts == 0.0
                cache_motion_broken = (
                    self._cached_roi_obs is not None
                    and not self._kp_stabilizer.is_box_optically_still(
                        frame_bgr, self._cached_roi_obs.bbox_norm
                    )
                )
                if cache_motion_broken:
                    self._cached_roi_obs = None
                    self._cached_roi_ts = 0.0
                if first_probe or cache_motion_broken or (now - self._last_roi_probe_ts) >= needed_interval:
                    self._last_roi_probe_ts = now
                    roi_cand = self._infer_bed_roi_fallback(
                        frame_bgr=frame_bgr,
                        zones=zones,
                        device_arg=device_arg,
                        eff_conf=eff_conf,
                        existing_obs=observations,
                        exhaustive=(first_probe or someone_else_present or recently_in_bed),
                        candidate_track_id=roi_tid,
                    )
                    if roi_cand is not None:
                        roi_cand = self._stabilize_bed_sleeper_bbox(
                            roi_cand,
                            zones.bed_polygon,
                            now,
                            occluders=observations,
                        )
                        self._cached_roi_obs = roi_cand
                        self._cached_roi_ts = now
                        self._last_bed_track_id = roi_cand.track_id
                        self._last_bed_bbox = self._bed_envelope_bbox or roi_cand.bbox_norm
                        self._last_bed_seen_ts = now
                        observations.append(roi_cand)
                    elif (
                        self._cached_roi_obs is not None
                        and not cache_motion_broken
                        and frame_bgr is not None
                        and int(np.ptp(frame_bgr)) >= 8
                        and 0.0 <= (now - self._cached_roi_ts) <= 2.5
                    ):
                        cached = self._cached_roi_obs
                        if cached.track_id in existing_ids:
                            cached = PoseObservation(
                                track_id=roi_tid,
                                bbox_norm=cached.bbox_norm,
                                conf=cached.conf,
                                keypoints_norm=cached.keypoints_norm,
                            )
                            self._cached_roi_obs = cached
                        if not self._is_duplicate_of_existing(cached, observations):
                            self._last_bed_track_id = cached.track_id
                            self._last_bed_bbox = cached.bbox_norm
                            observations.append(cached)
                    else:
                        self._cached_roi_obs = None
                        self._cached_roi_ts = 0.0
                elif self._cached_roi_obs is not None and (now - self._cached_roi_ts) <= max(
                    0.75, needed_interval + 0.25
                ):
                    cached = self._cached_roi_obs
                    if cached.track_id in existing_ids:
                        cached = PoseObservation(
                            track_id=roi_tid,
                            bbox_norm=cached.bbox_norm,
                            conf=cached.conf,
                            keypoints_norm=cached.keypoints_norm,
                        )
                        self._cached_roi_obs = cached
                    if not self._is_duplicate_of_existing(cached, observations):
                        self._last_bed_track_id = cached.track_id
                        self._last_bed_bbox = cached.bbox_norm
                        observations.append(cached)

            has_fullframe_fg_person = any(
                o.bbox_norm[3] >= 0.90 and not is_person_on_bed(o, zones.bed_polygon)
                for o in observations
            )
            if has_fullframe_fg_person:
                self._cached_fg_obs = []
                self._cached_fg_ts = 0.0
            else:
                fg_active = bool(self._cached_fg_obs) or has_foreground_edge_hint
                fg_interval = (
                    min(0.45, self._roi_probe_interval)
                    if fg_active
                    else self._roi_probe_interval
                )
                first_fg_probe = self._last_fg_probe_ts == 0.0
                if first_fg_probe or (now - self._last_fg_probe_ts) >= fg_interval:
                    self._last_fg_probe_ts = now
                    fg_found = self._infer_foreground_roi(
                        frame_bgr=frame_bgr,
                        zones=zones,
                        device_arg=device_arg,
                        eff_conf=eff_conf,
                        existing_obs=observations,
                        now=now,
                        has_edge_hint=has_foreground_edge_hint,
                    )
                    self._cached_fg_obs = list(fg_found)
                    self._cached_fg_ts = now if fg_found else 0.0
                    observations.extend(fg_found)
                elif self._cached_fg_obs and (now - self._cached_fg_ts) <= max(
                    0.85, self._roi_probe_interval + 0.25
                ):
                    used_now = {o.track_id for o in observations}
                    rebuilt_fg: list[PoseObservation] = []
                    for c_obs in self._cached_fg_obs:
                        if self._is_duplicate_of_existing(c_obs, observations):
                            continue
                        if c_obs.track_id in used_now:
                            new_tid = self._resolve_anon_track_id(c_obs.bbox_norm, used_now, now)
                            c_obs = PoseObservation(
                                track_id=new_tid,
                                bbox_norm=c_obs.bbox_norm,
                                conf=c_obs.conf,
                                keypoints_norm=c_obs.keypoints_norm,
                            )
                        used_now.add(c_obs.track_id)
                        observations.append(c_obs)
                        rebuilt_fg.append(c_obs)
                    self._cached_fg_obs = rebuilt_fg

        for o in observations:
            self._anon_tracks[o.track_id] = _AnonTrackRecord(
                track_id=o.track_id,
                bbox_norm=o.bbox_norm,
                center=((o.bbox_norm[0] + o.bbox_norm[2]) * 0.5, (o.bbox_norm[1] + o.bbox_norm[3]) * 0.5),
                last_ts=now,
            )

        if observations and zones is not None:
            observations = self._liveness_gate.update_and_filter(
                observations,
                zones=zones,
                now=now,
                frame_bgr=frame_bgr,
            )
        if self._smooth_keypoints:
            observations = self._stabilize_track_keypoints(
                observations,
                frame_bgr=frame_bgr,
                zones=zones,
                now=now,
            )
            if has_bed and zones is not None and observations:
                observations = [
                    sanitize_collapsed_bed_keypoints(
                        o,
                        zones.bed_polygon,
                        other_observations=observations,
                        zones=zones,
                    )
                    for o in observations
                ]
        return observations


# Компактные однострочные плашки ТОЛЬКО для тревожных состояний (в спокойных режимах видео чистое!)
ALERT_BANNER_RU: dict[PatientState, tuple[str, tuple[int, int, int]]] = {
    PatientState.HELP_CALL: ("Зовёт на помощь (поднята рука)", (30, 150, 255)),
    PatientState.SITTING_UP: ("Садится на кровати", (35, 165, 250)),
    PatientState.LEFT_BED: ("Встала с кровати одна", (35, 130, 250)),
    PatientState.MISSING: ("Нет в кровати и в кадре", (45, 120, 245)),
    PatientState.FALLEN: ("Падение вне кровати!", (55, 55, 245)),
}


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color_bgr: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 10,
    gap_len: int = 7,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    edges = (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )
    step = max(4, dash_len + gap_len)
    for (sx, sy), (ex, ey) in edges:
        length = int(np.hypot(ex - sx, ey - sy))
        if length <= 0:
            continue
        ux = (ex - sx) / float(length)
        uy = (ey - sy) / float(length)
        for pos in range(0, length, step):
            end_pos = min(length, pos + dash_len)
            p_start = (int(sx + ux * pos), int(sy + uy * pos))
            p_end = (int(sx + ux * end_pos), int(sy + uy * end_pos))
            cv2.line(img, p_start, p_end, color_bgr, thickness, cv2.LINE_AA)


_RENDER_KP_STABILIZER = OpticalKeypointStabilizer(ttl_seconds=1.8, bed_coast_ttl=0.0)


def render_annotated_frame(
    frame_bgr: np.ndarray,
    observations: Sequence[PoseObservation],
    identities: dict[int, TrackIdentity],
    zones: ZonesConfig,
    state: PatientState,
    pause_remaining_sec: float = 0.0,
    status_reason: str = "",
    draw_zones: bool = False,
) -> np.ndarray:
    """
    Минималистичный видео-HUD:
    - В спокойных режимах (IDLE, IN_BED, PAUSED, CAREGIVER_ASSIST) нет никаких плашек и баннеров.
    - При draw_zones=False (по умолчанию) полигоны кровати, двери и пола НЕ рисуются (чистые фото/MP4 для Telegram);
      при draw_zones=True рисуются тонкие контуры (1 px, alpha = 0.04) без текстовых наклеек.
    - Бейджи над людьми лаконичные: только «Бабушка» или «Свой».
    - Компактная однострочная плашка сверху слева рисуется ТОЛЬКО при тревоге (HELP_CALL, SITTING_UP, LEFT_BED, FALLEN, MISSING).
    - Оптически стабилизирует сырые наблюдения на реальных текстурированных кадрах камеры (ttl=1.8 с), исключая скачки точек в покое.
    """
    if observations and any(not getattr(o, "_kp_stabilized", False) for o in observations):
        if frame_bgr is not None and frame_bgr.size > 0 and int(np.ptp(frame_bgr[::8, ::8])) >= 8:
            observations = _RENDER_KP_STABILIZER.stabilize(
                observations,
                frame_bgr=frame_bgr,
                zones=zones,
            )
        else:
            _RENDER_KP_STABILIZER.reset()
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    # 1. Деликатные тонкие контуры кровати, двери и пола (рисуются только при явном draw_zones=True)
    if draw_zones:
        floor_poly = getattr(zones, "floor_polygon", None)
        has_floor = isinstance(floor_poly, (list, tuple)) and len(floor_poly) >= 4
        if zones.has_bed or zones.has_door or has_floor:
            overlay_zones = canvas.copy()
            if has_floor:
                floor_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in floor_poly[:4]], dtype=np.int32)
                cv2.fillPoly(overlay_zones, [floor_pts], (210, 185, 75))
                cv2.polylines(canvas, [floor_pts], True, (220, 195, 90), 1, cv2.LINE_AA)
            if zones.has_bed:
                bed_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zones.bed_polygon], dtype=np.int32)
                cv2.fillPoly(overlay_zones, [bed_pts], (80, 185, 75))
                cv2.polylines(canvas, [bed_pts], True, (95, 200, 85), 1, cv2.LINE_AA)
            if zones.has_door:
                door_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zones.door_polygon], dtype=np.int32)
                cv2.fillPoly(overlay_zones, [door_pts], (210, 155, 45))
                cv2.polylines(canvas, [door_pts], True, (220, 165, 60), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay_zones, 0.04, canvas, 0.96, 0, canvas)

    ignore_polys = getattr(zones, "ignore_polygons", None)
    door_poly_for_ignore = zones.door_polygon if zones.has_door else None
    badges_to_draw: list[tuple[int, int, str, tuple[int, int, int], bool]] = []

    # 2. Отрисовка подтверждённых живых людей
    for obs in observations:
        if ignore_polys and is_pose_in_ignore_zones(
            obs,
            ignore_polys,
            door_polygon=door_poly_for_ignore,
            bed_polygon=zones.bed_polygon if zones.has_bed else None,
            zones=zones,
        ):
            continue
        ident = identities.get(obs.track_id)
        if ident is None:
            continue
        role = ident.role

        x1 = int(obs.bbox_norm[0] * w)
        y1 = int(obs.bbox_norm[1] * h)
        x2 = int(obs.bbox_norm[2] * w)
        y2 = int(obs.bbox_norm[3] * h)

        h_cm = getattr(ident, "height_cm", None)
        if role == Role.PATIENT:
            if state == PatientState.FALLEN:
                color_bgr = (55, 55, 245)
            elif state in (PatientState.HELP_CALL, PatientState.SITTING_UP, PatientState.LEFT_BED):
                color_bgr = (30, 165, 250)
            else:
                color_bgr = (90, 215, 95)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)
            badges_to_draw.append((x1, y1, "Бабушка", color_bgr, True))
            bone_thick = 2
        elif role == Role.VISITOR:
            color_bgr = (225, 180, 60)
            _draw_dashed_rect(canvas, (x1, y1), (x2, y2), color_bgr, thickness=1, dash_len=10, gap_len=7)
            vis_label = f"Свой · {int(round(float(h_cm)))} см" if h_cm is not None else "Свой"
            badges_to_draw.append((x1, y1, vis_label, color_bgr, False))
            bone_thick = 2
        else:
            color_bgr = (160, 170, 180)
            _draw_dashed_rect(canvas, (x1, y1), (x2, y2), color_bgr, thickness=1, dash_len=6, gap_len=6)
            bone_thick = 1

        kp_draw_conf = 0.48
        visible_pts = [obs.kp(k, kp_draw_conf) for k in range(17)]
        valid_draw_pts = [pt for pt in visible_pts if pt is not None]
        if len(valid_draw_pts) >= 2:
            span_x = max(pt[0] for pt in valid_draw_pts) - min(pt[0] for pt in valid_draw_pts)
            span_y = max(pt[1] for pt in valid_draw_pts) - min(pt[1] for pt in valid_draw_pts)
            skeleton_span = max(span_x, span_y)
        else:
            skeleton_span = 0.0

        calm_bed_no_hip = (
            role == Role.PATIENT
            and state
            in (
                PatientState.IN_BED,
                PatientState.CAREGIVER_ASSIST,
                PatientState.PAUSED,
                PatientState.IDLE,
            )
            and obs.mid_hip is None
            and (not zones.has_bed or is_person_on_bed(obs, zones.bed_polygon, zones=zones))
        )
        hide_skeleton = skeleton_span < 0.11 or calm_bed_no_hip
        if not hide_skeleton:
            for a, b in SKELETON_EDGES:
                pa = visible_pts[a] if a < len(visible_pts) else None
                pb = visible_pts[b] if b < len(visible_pts) else None
                if pa and pb and math.hypot(pa[0] - pb[0], pa[1] - pb[1]) >= 0.025:
                    p1 = (int(pa[0] * w), int(pa[1] * h))
                    p2 = (int(pb[0] * w), int(pb[1] * h))
                    cv2.line(canvas, p1, p2, color_bgr, bone_thick, cv2.LINE_AA)

            for pk in valid_draw_pts:
                pt_px = (int(pk[0] * w), int(pk[1] * h))
                cv2.circle(canvas, pt_px, 3, color_bgr, -1, cv2.LINE_AA)

    alert_info = ALERT_BANNER_RU.get(state)
    if not badges_to_draw and alert_info is None:
        return canvas

    # 3. Компактная типографика Pillow (только лаконичные бейджи людей и однострочная плашка тревоги)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    base_rgba = Image.fromarray(rgb).convert("RGBA")
    ui_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ui_layer)

    font_badge = _load_font(max(12, int(h * 0.022)))
    font_alert = _load_font(max(15, int(h * 0.028)))

    banner_bottom_y = 6
    if alert_info is not None:
        alert_text, accent_bgr = alert_info
        if state == PatientState.SITTING_UP:
            for obs in observations:
                ident = identities.get(obs.track_id)
                if ident and ident.role == Role.PATIENT and zones.has_bed:
                    if are_legs_hanging_outside_bed(obs, zones.bed_polygon, zones=zones):
                        alert_text = "Свесила ноги с кровати"
                    break
        elif state == PatientState.FALLEN and status_reason:
            short_reason = status_reason.strip().rstrip(".")
            if len(short_reason) <= 46:
                alert_text = f"Падение: {short_reason[:1].lower() + short_reason[1:]}"

        accent_rgb = (accent_bgr[2], accent_bgr[1], accent_bgr[0])
        tb = draw.textbbox((0, 0), alert_text, font=font_alert)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        bx0, by0 = 12, 12
        bx1 = min(w - 12, bx0 + tw + 36)
        by1 = by0 + th + 16
        banner_bottom_y = by1 + 6

        bg_rgba = (135, 18, 24, 226) if state == PatientState.FALLEN else (22, 18, 14, 214)
        draw.rounded_rectangle(
            (bx0, by0, bx1, by1),
            radius=8,
            fill=bg_rgba,
            outline=(*accent_rgb, 210),
            width=1,
        )
        dot_r = max(4, int(th * 0.26))
        dot_cx = bx0 + 15
        dot_cy = (by0 + by1) // 2
        draw.ellipse(
            (dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r),
            fill=(*accent_rgb, 255),
        )
        draw.text((bx0 + 26, by0 + 7), alert_text, fill=(255, 255, 255, 255), font=font_alert)

    for lx, ly_box, label_txt, lbgr, is_pat in badges_to_draw:
        lrgb = (lbgr[2], lbgr[1], lbgr[0])
        tb = draw.textbbox((0, 0), label_txt, font=font_badge)
        lw = tb[2] - tb[0]
        lh = tb[3] - tb[1]
        pill_w = lw + 14
        pill_h = lh + 8
        px = max(6, min(w - pill_w - 6, lx))
        py = ly_box - pill_h - 4
        if py < banner_bottom_y and px < 340:
            py = min(h - pill_h - 6, max(banner_bottom_y, ly_box + 6))
        else:
            py = max(6, py)

        if is_pat:
            fill_rgba = (*lrgb, 228)
            text_rgba = (255, 255, 255, 255) if state == PatientState.FALLEN else (12, 16, 20, 255)
        else:
            fill_rgba = (14, 20, 28, 205)
            text_rgba = (*lrgb, 255)

        draw.rounded_rectangle(
            (px, py, px + pill_w, py + pill_h),
            radius=5,
            fill=fill_rgba,
            outline=(*lrgb, 190),
            width=1,
        )
        draw.text((px + 7, py + 3), label_txt, fill=text_rgba, font=font_badge)

    composed = Image.alpha_composite(base_rgba, ui_layer).convert("RGB")
    return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)
