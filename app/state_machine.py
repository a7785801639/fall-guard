from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import enum
import math
import threading
import time
from typing import Any, Sequence

from . import geometry as _geom_mod
from .geometry import (
    PoseObservation,
    are_legs_hanging_outside_bed,
    is_hand_raised_for_help,
    is_person_on_bed,
    polygon_distance_norm,
    polygon_y_bounds,
    sanitize_collapsed_bed_keypoints,
)
from .homography import FloorHomography
from .patient_id import Role, TrackIdentity
from .zones import ZonesConfig


class _FallbackPerspectiveScale:
    """Локальный совместимый расчёт линейного поля масштаба S(v), если PerspectiveScaleField недоступен."""

    def __init__(self, u_ref: float, v_ref: float, kp_slope: float) -> None:
        self._u_ref = u_ref
        self._v_ref = v_ref
        self._kp = kp_slope

    @classmethod
    def from_zones(cls, zones: ZonesConfig | None) -> _FallbackPerspectiveScale:
        if zones is not None and zones.has_bed and len(zones.bed_polygon) >= 3:
            poly = zones.bed_polygon
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            du_bed = max(0.05, max_x - min_x)
            dv_bed = max(0.05, max_y - min_y)
            l_ref = 1.95 if du_bed >= 0.95 * dv_bed else 1.00

            spans: list[tuple[float, float]] = []
            n_poly = len(poly)
            for step_i in range(18):
                vy = min_y + dv_bed * (0.06 + 0.84 * (step_i / 17.0))
                inter_x: list[float] = []
                for idx in range(n_poly):
                    x1, y1 = float(poly[idx][0]), float(poly[idx][1])
                    x2, y2 = float(poly[(idx + 1) % n_poly][0]), float(poly[(idx + 1) % n_poly][1])
                    if (y1 <= vy <= y2) or (y2 <= vy <= y1):
                        if abs(y2 - y1) > 1e-6:
                            tx = x1 + (vy - y1) * (x2 - x1) / (y2 - y1)
                            inter_x.append(tx)
                if len(inter_x) >= 2:
                    s_v = max(inter_x) - min(inter_x)
                    if s_v >= 0.04:
                        spans.append((vy, s_v))
            if spans:
                v_far, s_far = spans[0]
                v_wide, s_wide = max(spans, key=lambda it: it[1])
                u_wide = s_wide / l_ref
                if (v_wide - v_far) >= 0.03:
                    raw_kp = ((s_wide - s_far) / l_ref) / (v_wide - v_far)
                    kp = max(0.08, min(0.32, raw_kp))
                else:
                    kp = 0.16
                return cls(u_ref=u_wide, v_ref=v_wide, kp_slope=kp)
        return cls(u_ref=0.24, v_ref=0.50, kp_slope=0.16)

    def _u_1m(self, v: float) -> float:
        vc = max(0.0, min(1.0, float(v)))
        return max(0.14, min(0.38, self._u_ref + self._kp * (vc - self._v_ref)))

    def dist_plane_cm(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        vm = 0.5 * (float(p1[1]) + float(p2[1]))
        u1m = self._u_1m(vm)
        v1m_plane = u1m * (16.0 / 9.0) * 0.58
        du = (float(p1[0]) - float(p2[0])) / u1m
        dv = (float(p1[1]) - float(p2[1])) / v1m_plane
        return 100.0 * math.hypot(du, dv)

    def box_plane_metrics_cm(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        vc = 0.5 * (float(bbox[1]) + float(bbox[3]))
        u1m = self._u_1m(vc)
        v1m_plane = u1m * (16.0 / 9.0) * 0.58
        w_cm = 100.0 * max(0.0, float(bbox[2]) - float(bbox[0])) / u1m
        h_cm = 100.0 * max(0.0, float(bbox[3]) - float(bbox[1])) / v1m_plane
        diag_cm = math.hypot(w_cm, h_cm)
        return w_cm, h_cm, diag_cm, w_cm * h_cm

    def box_vert_height_cm(self, bbox: tuple[float, float, float, float]) -> float:
        vc = 0.5 * (float(bbox[1]) + float(bbox[3]))
        u1m = self._u_1m(vc)
        v1m_vert = u1m * (16.0 / 9.0) * 0.80
        return 100.0 * max(0.0, float(bbox[3]) - float(bbox[1])) / v1m_vert


def _get_perspective_scale(zones: ZonesConfig | None) -> Any:
    psf_cls = getattr(_geom_mod, "PerspectiveScaleField", None)
    if psf_cls is not None and hasattr(psf_cls, "from_zones"):
        bed_poly = zones.bed_polygon if (zones is not None and zones.has_bed) else None
        try:
            return psf_cls.from_zones(zones, bed_poly)
        except TypeError:
            return psf_cls.from_zones(zones)
    return _FallbackPerspectiveScale.from_zones(zones)


def _sanitize_bed_pose_sm(
    obs: PoseObservation,
    bed_polygon: Sequence[Sequence[float]],
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


def _has_adult_limb_cm(obs: PoseObservation, sf: Any, min_cm: float = 22.0) -> bool:
    chains = (
        (5, 7, 9),
        (6, 8, 10),
        (11, 13, 15),
        (12, 14, 16),
    )
    for a_idx, b_idx, c_idx in chains:
        pa = obs.kp(a_idx, 0.30)
        pb = obs.kp(b_idx, 0.30)
        pc = obs.kp(c_idx, 0.30)
        seg = 0.0
        if pa is not None and pb is not None:
            seg += float(sf.dist_plane_cm(pa, pb))
        if pb is not None and pc is not None:
            seg += float(sf.dist_plane_cm(pb, pc))
        if seg == 0.0 and pa is not None and pc is not None:
            seg = float(sf.dist_plane_cm(pa, pc))
        if seg >= min_cm:
            return True
    return False


class PatientState(str, enum.Enum):
    IDLE = "idle"                      # Ожидание появления / инициализация
    PAUSED = "paused"                  # Ручная пауза («Я рядом»)
    IN_BED = "in_bed"                  # Спокойно лежит в кровати
    CAREGIVER_ASSIST = "caregiver"     # Рядом помощник (подъём под присмотром)
    HELP_CALL = "help_call"            # Вызов помощи поднятой рукой
    SITTING_UP = "sitting_up"          # Садится в кровати / свешивает ноги
    LEFT_BED = "left_bed"              # Встала и вышла из кровати одна
    FALLEN = "fallen"                  # Экстренно: падение / сползание на пол
    MISSING = "missing"                # Ушла из зоны видимости камеры без помощника


class AlertSeverity(str, enum.Enum):
    WARNING = "warning"      # Упреждение (посадка / жест помощи / уход / смена позы)
    CRITICAL = "critical"    # Экстренная тревога с повторами (падение / долгое отсутствие)


@dataclass(frozen=True)
class AlertEvent:
    state: PatientState
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float
    patient_track_id: int | None


@dataclass
class BedBaseline:
    """
    Автоматически калибруемый эталон позы сна бабушки на матрасе в текущем ракурсе камеры.
    Устраняет ложные тревоги при торцевом/косом ракурсе (когда бабушка спит вдоль оси камеры)
    и даёт точный отсчёт подъёма корпуса вверх либо сползания вниз на пол.
    """

    samples: int = 0
    shoulder_y: float = 0.0
    hip_y: float = 0.0
    torso_rise: float = 0.0
    torso_len_2d: float = 0.0
    bbox_h: float = 0.0
    bed_signature: tuple[float, float] = (0.0, 0.0)
    recalib_accum_sec: float = 0.0
    hip_calibrated: bool = False

    @property
    def calibrated(self) -> bool:
        return self.samples >= 4

    def reset(self) -> None:
        self.samples = 0
        self.shoulder_y = 0.0
        self.hip_y = 0.0
        self.torso_rise = 0.0
        self.torso_len_2d = 0.0
        self.bbox_h = 0.0
        self.recalib_accum_sec = 0.0
        self.hip_calibrated = False

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "calibrated": self.calibrated,
            "samples": self.samples,
            "shoulder_y": round(self.shoulder_y, 4),
            "hip_y": round(self.hip_y, 4),
            "torso_rise": round(self.torso_rise, 4),
            "torso_len_2d": round(self.torso_len_2d, 4),
            "bbox_h": round(self.bbox_h, 4),
        }


class FallStateMachine:
    """
    Машина состояний безопасности и послеинсультного ухода, анализирующая ТОЛЬКО Role.PATIENT.
    Треки Role.VISITOR полностью исключены из генерации тревог.
    """

    def __init__(
        self,
        sit_confirm_seconds: float = 1.8,
        leave_confirm_seconds: float = 2.0,
        fall_confirm_seconds: float = 1.8,
        missing_confirm_seconds: float = 12.0,
        alert_cooldown_seconds: float = 45.0,
        suppress_sitting_when_caregiver: bool = True,
        enable_help_gesture: bool = True,
        help_confirm_seconds: float = 3.2,
        prolonged_absence_minutes: float = 20.0,
        immobility_reminder_minutes: float = 0.0,
        night_strict_mode: bool = True,
        night_start_hour: int = 22,
        night_end_hour: int = 7,
        floor_safety_net: bool = True,
    ) -> None:
        self._sit_confirm = float(sit_confirm_seconds)
        self._leave_confirm = float(leave_confirm_seconds)
        self._fall_confirm = float(fall_confirm_seconds)
        self._missing_confirm = float(missing_confirm_seconds)
        self._cooldown = float(alert_cooldown_seconds)
        self._suppress_with_caregiver = bool(suppress_sitting_when_caregiver)

        self._enable_help_gesture = bool(enable_help_gesture)
        self._help_confirm = float(help_confirm_seconds)
        self._prolonged_absence_minutes = max(0.0, float(prolonged_absence_minutes))
        self._immobility_reminder_minutes = max(0.0, float(immobility_reminder_minutes))
        self._night_strict_mode = bool(night_strict_mode)
        self._night_start_hour = int(night_start_hour) % 24
        self._night_end_hour = int(night_end_hour) % 24

        self._lock = threading.RLock()
        self._state = PatientState.IDLE
        self._status_reason: str = "ожидание в кадре"
        self._paused_until: float = 0.0
        self._last_eval_ts: float | None = None

        # Таймеры непрерывности кандидатов состояния
        self._help_timer = 0.0
        self._sit_timer = 0.0
        self._leave_timer = 0.0
        self._fall_timer = 0.0
        self._missing_timer = 0.0

        self._was_monitored_recently = False
        self._last_seen_deep_in_bed = False
        self._caregiver_was_present = False

        # Пост-визитное сидение («свой ушёл, а бабушка спокойно сидит и не упала»)
        self._caregiver_active_until: float = -1e9
        self._assisted_seated_latched: bool = False
        self._assisted_seated_on_chair: bool = False
        self._assisted_solo_confirmed: bool = False
        self._assisted_seated_upper_y: float = 0.0
        self._assisted_seated_hip_y: float = 0.0
        self._assisted_seated_center: tuple[float, float] = (0.0, 0.0)
        self._reclining_clear_timer: float = 0.0

        # Контроль затяжного отсутствия вне кровати
        self._away_since: float | None = None
        self._prolonged_absence_alerted = False

        # Контроль длительной неподвижности (профилактика пролежней) и микро-активности
        self._immobility_anchor: tuple[float, float, float, float | None] | None = None
        self._immobility_since: float | None = None
        self._immobility_alerted = False
        self._movement_ref_obs: PoseObservation | None = None
        self._last_movement_ts: float | None = None

        # История (timestamp, upper_y, hip_y) на ~12 с для детекции резких падений и медленного сползания
        self._upper_y_history: deque[tuple[float, float]] = deque(maxlen=65)
        self._hip_y_history: deque[tuple[float, float]] = deque(maxlen=65)
        self._baseline = BedBaseline()
        self._last_alert_by_state: dict[PatientState, float] = {}
        self._fall_unacknowledged = False

        # Страховка «человек на полу»: не зависит от того, кому присвоена роль бабушки
        self._floor_safety_net = bool(floor_safety_net)
        self._down_since: float | None = None
        self._down_center: tuple[float, float] = (0.0, 0.0)
        self._down_track_id: int | None = None
        self._down_alerting = False
        self._down_dismissed_center: tuple[float, float] | None = None
        self._down_dismissed_until: float = -1e9
        self._unattended_fall_timer = 0.0

        # Контроль падения во время паузы «Я рядом»: эпизод, который уже был на момент паузы,
        # считается подтверждённым; новое падение при пустой комнате снова поднимает тревогу
        self._pause_fall_acked = False
        self._pause_fall_timer = 0.0
        self._pause_clear_timer = 0.0

    @property
    def state(self) -> PatientState:
        with self._lock:
            return self._state

    @property
    def status_reason(self) -> str:
        with self._lock:
            return self._status_reason

    @property
    def baseline_in_bed(self) -> dict[str, float | int | bool]:
        with self._lock:
            return self._baseline.to_dict()

    @property
    def fall_unacknowledged(self) -> bool:
        with self._lock:
            return self._fall_unacknowledged

    def pause_remaining_sec(self, now: float) -> float:
        with self._lock:
            return max(0.0, self._paused_until - now)

    def reset_baseline(self) -> None:
        """Принудительный сброс калибровки эталона позы сна на матрасе."""
        with self._lock:
            self._baseline.reset()

    def _is_night_hour(self, hour: int) -> bool:
        if not self._night_strict_mode:
            return False
        h = int(hour) % 24
        start = self._night_start_hour % 24
        end = self._night_end_hour % 24
        if start == end:
            return True
        if start > end:
            return h >= start or h < end
        return start <= h < end

    def _effective_confirm_thresholds(self, hour: int) -> tuple[float, float]:
        if self._is_night_hour(hour):
            eff_sit = min(self._sit_confirm, max(1.0, self._sit_confirm * 0.72))
            eff_leave = min(self._leave_confirm, max(1.1, self._leave_confirm * 0.75))
            return eff_sit, eff_leave
        return self._sit_confirm, self._leave_confirm

    def update_runtime_prefs(self, **kwargs: Any) -> dict[str, Any]:
        """Горячее потокобезопасное обновление настроек чувствительности и ухода."""
        with self._lock:
            if "sit_confirm_seconds" in kwargs and kwargs["sit_confirm_seconds"] is not None:
                self._sit_confirm = max(0.4, min(15.0, float(kwargs["sit_confirm_seconds"])))
            if "leave_confirm_seconds" in kwargs and kwargs["leave_confirm_seconds"] is not None:
                self._leave_confirm = max(0.4, min(15.0, float(kwargs["leave_confirm_seconds"])))
            if "fall_confirm_seconds" in kwargs and kwargs["fall_confirm_seconds"] is not None:
                self._fall_confirm = max(0.4, min(15.0, float(kwargs["fall_confirm_seconds"])))
            if "missing_confirm_seconds" in kwargs and kwargs["missing_confirm_seconds"] is not None:
                self._missing_confirm = max(1.0, min(120.0, float(kwargs["missing_confirm_seconds"])))
            if "alert_cooldown_seconds" in kwargs and kwargs["alert_cooldown_seconds"] is not None:
                self._cooldown = max(1.0, float(kwargs["alert_cooldown_seconds"]))
            if "suppress_sitting_when_caregiver" in kwargs and kwargs["suppress_sitting_when_caregiver"] is not None:
                self._suppress_with_caregiver = bool(kwargs["suppress_sitting_when_caregiver"])
            if "enable_help_gesture" in kwargs and kwargs["enable_help_gesture"] is not None:
                self._enable_help_gesture = bool(kwargs["enable_help_gesture"])
                if not self._enable_help_gesture:
                    self._help_timer = 0.0
            if "help_confirm_seconds" in kwargs and kwargs["help_confirm_seconds"] is not None:
                self._help_confirm = max(1.0, min(15.0, float(kwargs["help_confirm_seconds"])))
            if "prolonged_absence_minutes" in kwargs and kwargs["prolonged_absence_minutes"] is not None:
                self._prolonged_absence_minutes = max(0.0, float(kwargs["prolonged_absence_minutes"]))
            if "immobility_reminder_minutes" in kwargs and kwargs["immobility_reminder_minutes"] is not None:
                prev_imm = self._immobility_reminder_minutes
                self._immobility_reminder_minutes = max(0.0, float(kwargs["immobility_reminder_minutes"]))
                if self._immobility_reminder_minutes != prev_imm:
                    self._immobility_alerted = False
            if "night_strict_mode" in kwargs and kwargs["night_strict_mode"] is not None:
                self._night_strict_mode = bool(kwargs["night_strict_mode"])
            if "night_start_hour" in kwargs and kwargs["night_start_hour"] is not None:
                self._night_start_hour = int(kwargs["night_start_hour"]) % 24
            if "night_end_hour" in kwargs and kwargs["night_end_hour"] is not None:
                self._night_end_hour = int(kwargs["night_end_hour"]) % 24
            if "floor_safety_net" in kwargs and kwargs["floor_safety_net"] is not None:
                self._floor_safety_net = bool(kwargs["floor_safety_net"])
                if not self._floor_safety_net:
                    self._clear_person_down()

            return {
                "floor_safety_net": self._floor_safety_net,
                "sit_confirm_seconds": round(self._sit_confirm, 2),
                "leave_confirm_seconds": round(self._leave_confirm, 2),
                "fall_confirm_seconds": round(self._fall_confirm, 2),
                "missing_confirm_seconds": round(self._missing_confirm, 2),
                "enable_help_gesture": self._enable_help_gesture,
                "help_confirm_seconds": round(self._help_confirm, 2),
                "prolonged_absence_minutes": round(self._prolonged_absence_minutes, 1),
                "immobility_reminder_minutes": round(self._immobility_reminder_minutes, 1),
                "night_strict_mode": self._night_strict_mode,
                "night_start_hour": self._night_start_hour,
                "night_end_hour": self._night_end_hour,
            }

    def get_care_diagnostics(
        self,
        now: float | None = None,
        local_hour: int | None = None,
    ) -> dict[str, Any]:
        """Снимок диагностики послеинсультного ухода и активных порогов для UI/API."""
        with self._lock:
            ts = float(now) if now is not None else (
                self._last_eval_ts if self._last_eval_ts is not None else time.monotonic()
            )
            hour = datetime.now().hour if local_hour is None else int(local_hour)
            night_active = self._is_night_hour(hour)
            eff_sit, eff_leave = self._effective_confirm_thresholds(hour)

            imm_min = (
                round(max(0.0, ts - self._immobility_since) / 60.0, 1)
                if self._immobility_since is not None
                else 0.0
            )
            away_min = (
                round(max(0.0, ts - self._away_since) / 60.0, 1)
                if self._away_since is not None
                else 0.0
            )
            move_ago = (
                round(max(0.0, ts - self._last_movement_ts), 1)
                if self._last_movement_ts is not None
                else None
            )

            return {
                "baseline": self._baseline.to_dict(),
                "night_active": night_active,
                "effective_sit_confirm_sec": round(eff_sit, 2),
                "effective_leave_confirm_sec": round(eff_leave, 2),
                "immobility_minutes": imm_min,
                "away_from_bed_minutes": away_min,
                "last_movement_sec_ago": move_ago,
                "enable_help_gesture": self._enable_help_gesture,
                "prolonged_absence_minutes": round(self._prolonged_absence_minutes, 1),
                "immobility_reminder_minutes": round(self._immobility_reminder_minutes, 1),
            }

    def _clear_away_tracking(self) -> None:
        self._away_since = None
        self._prolonged_absence_alerted = False

    def _clear_immobility_tracking(self) -> None:
        self._immobility_anchor = None
        self._immobility_since = None
        self._immobility_alerted = False

    def _clear_assisted_seated(self) -> None:
        self._assisted_seated_latched = False
        self._assisted_seated_on_chair = False
        self._assisted_solo_confirmed = False
        self._assisted_seated_upper_y = 0.0
        self._assisted_seated_hip_y = 0.0
        self._assisted_seated_center = (0.0, 0.0)
        self._reclining_clear_timer = 0.0

    def acknowledge_and_pause(self, now: float, pause_seconds: float) -> None:
        """Вызывается при нажатии кнопки «Я подошёл» в Telegram или веб-пульте."""
        with self._lock:
            was_calm_in_bed = self._state in (PatientState.IN_BED, PatientState.CAREGIVER_ASSIST)
            self._fall_unacknowledged = False
            self._pause_fall_acked = True
            self._pause_fall_timer = 0.0
            self._pause_clear_timer = 0.0
            if pause_seconds > 0:
                self._paused_until = now + pause_seconds
                self._caregiver_active_until = max(self._caregiver_active_until, now + pause_seconds + 8.0)
                self._state = PatientState.PAUSED
                self._status_reason = "ручная пауза («Я рядом»)"
            else:
                self._paused_until = 0.0
                self._state = PatientState.IDLE
                self._status_reason = "охрана возобновлена"
            self._help_timer = 0.0
            self._sit_timer = 0.0
            self._leave_timer = 0.0
            self._fall_timer = 0.0
            self._missing_timer = 0.0
            self._unattended_fall_timer = 0.0
            self._clear_person_down()
            if not was_calm_in_bed:
                self._last_seen_deep_in_bed = False
            self._clear_away_tracking()
            self._clear_immobility_tracking()

    def resume(self) -> None:
        with self._lock:
            was_paused = self._state == PatientState.PAUSED
            self._paused_until = 0.0
            self._fall_unacknowledged = False
            if was_paused:
                if self._last_eval_ts is not None:
                    self._caregiver_active_until = max(
                        self._caregiver_active_until,
                        self._last_eval_ts + 8.0,
                    )
                self._state = PatientState.IDLE
                self._status_reason = "охрана активна"

    def reset_to_idle(self) -> None:
        """Сбрасывает накопленное состояние тревоги и флаг недавнего нахождения в кровати."""
        with self._lock:
            self._help_timer = 0.0
            self._sit_timer = 0.0
            self._leave_timer = 0.0
            self._fall_timer = 0.0
            self._missing_timer = 0.0
            self._unattended_fall_timer = 0.0
            self._clear_person_down()
            self._was_monitored_recently = False
            self._last_seen_deep_in_bed = False
            self._fall_unacknowledged = False
            self._caregiver_active_until = -1e9
            self._clear_assisted_seated()
            self._clear_away_tracking()
            self._clear_immobility_tracking()
            if self._state != PatientState.PAUSED:
                self._state = PatientState.IDLE
                self._status_reason = "ожидание в кадре"

    def dismiss_false_alert(self, patient_in_bed_latched: bool = False) -> None:
        """Снимает ложную тревогу по кнопке «Ты ошибся» без постановки охраны на 15-минутную паузу."""
        with self._lock:
            prev_state = self._state
            self._fall_unacknowledged = False
            self._help_timer = 0.0
            self._sit_timer = 0.0
            self._leave_timer = 0.0
            self._fall_timer = 0.0
            self._missing_timer = 0.0
            self._unattended_fall_timer = 0.0
            if self._down_alerting and self._last_eval_ts is not None:
                # «Ты ошибся» по страховке: 10 минут не тревожим по этому же месту
                self._down_dismissed_center = self._down_center
                self._down_dismissed_until = self._last_eval_ts + 600.0
            self._clear_person_down()
            self._upper_y_history.clear()
            self._hip_y_history.clear()
            self._clear_away_tracking()
            if prev_state == PatientState.SITTING_UP:
                if self._last_eval_ts is not None:
                    self._caregiver_active_until = max(
                        self._caregiver_active_until,
                        self._last_eval_ts + 8.0,
                    )
            elif prev_state in (PatientState.FALLEN, PatientState.LEFT_BED, PatientState.MISSING):
                self._clear_assisted_seated()
            if self._state != PatientState.PAUSED:
                if patient_in_bed_latched:
                    self._state = PatientState.IN_BED
                    self._status_reason = "лежит в кровати (под одеялом)"
                    self._was_monitored_recently = True
                    self._last_seen_deep_in_bed = True
                else:
                    self._state = PatientState.IDLE
                    self._status_reason = "ложная тревога снята"
                    self._was_monitored_recently = False
                    self._last_seen_deep_in_bed = False

    def _update_micro_movement(self, obs: PoseObservation, now: float) -> None:
        """Фиксирует малейшие живые шевеления пациента (>= 0.012) относительно последнего якоря движения."""
        if self._movement_ref_obs is None or self._movement_ref_obs.track_id != obs.track_id:
            if self._last_movement_ts is None:
                self._last_movement_ts = now
            self._movement_ref_obs = obs
            return

        moved = False
        ref_bc = self._movement_ref_obs.body_center
        cur_bc = obs.body_center
        if math.hypot(cur_bc[0] - ref_bc[0], cur_bc[1] - ref_bc[1]) >= 0.012:
            moved = True
        else:
            for k in range(17):
                p_ref = self._movement_ref_obs.kp(k, 0.32)
                p_cur = obs.kp(k, 0.32)
                if p_ref is not None and p_cur is not None:
                    if math.hypot(p_cur[0] - p_ref[0], p_cur[1] - p_ref[1]) >= 0.012:
                        moved = True
                        break

        if moved:
            self._last_movement_ts = now
            self._movement_ref_obs = obs

    def _check_prolonged_absence(self, now: float, track_id: int | None) -> AlertEvent | None:
        """Однократная критическая эскалация, если бабушка долго не возвращается в кровать."""
        if self._prolonged_absence_minutes <= 0.0:
            return None
        if self._away_since is None or self._prolonged_absence_alerted:
            return None
        if (now - self._away_since) < (self._prolonged_absence_minutes * 60.0):
            return None

        self._prolonged_absence_alerted = True
        self._fall_unacknowledged = True
        mins = int(self._prolonged_absence_minutes)
        self._status_reason = f"вне кровати более {mins} мин"
        return AlertEvent(
            state=self._state,
            severity=AlertSeverity.CRITICAL,
            title="Долго не возвращается в кровать!",
            description=(
                f"Вне кровати без сопровождения более {mins} мин — "
                "проверьте комнату и ванную!"
            ),
            timestamp=now,
            patient_track_id=track_id,
        )

    def _update_immobility_in_bed(self, obs: PoseObservation, now: float) -> AlertEvent | None:
        """Контроль длительного лежания без смены позы (профилактика пролежней)."""
        upper_y = obs.upper_body_point[1]
        hip_y = (obs.mid_hip or obs.body_center)[1]
        center_x = obs.body_center[0]
        angle = obs.torso_angle_from_vertical_deg()
        cur_pose = (upper_y, hip_y, center_x, angle)

        if self._immobility_anchor is None or self._immobility_since is None:
            self._immobility_anchor = cur_pose
            self._immobility_since = now
            self._immobility_alerted = False
            return None

        anc_uy, anc_hy, anc_cx, anc_ang = self._immobility_anchor
        pos_shift = max(
            abs(upper_y - anc_uy),
            abs(hip_y - anc_hy),
            abs(center_x - anc_cx),
            math.hypot(center_x - anc_cx, upper_y - anc_uy),
        )
        ang_shift = (
            abs(angle - anc_ang)
            if (angle is not None and anc_ang is not None)
            else 0.0
        )

        if pos_shift >= 0.045 or ang_shift >= 18.0:
            self._immobility_anchor = cur_pose
            self._immobility_since = now
            self._immobility_alerted = False
            return None

        if (
            self._immobility_reminder_minutes > 0.0
            and not self._immobility_alerted
            and (now - self._immobility_since) >= (self._immobility_reminder_minutes * 60.0)
        ):
            self._immobility_alerted = True
            mins = max(1, int(round(self._immobility_reminder_minutes)))
            return AlertEvent(
                state=PatientState.IN_BED,
                severity=AlertSeverity.WARNING,
                title="Напоминание об уходе: смена позы",
                description=(
                    f"Лежит в одной позе более {mins} мин — "
                    "рекомендуется бережно помочь сменить положение."
                ),
                timestamp=now,
                patient_track_id=obs.track_id,
            )
        return None

    def _has_rapid_drop(self, now: float) -> bool:
        """Проверка резкого падения плеч/головы вниз за последние 1.3 с (dy >= 0.16 высоты кадра)."""
        if len(self._upper_y_history) < 2:
            return False
        latest_t, latest_y = self._upper_y_history[-1]
        if (now - latest_t) > 0.6:
            return False
        for prev_t, prev_y in self._upper_y_history:
            dt = latest_t - prev_t
            if 0.12 <= dt <= 1.35 and (latest_y - prev_y) >= 0.16:
                return True
        return False

    def _has_slow_slide_down(self, now: float) -> bool:
        """
        Детекция медленного послеинсультного сползания вниз с кровати на пол:
        плавное опускание плеч и таза вниз на >= 0.14 высоты кадра за последние 1.4..9.0 с.
        """
        if len(self._upper_y_history) < 3 or len(self._hip_y_history) < 3:
            return False
        latest_t, latest_uy = self._upper_y_history[-1]
        if (now - latest_t) > 0.8:
            return False
        _, latest_hy = self._hip_y_history[-1]

        min_prev_uy = latest_uy
        min_prev_hy = latest_hy
        for prev_t, prev_uy in self._upper_y_history:
            dt = latest_t - prev_t
            if 1.0 <= dt <= 9.5 and prev_uy < min_prev_uy:
                min_prev_uy = prev_uy
        for prev_t, prev_hy in self._hip_y_history:
            dt = latest_t - prev_t
            if 1.0 <= dt <= 9.5 and prev_hy < min_prev_hy:
                min_prev_hy = prev_hy

        return (latest_uy - min_prev_uy) >= 0.14 and (latest_hy - min_prev_hy) >= 0.11

    def _update_bed_baseline(self, obs: PoseObservation, zones: ZonesConfig, dt: float) -> None:
        """
        Автоматическая самокалибровка базовой позы сна на матрасе (baseline_in_bed).
        Учитывает как боковой, так и торцевой/косой ракурс (лежание головой/ногами к камере).
        """
        if not zones.has_bed or not is_person_on_bed(obs, zones.bed_polygon, zones=zones):
            return
        if are_legs_hanging_outside_bed(obs, zones.bed_polygon, zones=zones):
            return

        bed_sig = polygon_y_bounds(zones.bed_polygon)
        if self._baseline.samples > 0 and (
            abs(bed_sig[0] - self._baseline.bed_signature[0]) > 0.05
            or abs(bed_sig[1] - self._baseline.bed_signature[1]) > 0.05
        ):
            # Если пользователь повернул кадр или изменил полигон кровати — перекалибруемся с нуля
            self._baseline.reset()
        self._baseline.bed_signature = bed_sig

        upper = obs.upper_body_point
        raw_mh = obs.mid_hip
        ms = obs.mid_shoulder
        mh = raw_mh or obs.body_center
        rise = obs.torso_vertical_rise()
        t_len = obs.torso_length_2d() or abs(rise)
        angle = obs.torso_angle_from_vertical_deg()

        sf = _get_perspective_scale(zones)
        torso_cm = (
            float(sf.dist_plane_cm(ms, raw_mh))
            if (ms is not None and raw_mh is not None)
            else 0.0
        )
        valid_hip_torso = raw_mh is not None and (ms is None or torso_cm >= 30.5)

        # Не калибруемся на человеке, который с первого же кадра сидит высоко вертикально
        if not self._baseline.calibrated:
            if angle is not None and angle <= 38.0 and rise >= 0.165:
                return
            if len(self._upper_y_history) >= 2:
                prev_uy = self._upper_y_history[-2][1]
                if abs(upper[1] - prev_uy) > 0.045:
                    return
            alpha = 1.0 if self._baseline.samples == 0 else (1.0 / float(self._baseline.samples + 1))
            self._baseline.shoulder_y += alpha * (upper[1] - self._baseline.shoulder_y)
            self._baseline.bbox_h += alpha * (obs.bbox_height - self._baseline.bbox_h)
            if valid_hip_torso and raw_mh is not None:
                if not self._baseline.hip_calibrated:
                    self._baseline.hip_y = raw_mh[1]
                    self._baseline.torso_rise = rise
                    self._baseline.torso_len_2d = t_len
                    self._baseline.hip_calibrated = True
                else:
                    self._baseline.hip_y += alpha * (raw_mh[1] - self._baseline.hip_y)
                    self._baseline.torso_rise += alpha * (rise - self._baseline.torso_rise)
                    self._baseline.torso_len_2d += alpha * (t_len - self._baseline.torso_len_2d)
            self._baseline.samples += 1
            return

        # При укрытом или сжатом тазе (mid_hip is None либо L_torso_cm < 30.5 см) обновляем только shoulder_y!
        if not valid_hip_torso or raw_mh is None:
            self._baseline.recalib_accum_sec = 0.0
            if abs(upper[1] - self._baseline.shoulder_y) <= 0.045:
                self._baseline.shoulder_y += 0.05 * (upper[1] - self._baseline.shoulder_y)
            return

        if not self._baseline.hip_calibrated:
            self._baseline.hip_y = raw_mh[1]
            self._baseline.torso_rise = rise
            self._baseline.torso_len_2d = t_len
            self._baseline.hip_calibrated = True

        # Когда эталон уже откалиброван: обновляем медленным EMA только пока человек лежит близко к уровню сна,
        # чтобы эталон НЕ «уплывал» за человеком в момент медленной посадки или медленного сползания!
        near_baseline = (
            abs(upper[1] - self._baseline.shoulder_y) <= 0.045
            and abs(rise - self._baseline.torso_rise) <= 0.045
        )
        if near_baseline:
            self._baseline.recalib_accum_sec = 0.0
            a_slow = 0.05
            self._baseline.shoulder_y += a_slow * (upper[1] - self._baseline.shoulder_y)
            self._baseline.hip_y += a_slow * (raw_mh[1] - self._baseline.hip_y)
            self._baseline.torso_rise += a_slow * (rise - self._baseline.torso_rise)
            self._baseline.torso_len_2d += a_slow * (t_len - self._baseline.torso_len_2d)
            self._baseline.bbox_h += a_slow * (obs.bbox_height - self._baseline.bbox_h)
        elif angle is not None and angle >= 65.0:
            # Если бабушка перелегла на другой край подушки строго горизонтально и спит >25 с — мягко подстроимся
            self._baseline.recalib_accum_sec += dt
            if self._baseline.recalib_accum_sec >= 25.0:
                self._baseline.shoulder_y = upper[1]
                self._baseline.hip_y = raw_mh[1]
                self._baseline.torso_rise = rise
                self._baseline.torso_len_2d = t_len
                self._baseline.bbox_h = obs.bbox_height
                self._baseline.recalib_accum_sec = 0.0
        else:
            self._baseline.recalib_accum_sec = 0.0

    def _is_upright_standing_pose(
        self,
        obs: PoseObservation,
        zones: ZonesConfig,
        now: float,
    ) -> bool:
        """Явно вертикально стоящая поза (включая наклон вперёд стоя на прямых ногах, кейс 130059)."""
        upper = obs.upper_body_point
        mh = obs.mid_hip or obs.body_center
        angle = obs.torso_angle_from_vertical_deg()
        m3d = FloorHomography.from_zones(
            zones, bed_polygon=zones.bed_polygon if zones.has_bed else None
        ).measure_pose(obs, zones.bed_polygon if zones.has_bed else None)
        sf = _get_perspective_scale(zones)

        has_standing_legs = False
        has_tall_support_leg = False
        for idx in (15, 16):
            ak = obs.kp(idx, 0.35)
            if ak is not None:
                dy_leg = ak[1] - mh[1]
                if dy_leg >= 0.10:
                    has_standing_legs = True
                leg_v_cm = (
                    float(sf.box_vert_height_cm((mh[0], mh[1], ak[0], ak[1])))
                    if dy_leg > 0.0
                    else 0.0
                )
                if dy_leg >= 0.16 or leg_v_cm >= 46.0:
                    has_tall_support_leg = True
        if (
            angle is not None
            and angle < 72.0
            and has_tall_support_leg
            and mh[1] <= 0.72
            and obs.bbox_height >= 0.40
            and obs.aspect_ratio_wh < 0.68
            and not self._has_rapid_drop(now)
        ):
            return True
        return bool(
            angle is not None
            and angle < 30.0
            and not self._has_rapid_drop(now)
            and not self._has_slow_slide_down(now)
            and (
                obs.aspect_ratio_wh < 0.50
                or (obs.aspect_ratio_wh < 0.60 and has_standing_legs and (mh[1] - upper[1]) >= 0.06)
                or (obs.aspect_ratio_wh < 0.62 and m3d.height_cm is not None and m3d.height_cm >= 70.0)
            )
        )

    def _classify_calm_seated_pose(
        self,
        obs: PoseObservation,
        zones: ZonesConfig,
        now: float,
    ) -> tuple[bool, bool]:
        """
        Определяет, находится ли бабушка в ровной сидячей позе без признаков падения/сползания:
        возвращает (is_calm_seated, is_on_bedside_chair).
        """
        upper = obs.upper_body_point
        angle = obs.torso_angle_from_vertical_deg()
        if (
            upper[1] >= 0.64
            or (angle is not None and angle >= 48.0)
            or obs.aspect_ratio_wh >= 1.05
            or self._has_rapid_drop(now)
        ):
            return False, False

        sitting_bed, _ = self._is_sitting_or_exiting_bed(obs, zones)
        if sitting_bed:
            fallen_now, _ = self._check_fallen_or_slumped(obs, zones, now)
            if not fallen_now:
                return True, False
            return False, False

        on_bed = zones.has_bed and is_person_on_bed(obs, zones.bed_polygon, zones=zones)
        if zones.has_bed and not on_bed:
            upright_standing = self._is_upright_standing_pose(obs, zones, now)
            if (
                angle is not None
                and angle < 40.0
                and obs.aspect_ratio_wh < 0.95
                and upper[1] < 0.64
                and not (upright_standing and obs.bbox_height >= 0.54)
                and polygon_distance_norm(obs.body_center, zones.bed_polygon) >= -0.35
            ):
                return True, True
        return False, False

    def _latch_or_update_assisted_seated(
        self,
        obs: PoseObservation,
        on_chair: bool,
        solo_confirmed: bool,
    ) -> None:
        upper_y = obs.upper_body_point[1]
        hip_y = (obs.mid_hip or obs.body_center)[1]
        bc = obs.body_center
        if not solo_confirmed:
            # Исключаем из истории скольжения стоячие высоты во время усаживания с родственником,
            # чтобы после его ухода сползание отсчитывалось строго от сидячей позы, а не от стояния.
            self._upper_y_history = deque(
                ((ts_i, max(uy_i, upper_y - 0.04)) for ts_i, uy_i in self._upper_y_history),
                maxlen=self._upper_y_history.maxlen,
            )
            self._hip_y_history = deque(
                ((ts_i, max(hy_i, hip_y - 0.04)) for ts_i, hy_i in self._hip_y_history),
                maxlen=self._hip_y_history.maxlen,
            )
        if not self._assisted_seated_latched:
            self._assisted_seated_latched = True
            self._assisted_seated_on_chair = on_chair
            self._assisted_solo_confirmed = solo_confirmed
            self._assisted_seated_upper_y = upper_y
            self._assisted_seated_hip_y = hip_y
            self._assisted_seated_center = bc
            self._reclining_clear_timer = 0.0
        else:
            self._assisted_seated_on_chair = on_chair
            self._assisted_solo_confirmed = solo_confirmed
            self._assisted_seated_upper_y = 0.7 * self._assisted_seated_upper_y + 0.3 * upper_y
            self._assisted_seated_hip_y = 0.7 * self._assisted_seated_hip_y + 0.3 * hip_y
            self._assisted_seated_center = (
                0.7 * self._assisted_seated_center[0] + 0.3 * bc[0],
                0.7 * self._assisted_seated_center[1] + 0.3 * bc[1],
            )
            self._reclining_clear_timer = 0.0

    def _check_fallen_or_slumped(
        self,
        obs: PoseObservation,
        zones: ZonesConfig,
        now: float,
    ) -> tuple[bool, str]:
        """
        Критерий резкого падения, медленного сползания с кровати на пол (SLIDING / SLOW FALL)
        и затяжного сидения/лежания на полу у самой кромки кровати (даже при частичном касании
        границы bed_polygon боксом или краем таза).
        """
        upper = obs.upper_body_point
        mh = obs.mid_hip or obs.body_center
        angle = obs.torso_angle_from_vertical_deg()
        on_bed = is_person_on_bed(obs, zones.bed_polygon, zones=zones) if zones.has_bed else False
        bed_min_y, bed_max_y = polygon_y_bounds(zones.bed_polygon) if zones.has_bed else (0.0, 1.0)

        if self._is_upright_standing_pose(obs, zones, now):
            return False, ""

        # 1. Проверка пограничного сползания у самой кромки кровати
        if zones.has_bed and on_bed:
            dist_inside = polygon_distance_norm(mh, zones.bed_polygon)
            at_bottom_rim = dist_inside <= 0.045 or mh[1] >= (bed_max_y - 0.04)
            max_limb_y = max((pt[1] for pt in obs.lower_limbs), default=obs.bbox_norm[3])
            spilled_onto_floor = obs.bbox_norm[3] >= (bed_max_y + 0.055) or max_limb_y >= (bed_max_y + 0.055)
            dropped_below_mattress = (
                (
                    self._baseline.calibrated
                    and self._baseline.hip_calibrated
                    and (upper[1] - self._baseline.shoulder_y) >= 0.125
                    and (mh[1] - self._baseline.hip_y) >= 0.09
                )
                or (
                    self._assisted_seated_latched
                    and (upper[1] - self._assisted_seated_upper_y) >= 0.11
                    and (mh[1] - self._assisted_seated_hip_y) >= 0.085
                )
                or (upper[1] >= (bed_max_y - 0.025) and mh[1] >= (bed_max_y - 0.015))
            )
            compact_or_tilted = (
                obs.bbox_height <= 0.45
                or obs.aspect_ratio_wh >= 1.08
                or (angle is not None and angle >= 50.0)
            )
            if at_bottom_rim and spilled_onto_floor and dropped_below_mattress and compact_or_tilted:
                return True, "Сползла с края кровати на пол у борта матраса."
            return False, ""

        # 2. Человек ВНЕ полигона кровати (на полу в комнате или рядом с кроватью):
        if self._assisted_seated_latched and self._assisted_seated_on_chair:
            if obs.aspect_ratio_wh >= 1.08:
                return True, "Обнаружено горизонтальное положение на полу после сидения на стуле."
            if angle is not None and angle >= 48.0:
                return True, "Сильный наклон корпуса / падение со стула вне кровати."
            if upper[1] >= 0.66:
                return True, "Осела или сползла со стула на пол вне кровати."
            if (upper[1] - self._assisted_seated_upper_y) >= 0.095 or (
                (mh[1] - self._assisted_seated_hip_y) >= 0.095
                and (upper[1] - self._assisted_seated_upper_y) >= 0.06
            ):
                return True, "Медленное сползание / оседание со стула на пол."
            if self._has_slow_slide_down(now) and upper[1] >= 0.56 and obs.bbox_height <= 0.45:
                return True, "Зафиксировано сползание на пол вне кровати."
            return False, ""

        if obs.aspect_ratio_wh >= (1.08 if self._assisted_seated_latched else 1.12):
            return True, "Обнаружено горизонтальное положение на полу вне кровати."

        if angle is not None and angle >= (48.0 if self._assisted_seated_latched else 52.0):
            return True, "Сильный наклон корпуса / падение на пол вне кровати."

        if upper[1] >= 0.64 and obs.bbox_height <= 0.42:
            return True, "Осела или сидит на полу вне кровати."

        if (
            self._assisted_seated_latched
            and (upper[1] - self._assisted_seated_upper_y) >= 0.095
            and obs.bbox_height <= 0.46
        ):
            return True, "Медленное сползание из сидячего положения на пол."

        if (
            self._baseline.calibrated
            and self._baseline.hip_calibrated
            and (upper[1] - self._baseline.shoulder_y) >= 0.12
            and (mh[1] - self._baseline.hip_y) >= 0.10
            and obs.bbox_height <= 0.45
        ):
            return True, "Медленное сползание / оседание на пол ниже уровня кровати."

        if self._has_slow_slide_down(now) and upper[1] >= 0.56 and obs.bbox_height <= 0.45:
            return True, "Зафиксировано сползание на пол вне кровати."

        return False, ""

    def _is_fallen_posture(self, obs: PoseObservation, zones: ZonesConfig) -> bool:
        """Обратно-совместимая обёртка геометрической проверки падения/сползания."""
        now = self._last_eval_ts if self._last_eval_ts is not None else 0.0
        fallen, _ = self._check_fallen_or_slumped(obs, zones, now)
        return fallen

    def _is_sitting_or_exiting_bed(self, obs: PoseObservation, zones: ZonesConfig) -> tuple[bool, str]:
        """
        Ранний перехват: свесила колени/стопы за край кровати ИЛИ реально подняла корпус в постели
        относительно своего откалиброванного уровня сна (без ложных сработок на торцевой ракурс!).
        """
        if not zones.has_bed:
            return False, ""

        if are_legs_hanging_outside_bed(obs, zones.bed_polygon, zones=zones):
            return True, "Свесила ноги за край кровати (готовится встать)"

        if is_person_on_bed(obs, zones.bed_polygon, zones=zones):
            angle = obs.torso_angle_from_vertical_deg()
            if angle is not None and angle >= 62.0 and obs.bbox_height <= 0.28:
                return False, ""
            rise = obs.torso_vertical_rise()
            upper = obs.upper_body_point

            if self._baseline.calibrated:
                shoulder_lift = self._baseline.shoulder_y - upper[1]
                rise_gain = rise - self._baseline.torso_rise
                if shoulder_lift >= 0.068 and (
                    (angle is not None and angle <= 52.0) or rise_gain >= 0.055
                ):
                    return True, "Поднялась и села на кровати"
            else:
                if angle is not None and angle <= 38.0 and rise >= 0.16:
                    return True, "Сидит на кровати"

        return False, ""

    def _clear_person_down(self) -> None:
        self._down_since = None
        self._down_track_id = None
        self._down_alerting = False

    @staticmethod
    def _is_lying_on_floor(obs: PoseObservation, zones: ZonesConfig) -> bool:
        """
        Метрически и анатомически верифицированная низкая поза на полу вне матраса:
        - пропускает горизонтальное падение, позу эмбриона и ракурс вдоль оси камеры (AR >= 0.68 при bh <= 0.34);
        - отсекает мелкие предметы (кошка, тапки, подушка: < 52×26 см или Area < 1650 см²) и ходунки без анатомии.
        """
        if obs.conf < 0.5 or obs.bbox_height > 0.40 or obs.bbox_height < 0.05:
            return False
        if zones.has_bed and is_person_on_bed(obs, zones.bed_polygon, zones=zones):
            return False
        bed_max_y = 0.0
        if zones.has_bed:
            _, bed_max_y = polygon_y_bounds(zones.bed_polygon)
            anchor = obs.mid_hip or obs.body_center
            # Лежащий «на полу» не может быть целиком выше нижней кромки матраса и внутри его контура
            if (
                obs.bbox_norm[3] <= bed_max_y
                and polygon_distance_norm(anchor, zones.bed_polygon) >= 0.0
            ):
                return False

        sf = _get_perspective_scale(zones)
        w_cm, h_pl_cm, _, area_cm2 = sf.box_plane_metrics_cm(obs.bbox_norm)
        border_k = 0.78 if obs.is_near_border(0.02) else 1.0
        l_major = max(float(w_cm), float(h_pl_cm))
        l_minor = min(float(w_cm), float(h_pl_cm))
        if (
            l_major < 52.0 * border_k
            or l_minor < 26.0 * border_k
            or float(area_cm2) < 1650.0 * border_k
        ):
            return False

        ms = obs.mid_shoulder
        mh = obs.mid_hip
        has_valid_anatomy = obs.has_confident_head(0.35) or (
            ms is not None
            and mh is not None
            and float(sf.dist_plane_cm(ms, mh)) >= 28.0 * border_k
            and _has_adult_limb_cm(obs, sf, 22.0 * border_k)
        )
        if not has_valid_anatomy:
            return False

        angle = obs.torso_angle_from_vertical_deg()
        ar = obs.aspect_ratio_wh
        if angle is not None and angle >= 62.0 and ar >= 0.95:
            return True
        if angle is None and ar >= 1.15:
            return True
        if ar >= 0.68 and obs.bbox_height <= 0.34:
            low_floor_band = (not zones.has_bed) or (obs.bbox_norm[1] >= (bed_max_y - 0.04))
            if low_floor_band and (angle is None or angle >= 42.0 or obs.bbox_height <= 0.28):
                return True
        return False

    def _check_person_down(
        self,
        observations: Sequence[PoseObservation],
        identities: dict[int, TrackIdentity],
        zones: ZonesConfig,
        now: float,
    ) -> AlertEvent | None:
        """
        Страховка от путаницы ролей: если ЛЮБОЙ человек (гость / неопознанный) неподвижно
        лежит на полу и рядом нет никого на ногах — это тревога падения, кем бы его ни считал
        идентификатор. Пациент со своей ролью обрабатывается основной веткой.
        """
        if not self._floor_safety_net:
            return None

        candidate: PoseObservation | None = None
        helper_present = False
        for obs in observations:
            ident = identities.get(obs.track_id)
            if ident is None:
                continue
            if self._is_lying_on_floor(obs, zones):
                dismissed_spot = (
                    self._down_dismissed_center is not None
                    and now < self._down_dismissed_until
                    and math.hypot(
                        obs.body_center[0] - self._down_dismissed_center[0],
                        obs.body_center[1] - self._down_dismissed_center[1],
                    )
                    <= 0.10
                )
                if dismissed_spot:
                    continue
                if ident.role != Role.PATIENT and (
                    candidate is None or obs.track_id == self._down_track_id
                ):
                    candidate = obs
                continue
            on_bed = zones.has_bed and is_person_on_bed(obs, zones.bed_polygon, zones=zones)
            angle = obs.torso_angle_from_vertical_deg()
            if not on_bed and obs.bbox_height >= 0.22 and (angle is None or angle < 55.0):
                helper_present = True

        if candidate is None or helper_present:
            if self._down_alerting and self._state == PatientState.FALLEN:
                self._state = PatientState.IDLE
                self._status_reason = "ожидание в кадре"
            self._clear_person_down()
            return None

        center = candidate.body_center
        if (
            self._down_since is None
            or math.hypot(center[0] - self._down_center[0], center[1] - self._down_center[1]) > 0.07
        ):
            # Новый эпизод либо человек заметно перемещается (моет пол, играет ребёнок) — отсчёт заново
            self._down_since = now
            self._down_center = center
            self._down_track_id = candidate.track_id
            if not self._down_alerting:
                return None
        self._down_track_id = candidate.track_id

        needed = max(6.0, self._fall_confirm * 3.0)
        if not self._down_alerting and (now - self._down_since) < needed:
            return None

        first = not self._down_alerting
        self._down_alerting = True
        self._state = PatientState.FALLEN
        self._status_reason = "человек неподвижно лежит на полу"
        if first:
            return self._emit_if_cooled_down(
                PatientState.FALLEN,
                AlertSeverity.CRITICAL,
                "Человек лежит на полу!",
                "Кто-то неподвижно лежит на полу, рядом никого нет. "
                "Нажмите «Я подошёл», чтобы отключить повтор сигнала.",
                now,
                candidate.track_id,
            )
        return None

    def _watch_fall_during_pause(
        self,
        observations: Sequence[PoseObservation],
        identities: dict[int, TrackIdentity],
        zones: ZonesConfig,
        now: float,
        dt: float,
        caregiver_visible: bool,
    ) -> AlertEvent | None:
        """
        Пауза «Я рядом» глушит посадку, выход и жест, но не падение: если в кадре никого из своих,
        а бабушка (или любой человек) оказалась на полу уже после начала паузы — тревога и конец паузы.
        """
        fallen, reason = False, ""
        for obs in observations:
            ident = identities.get(obs.track_id)
            if ident is None:
                continue
            if ident.role == Role.PATIENT:
                pobs = (
                    _sanitize_bed_pose_sm(
                        obs,
                        zones.bed_polygon,
                        other_observations=[x for x in observations if x.track_id != obs.track_id],
                        zones=zones,
                    )
                    if zones.has_bed
                    else obs
                )
                fallen, reason = self._check_fallen_or_slumped(pobs, zones, now)
                if fallen:
                    break
            elif self._floor_safety_net and self._is_lying_on_floor(obs, zones):
                fallen, reason = True, "Человек лежит на полу."
                break

        if caregiver_visible:
            self._pause_fall_timer = 0.0
            return None
        if not fallen:
            self._pause_fall_timer = 0.0
            self._pause_clear_timer += dt
            if self._pause_clear_timer >= 5.0:
                self._pause_fall_acked = False  # прежний эпизод закончился — следующее падение уже новое
            return None
        self._pause_clear_timer = 0.0
        if self._pause_fall_acked:
            return None
        self._pause_fall_timer += dt
        if self._pause_fall_timer < max(2.0, self._fall_confirm * 1.5):
            return None

        self._paused_until = 0.0
        self._pause_fall_timer = 0.0
        self._clear_assisted_seated()
        self._state = PatientState.FALLEN
        self._status_reason = reason.rstrip(".")
        return self._emit_if_cooled_down(
            PatientState.FALLEN,
            AlertSeverity.CRITICAL,
            "Падение во время паузы!",
            f"{reason} Рядом никого нет, пауза охраны снята. "
            "Нажмите «Я подошёл», чтобы отключить повтор сигнала.",
            now,
            None,
        )

    def _emit_if_cooled_down(
        self,
        new_state: PatientState,
        severity: AlertSeverity,
        title: str,
        description: str,
        now: float,
        track_id: int | None,
    ) -> AlertEvent | None:
        last_ts = self._last_alert_by_state.get(new_state, -1e9)
        cooldown = 15.0 if severity == AlertSeverity.CRITICAL else self._cooldown
        if (now - last_ts) < cooldown:
            return None
        self._last_alert_by_state[new_state] = now
        if new_state == PatientState.FALLEN or severity == AlertSeverity.CRITICAL:
            self._fall_unacknowledged = True
        return AlertEvent(
            state=new_state,
            severity=severity,
            title=title,
            description=description,
            timestamp=now,
            patient_track_id=track_id,
        )

    def step(
        self,
        observations: Sequence[PoseObservation],
        identities: dict[int, TrackIdentity],
        caregiver_nearby: bool,
        zones: ZonesConfig,
        now: float,
        local_hour: int | None = None,
        patient_in_bed_latched: bool = False,
        caregiver_visible: bool | None = None,
    ) -> AlertEvent | None:
        """
        caregiver_visible — помощник реально виден в кадре сейчас (None = считать равным
        caregiver_nearby). Если визит только «защёлкнут» по двери, а в кадре никого нет,
        падение пациента НЕ подавляется.
        """
        with self._lock:
            if caregiver_visible is None:
                caregiver_visible = bool(caregiver_nearby)
            dt = 0.2 if self._last_eval_ts is None else max(0.0, min(1.0, now - self._last_eval_ts))
            self._last_eval_ts = now
            hour = datetime.now().hour if local_hour is None else int(local_hour)
            eff_sit_confirm, eff_leave_confirm = self._effective_confirm_thresholds(hour)

            if patient_in_bed_latched and zones.has_bed:
                self._was_monitored_recently = True

            if now < self._paused_until:
                self._caregiver_active_until = max(self._caregiver_active_until, now + 8.0)
                self._state = PatientState.PAUSED
                self._status_reason = "ручная пауза («Я рядом»)"
                self._clear_away_tracking()
                self._clear_immobility_tracking()
                self._clear_person_down()
                self._unattended_fall_timer = 0.0
                return self._watch_fall_during_pause(observations, identities, zones, now, dt, bool(caregiver_visible))
            if self._state == PatientState.PAUSED:
                if zones.has_bed and (
                    patient_in_bed_latched
                    or (self._last_seen_deep_in_bed and zones.has_door and self._missing_confirm >= 8.0)
                ):
                    self._state = PatientState.IN_BED
                    self._status_reason = "спит в кровати · укрыта одеялом"
                    self._was_monitored_recently = True
                    self._last_seen_deep_in_bed = True
                else:
                    self._state = PatientState.IDLE
                    self._status_reason = "ожидание в кадре"

            # Страховка «человек на полу» — раньше любых подавлений по ролям и визитам
            down_ev = self._check_person_down(observations, identities, zones, now)
            if down_ev is not None:
                return down_ev
            if self._down_alerting:
                return None

            # Ищем наблюдение подтверждённой бабушки (Role.PATIENT)
            patient_obs: PoseObservation | None = None
            for obs in observations:
                ident = identities.get(obs.track_id)
                if ident is not None and ident.role == Role.PATIENT:
                    patient_obs = (
                        _sanitize_bed_pose_sm(
                            obs,
                            zones.bed_polygon,
                            other_observations=[x for x in observations if x.track_id != obs.track_id],
                            zones=zones,
                        )
                        if zones.has_bed
                        else obs
                    )
                    break

            # Случай 1: Бабушки нет в кадре (или временно перекрыта ухаживающим родственником у кровати / одеялом)
            if patient_obs is None:
                self._help_timer = 0.0
                self._sit_timer = 0.0
                self._leave_timer = 0.0
                self._fall_timer = 0.0
                self._clear_immobility_tracking()

                if caregiver_nearby and self._suppress_with_caregiver:
                    self._caregiver_active_until = max(self._caregiver_active_until, now + 8.0)
                    self._caregiver_was_present = True
                    self._missing_timer = 0.0
                    self._clear_away_tracking()
                    self._state = PatientState.CAREGIVER_ASSIST
                    self._status_reason = "рядом родственник · уход у кровати"
                    return None

                # Закон сохранения Бабушки в кровати (Волна 7):
                # если пациентка защёлкнута на матрасе (patient_in_bed_latched) или спала глубоко в кровати
                # при размеченной двери без фазы посадки/ухода, она НЕ могла телепортироваться из комнаты —
                # бессрочно удерживаем спокойный IN_BED («спит в кровати · укрыта одеялом») с 0 ложных MISSING!
                hold_covered_in_bed = zones.has_bed and (
                    (
                        patient_in_bed_latched
                        and self._state
                        not in (
                            PatientState.SITTING_UP,
                            PatientState.LEFT_BED,
                            PatientState.FALLEN,
                            PatientState.MISSING,
                        )
                    )
                    or (
                        self._last_seen_deep_in_bed
                        and zones.has_door
                        and self._missing_confirm >= 8.0
                    )
                )
                if hold_covered_in_bed:
                    self._missing_timer = 0.0
                    self._clear_away_tracking()
                    self._state = PatientState.IN_BED
                    self._status_reason = "спит в кровати · укрыта одеялом"
                    return None

                # Тревога MISSING срабатывает строго ОДИН РАЗ за эпизод реального ухода (после SITTING_UP / LEFT_BED).
                if self._was_monitored_recently and not self._caregiver_was_present and zones.has_bed:
                    eff_missing_confirm = (
                        (
                            max(60.0, self._missing_confirm * 5.0)
                            if self._missing_confirm >= 8.0
                            else self._missing_confirm * 2.2
                        )
                        if self._last_seen_deep_in_bed
                        else self._missing_confirm
                    )
                    self._missing_timer += dt
                    if self._missing_timer < eff_missing_confirm:
                        if self._last_seen_deep_in_bed and self._state == PatientState.IN_BED:
                            self._status_reason = "спит в кровати · укрыта одеялом"
                    else:
                        self._was_monitored_recently = False
                        self._last_seen_deep_in_bed = False
                        self._missing_timer = 0.0
                        self._state = PatientState.MISSING
                        self._status_reason = "нет в кровати и в секторе камеры"
                        if self._away_since is None:
                            self._away_since = now
                            self._prolonged_absence_alerted = False
                        return self._emit_if_cooled_down(
                            PatientState.MISSING,
                            AlertSeverity.WARNING,
                            "Бабушки нет в кровати и в кадре",
                            "Покинула сектор обзора камеры без сопровождения.",
                            now,
                            None,
                        )

                if self._state == PatientState.CAREGIVER_ASSIST:
                    self._state = PatientState.IDLE
                    self._status_reason = "ожидание в кадре"
                if self._state in (PatientState.MISSING, PatientState.LEFT_BED):
                    return self._check_prolonged_absence(now, None)
                return None

            # Бабушка в кадре — сбрасываем таймер пропажи и обновляем детектор микро-шевелений
            self._missing_timer = 0.0
            self._caregiver_was_present = caregiver_nearby
            self._update_micro_movement(patient_obs, now)

            upper_pt = patient_obs.upper_body_point
            pat_on_bed_step = zones.has_bed and is_person_on_bed(
                patient_obs, zones.bed_polygon, zones=zones
            )
            self._upper_y_history.append((now, upper_pt[1]))
            if patient_obs.mid_hip is not None:
                self._hip_y_history.append((now, patient_obs.mid_hip[1]))
            elif not pat_on_bed_step:
                self._hip_y_history.append((now, patient_obs.body_center[1]))

            # Приоритет №1: Если рядом помощник (Caregiver Co-Presence) — уход под присмотром, подавляем тревоги
            if caregiver_nearby and self._suppress_with_caregiver:
                self._caregiver_active_until = max(self._caregiver_active_until, now + 8.0)
                calm_seated, on_chair = self._classify_calm_seated_pose(patient_obs, zones, now)
                if calm_seated:
                    self._latch_or_update_assisted_seated(
                        patient_obs,
                        on_chair=on_chair,
                        solo_confirmed=False,
                    )
                elif (
                    zones.has_bed
                    and is_person_on_bed(patient_obs, zones.bed_polygon, zones=zones)
                    and not self._is_sitting_or_exiting_bed(patient_obs, zones)[0]
                    and upper_pt[1] < 0.64
                ):
                    self._clear_assisted_seated()

                # Визит «защёлкнут», но помощника в кадре нет: падение пациента не подавляем
                if not caregiver_visible:
                    lone_fallen, lone_reason = self._check_fallen_or_slumped(patient_obs, zones, now)
                    if lone_fallen:
                        self._unattended_fall_timer += dt
                        if self._unattended_fall_timer >= max(self._fall_confirm * 1.5, self._fall_confirm + 1.0):
                            self._clear_assisted_seated()
                            self._last_seen_deep_in_bed = False
                            prev = self._state
                            self._state = PatientState.FALLEN
                            self._status_reason = lone_reason.rstrip(".")
                            if prev != PatientState.FALLEN or not self._fall_unacknowledged:
                                return self._emit_if_cooled_down(
                                    PatientState.FALLEN,
                                    AlertSeverity.CRITICAL,
                                    "Падение / сползание вне кровати!",
                                    f"{lone_reason} Помощника в кадре нет. "
                                    "Нажмите «Я подошёл», чтобы отключить повтор сигнала.",
                                    now,
                                    patient_obs.track_id,
                                )
                            return None
                    else:
                        self._unattended_fall_timer = max(0.0, self._unattended_fall_timer - dt * 1.5)
                    if self._state == PatientState.FALLEN and self._fall_unacknowledged and lone_fallen:
                        return None
                else:
                    self._unattended_fall_timer = 0.0
                self._fall_timer = 0.0
                self._help_timer = 0.0
                self._sit_timer = 0.0
                self._leave_timer = 0.0
                self._missing_timer = 0.0
                self._fall_unacknowledged = False
                self._clear_away_tracking()
                self._clear_immobility_tracking()
                self._state = PatientState.CAREGIVER_ASSIST
                self._status_reason = "рядом родственник · уход у кровати"
                return None

            # Приоритет №2: ПАДЕНИЕ ИЛИ СПОЛЗАНИЕ НА ПОЛ (когда помощника рядом нет или подавление отключено)
            is_fallen, fall_reason = self._check_fallen_or_slumped(patient_obs, zones, now)
            if is_fallen:
                self._last_seen_deep_in_bed = False
                self._fall_timer += dt
                self._clear_immobility_tracking()
                needed_fall = max(0.8, self._fall_confirm * 0.55) if self._has_rapid_drop(now) else self._fall_confirm
                if self._fall_timer >= needed_fall:
                    self._clear_assisted_seated()
                    self._caregiver_active_until = -1e9
                    prev = self._state
                    self._state = PatientState.FALLEN
                    self._status_reason = fall_reason.rstrip(".")
                    if prev != PatientState.FALLEN or not self._fall_unacknowledged:
                        return self._emit_if_cooled_down(
                            PatientState.FALLEN,
                            AlertSeverity.CRITICAL,
                            "Падение / сползание вне кровати!",
                            f"{fall_reason} Нажмите «Я подошёл», чтобы отключить повтор сигнала.",
                            now,
                            patient_obs.track_id,
                        )
                return None
            else:
                self._fall_timer = max(0.0, self._fall_timer - dt * 1.5)

            # Взведение пост-визитного сидения в первые секунды после ухода родственника / снятия паузы
            if not self._assisted_seated_latched and now <= self._caregiver_active_until:
                calm_seated_after, on_chair_after = self._classify_calm_seated_pose(patient_obs, zones, now)
                if calm_seated_after:
                    self._latch_or_update_assisted_seated(
                        patient_obs,
                        on_chair=on_chair_after,
                        solo_confirmed=True,
                    )
                elif (
                    zones.has_bed
                    and is_person_on_bed(patient_obs, zones.bed_polygon, zones=zones)
                    and not self._is_sitting_or_exiting_bed(patient_obs, zones)[0]
                ):
                    self._caregiver_active_until = -1e9

            # Приоритет №3: Жест вызова помощи поднятой рукой (ДО SITTING_UP!)
            in_bed_area = (
                not zones.has_bed
                or is_person_on_bed(patient_obs, zones.bed_polygon, zones=zones)
                or are_legs_hanging_outside_bed(patient_obs, zones.bed_polygon, zones=zones)
                or self._assisted_seated_latched
            )
            bs_y = (
                self._assisted_seated_upper_y
                if (self._assisted_seated_latched and self._assisted_seated_on_chair)
                else (self._baseline.shoulder_y if self._baseline.calibrated else None)
            )
            hand_raised, help_reason = (
                is_hand_raised_for_help(patient_obs, baseline_shoulder_y=bs_y)
                if (self._enable_help_gesture and not caregiver_nearby and in_bed_area)
                else (False, "")
            )
            if hand_raised:
                self._was_monitored_recently = True
                self._help_timer += dt
                if self._help_timer >= self._help_confirm:
                    self._leave_timer = 0.0
                    prev = self._state
                    self._state = PatientState.HELP_CALL
                    self._status_reason = help_reason.lower()
                    if prev != PatientState.HELP_CALL:
                        return self._emit_if_cooled_down(
                            PatientState.HELP_CALL,
                            AlertSeverity.WARNING,
                            "Зовёт на помощь (поднята рука)",
                            f"{help_reason}. Проверьте бабушку.",
                            now,
                            patient_obs.track_id,
                        )
                    return None
            else:
                self._help_timer = max(0.0, self._help_timer - dt * 1.5)

            # Пост-визитное спокойное сидение («свой ушёл, а бабушка сидит и не упала»)
            sitting, sit_reason = self._is_sitting_or_exiting_bed(patient_obs, zones)
            on_bed_now = zones.has_bed and is_person_on_bed(patient_obs, zones.bed_polygon, zones=zones)
            if self._assisted_seated_latched:
                if on_bed_now and not sitting:
                    if not self._assisted_solo_confirmed:
                        self._clear_assisted_seated()
                        self._caregiver_active_until = -1e9
                    else:
                        self._reclining_clear_timer += dt
                        if self._reclining_clear_timer >= 4.0:
                            self._clear_assisted_seated()
                            self._caregiver_active_until = -1e9
                elif sitting:
                    self._assisted_solo_confirmed = True
                    self._reclining_clear_timer = 0.0
                    self._was_monitored_recently = True
                    self._last_seen_deep_in_bed = False
                    self._sit_timer = 0.0
                    self._leave_timer = 0.0
                    self._clear_away_tracking()
                    self._clear_immobility_tracking()
                    self._state = PatientState.IN_BED
                    self._status_reason = "сидит после ухода родственника"
                    self._fall_unacknowledged = False
                    return None
                else:
                    bc_now = patient_obs.body_center
                    dist_from_seat = math.hypot(
                        bc_now[0] - self._assisted_seated_center[0],
                        bc_now[1] - self._assisted_seated_center[1],
                    )
                    angle_now = patient_obs.torso_angle_from_vertical_deg()
                    upright_standing = self._is_upright_standing_pose(patient_obs, zones, now)
                    calmly_on_same_chair = (
                        not on_bed_now
                        and angle_now is not None
                        and angle_now < 42.0
                        and patient_obs.aspect_ratio_wh < 0.98
                        and upper_pt[1] < 0.65
                        and dist_from_seat <= 0.14
                        and not (
                            upright_standing
                            and patient_obs.bbox_height >= 0.52
                            and dist_from_seat > 0.10
                        )
                    )
                    if calmly_on_same_chair:
                        self._assisted_solo_confirmed = True
                        self._reclining_clear_timer = 0.0
                        self._was_monitored_recently = True
                        self._last_seen_deep_in_bed = False
                        self._sit_timer = 0.0
                        self._leave_timer = 0.0
                        self._clear_away_tracking()
                        self._clear_immobility_tracking()
                        self._state = PatientState.IN_BED
                        self._status_reason = "сидит после ухода родственника"
                        self._fall_unacknowledged = False
                        return None
                    self._clear_assisted_seated()
                    self._caregiver_active_until = -1e9

            # Приоритет №4: Посадка в кровати / свешивание ног (Упреждение!)
            if sitting:
                self._was_monitored_recently = True
                self._last_seen_deep_in_bed = False
                self._sit_timer += dt
                self._leave_timer = 0.0
                self._clear_immobility_tracking()
                if self._sit_timer >= eff_sit_confirm:
                    prev = self._state
                    self._state = PatientState.SITTING_UP
                    self._status_reason = sit_reason.lower()
                    if prev != PatientState.SITTING_UP:
                        return self._emit_if_cooled_down(
                            PatientState.SITTING_UP,
                            AlertSeverity.WARNING,
                            "Пытается встать с кровати",
                            sit_reason,
                            now,
                            patient_obs.track_id,
                        )
                return None
            else:
                self._sit_timer = max(0.0, self._sit_timer - dt)

            # Приоритет №5: Спокойно лежит в кровати (и калибрует эталон сна baseline_in_bed)
            if zones.has_bed and is_person_on_bed(patient_obs, zones.bed_polygon, zones=zones):
                self._was_monitored_recently = True
                bed_anchor_pt = patient_obs.mid_hip or patient_obs.upper_body_point
                self._last_seen_deep_in_bed = (
                    polygon_distance_norm(bed_anchor_pt, zones.bed_polygon) >= 0.025
                    and not are_legs_hanging_outside_bed(patient_obs, zones.bed_polygon, zones=zones)
                )
                if not hand_raised:
                    self._update_bed_baseline(patient_obs, zones, dt)
                self._leave_timer = 0.0
                self._clear_away_tracking()
                self._state = PatientState.IN_BED
                self._status_reason = (
                    "спит в кровати · эталон калиброван"
                    if self._baseline.calibrated
                    else "лежит в кровати · калибровка позы"
                )
                self._fall_unacknowledged = False
                return self._update_immobility_in_bed(patient_obs, now)

            # Приоритет №6: Стоит / идёт вне кровати одна
            if zones.has_bed:
                self._last_seen_deep_in_bed = False
                self._clear_immobility_tracking()
                self._leave_timer += dt
                if self._leave_timer >= eff_leave_confirm:
                    prev = self._state
                    self._state = PatientState.LEFT_BED
                    self._status_reason = "вышла из кровати без сопровождения"
                    if self._away_since is None:
                        self._away_since = now
                        self._prolonged_absence_alerted = False
                    if prev != PatientState.LEFT_BED:
                        ev = self._emit_if_cooled_down(
                            PatientState.LEFT_BED,
                            AlertSeverity.WARNING,
                            "Встала с кровати одна",
                            "Находится вне зоны кровати без сопровождения.",
                            now,
                            patient_obs.track_id,
                        )
                        if ev is not None:
                            return ev
                    return self._check_prolonged_absence(now, patient_obs.track_id)
            return None
