from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("fallguard.runtime_prefs")

SENSITIVITY_PRESETS: dict[str, tuple[float, float, float]] = {
    "high": (1.2, 1.4, 1.3),
    "standard": (1.8, 2.0, 1.8),
    "gentle": (2.8, 3.0, 2.2),
}

VALID_PRESETS = ("high", "standard", "gentle", "custom")


def _clamp_float(val: object, low: float, high: float, default: float) -> float:
    if isinstance(val, bool) or val is None:
        return default
    try:
        num = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if num != num:  # NaN check
        return default
    return round(max(low, min(high, num)), 2)


def _clamp_int(val: object, low: int, high: int, default: int) -> int:
    if isinstance(val, bool) or val is None:
        return default
    try:
        num = int(round(float(val)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(low, min(high, num))


def _clamp_optional_minutes(val: object, min_active: int, max_active: int, default: int) -> int:
    if isinstance(val, bool) or val is None:
        return default
    try:
        num = int(round(float(val)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if num <= 0:
        return 0
    return max(min_active, min(max_active, num))


def _parse_bool(val: object, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    return default


def _detect_matching_preset(sit: float, leave: float, fall: float) -> str:
    for name, (ps, pl, pf) in SENSITIVITY_PRESETS.items():
        if abs(sit - ps) < 1e-2 and abs(leave - pl) < 1e-2 and abs(fall - pf) < 1e-2:
            return name
    return "custom"


def default_runtime_prefs() -> dict[str, Any]:
    return {
        "sensitivity_preset": "standard",
        "sit_confirm_seconds": 1.8,
        "leave_confirm_seconds": 2.0,
        "fall_confirm_seconds": 1.8,
        "missing_confirm_seconds": 12.0,
        "enable_help_gesture": True,
        "help_confirm_seconds": 3.2,
        "prolonged_absence_minutes": 20,
        "immobility_reminder_minutes": 0,
        "night_strict_mode": True,
        "night_start_hour": 22,
        "night_end_hour": 7,
        "floor_safety_net": True,
    }


class RuntimePrefsStore:
    """
    Потокобезопасное хранилище живых настроек послеинсультного ухода (data/runtime_prefs.json)
    с клэмпингом диапазонов и синхронизацией пресетов чувствительности.
    """

    def __init__(self, path: Path | str, initial_defaults: dict[str, Any] | None = None) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        base = default_runtime_prefs()
        if initial_defaults:
            base = self._sanitize_full(base, initial_defaults)
        self._base_defaults = dict(base)
        self._prefs: dict[str, Any] = dict(base)
        self._load_locked()

    @property
    def path(self) -> Path:
        return self._path

    def _sanitize_full(self, current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        sit = _clamp_float(
            patch.get("sit_confirm_seconds", current["sit_confirm_seconds"]),
            0.8,
            6.0,
            float(current["sit_confirm_seconds"]),
        )
        leave = _clamp_float(
            patch.get("leave_confirm_seconds", current["leave_confirm_seconds"]),
            0.8,
            6.0,
            float(current["leave_confirm_seconds"]),
        )
        fall = _clamp_float(
            patch.get("fall_confirm_seconds", current["fall_confirm_seconds"]),
            0.8,
            5.0,
            float(current["fall_confirm_seconds"]),
        )

        timers_changed = (
            abs(sit - float(current["sit_confirm_seconds"])) > 1e-3
            or abs(leave - float(current["leave_confirm_seconds"])) > 1e-3
            or abs(fall - float(current["fall_confirm_seconds"])) > 1e-3
        )

        raw_preset = patch.get("sensitivity_preset")
        preset_str = str(raw_preset).strip().lower() if isinstance(raw_preset, str) else None

        if preset_str in SENSITIVITY_PRESETS:
            preset_switched = preset_str != current.get("sensitivity_preset")
            timers_explicit = any(
                k in patch
                for k in ("sit_confirm_seconds", "leave_confirm_seconds", "fall_confirm_seconds")
            )
            if preset_switched or not timers_explicit or not timers_changed:
                sit, leave, fall = SENSITIVITY_PRESETS[preset_str]
                final_preset = preset_str
            else:
                final_preset = _detect_matching_preset(sit, leave, fall)
        elif preset_str == "custom":
            final_preset = "custom"
        elif timers_changed:
            final_preset = _detect_matching_preset(sit, leave, fall)
        else:
            cur_p = str(current.get("sensitivity_preset", "standard"))
            final_preset = cur_p if cur_p in VALID_PRESETS else _detect_matching_preset(sit, leave, fall)

        missing = _clamp_float(
            patch.get("missing_confirm_seconds", current["missing_confirm_seconds"]),
            5.0,
            60.0,
            float(current["missing_confirm_seconds"]),
        )
        enable_help = _parse_bool(
            patch.get("enable_help_gesture", current["enable_help_gesture"]),
            bool(current["enable_help_gesture"]),
        )
        help_sec = _clamp_float(
            patch.get("help_confirm_seconds", current["help_confirm_seconds"]),
            1.5,
            8.0,
            float(current["help_confirm_seconds"]),
        )
        prolonged_min = _clamp_optional_minutes(
            patch.get("prolonged_absence_minutes", current["prolonged_absence_minutes"]),
            5,
            180,
            int(current["prolonged_absence_minutes"]),
        )
        immobility_min = _clamp_optional_minutes(
            patch.get("immobility_reminder_minutes", current["immobility_reminder_minutes"]),
            30,
            480,
            int(current["immobility_reminder_minutes"]),
        )
        night_strict = _parse_bool(
            patch.get("night_strict_mode", current["night_strict_mode"]),
            bool(current["night_strict_mode"]),
        )
        night_start = _clamp_int(
            patch.get("night_start_hour", current["night_start_hour"]),
            0,
            23,
            int(current["night_start_hour"]),
        )
        night_end = _clamp_int(
            patch.get("night_end_hour", current["night_end_hour"]),
            0,
            23,
            int(current["night_end_hour"]),
        )

        floor_net = _parse_bool(
            patch.get("floor_safety_net", current.get("floor_safety_net", True)),
            bool(current.get("floor_safety_net", True)),
        )

        return {
            "floor_safety_net": floor_net,
            "sensitivity_preset": final_preset,
            "sit_confirm_seconds": sit,
            "leave_confirm_seconds": leave,
            "fall_confirm_seconds": fall,
            "missing_confirm_seconds": missing,
            "enable_help_gesture": enable_help,
            "help_confirm_seconds": help_sec,
            "prolonged_absence_minutes": prolonged_min,
            "immobility_reminder_minutes": immobility_min,
            "night_strict_mode": night_strict,
            "night_start_hour": night_start,
            "night_end_hour": night_end,
        }

    def _load_locked(self) -> None:
        with self._lock:
            if not self._path.exists():
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    # При загрузке с диска уважаем сохранённые таймеры без перезаписи пресетом
                    loaded_base = dict(self._base_defaults)
                    if isinstance(raw.get("sensitivity_preset"), str):
                        loaded_base["sensitivity_preset"] = raw["sensitivity_preset"].strip().lower()
                    self._prefs = self._sanitize_full(loaded_base, raw)
            except Exception as exc:
                logger.warning("Не удалось прочитать %s: %s", self._path, exc)

    def _save_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_name(
                f"{self._path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
            )
            tmp_path.write_text(
                json.dumps(self._prefs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._path)
        except OSError as exc:
            logger.warning("Не удалось сохранить %s: %s", self._path, exc)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._prefs)

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(patch, dict):
            return self.get()
        with self._lock:
            self._prefs = self._sanitize_full(self._prefs, patch)
            self._save_locked()
            return dict(self._prefs)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._prefs = dict(self._base_defaults)
            self._save_locked()
            return dict(self._prefs)
