from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Исключаем busy-spin потоков OpenMP/BLAS на CPU до импорта numpy/cv2/onnxruntime
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


class ConfigError(RuntimeError):
    """Ошибка отсутствия или некорректности обязательных параметров окружения."""


def _parse_dotenv_file(dotenv_path: Path) -> dict[str, str]:
    if not dotenv_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded[key.strip()] = value.strip().strip('"').strip("'")
    return loaded


def _is_placeholder(val: str) -> bool:
    low = val.strip().lower()
    return (
        not low
        or low.startswith("your_")
        or low.startswith("<")
        or "changeme" in low
        or "example" in low
    )


def _normalize_rotate(val: int) -> int:
    norm = val % 360
    if norm in (0, 90, 180, 270):
        return norm
    return 180


def _normalize_hwaccel(val: str) -> str:
    norm = (val or "").strip().lower()
    if norm in ("auto", "nvdec", "off"):
        return norm
    return "auto"


def _normalize_video_encoder(val: str) -> str:
    norm = (val or "").strip().lower()
    if norm in ("auto", "nvenc", "off"):
        return norm
    if norm in ("cpu", "libx264", "x264"):
        return "off"
    return "auto"


@dataclass(frozen=True)
class Settings:
    rtsp_url: str
    tg_bot_token: str
    tg_chat_ids: tuple[str, ...]
    data_dir: Path
    models_dir: Path
    web_host: str
    web_port: int
    infer_fps: float
    capture_fps: float
    pre_event_seconds: float
    post_event_seconds: float
    pose_model: str
    device: str
    pose_conf: float
    bed_lock_seconds: float
    initial_bed_lock_seconds: float
    sit_confirm_seconds: float
    leave_confirm_seconds: float
    fall_confirm_seconds: float
    missing_confirm_seconds: float
    alert_cooldown_seconds: float
    repeat_fall_alert_seconds: float
    max_fall_repeats: int
    suppress_sitting_when_caregiver: bool
    face_match_threshold: float
    tg_bot_username: str = "NoFallenAi"
    public_webapp_url: str = ""
    frame_rotate: int = 180
    frame_flip_h: bool = False
    low_light_enhance: bool = True
    rtsp_hwaccel: str = "auto"
    stream_h264_enabled: bool = True
    video_encoder: str = "auto"
    door_visit_idle_timeout_sec: float = 14400.0

    @property
    def camera_configured(self) -> bool:
        return not _is_placeholder(self.rtsp_url)

    @property
    def telegram_configured(self) -> bool:
        return not _is_placeholder(self.tg_bot_token)

    def missing_required_keys(self) -> list[str]:
        missing: list[str] = []
        if not self.camera_configured:
            missing.append("RTSP_URL")
        if not self.telegram_configured:
            missing.append("TG_BOT_TOKEN")
        return missing

    def require_all_secrets(self) -> None:
        missing = self.missing_required_keys()
        if missing:
            raise ConfigError(
                f"Не заданы обязательные переменные окружения в .env: {', '.join(missing)}"
            )

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> Settings:
        base_dir = Path(__file__).resolve().parent.parent
        env_file = dotenv_path if dotenv_path is not None else (base_dir / ".env")
        file_vars = _parse_dotenv_file(env_file)

        def get_raw(key: str) -> str:
            if key in os.environ:
                return os.environ[key].strip()
            return file_vars.get(key, "").strip()

        def get_float(key: str, default: float) -> float:
            raw = get_raw(key)
            return float(raw) if raw else default

        def get_int(key: str, default: int) -> int:
            raw = get_raw(key)
            return int(raw) if raw else default

        def get_bool(key: str, default: bool) -> bool:
            raw = get_raw(key).lower()
            if not raw:
                return default
            return raw in ("1", "true", "yes", "on")

        raw_chats = get_raw("TG_CHAT_IDS")
        chat_ids = tuple(
            cid.strip()
            for cid in raw_chats.replace(";", ",").split(",")
            if cid.strip() and not _is_placeholder(cid)
        )

        bot_username = get_raw("TG_BOT_USERNAME").lstrip("@").strip() or "NoFallenAi"
        public_webapp_url = get_raw("PUBLIC_WEBAPP_URL").strip().rstrip("/")
        frame_rotate = _normalize_rotate(get_int("FRAME_ROTATE", 180))
        frame_flip_h = get_bool("FRAME_FLIP_H", False)
        low_light_enhance = get_bool("LOW_LIGHT_ENHANCE", True)
        rtsp_hwaccel = _normalize_hwaccel(get_raw("RTSP_HWACCEL"))
        stream_h264_enabled = get_bool("STREAM_H264_ENABLED", True)
        raw_venc = get_raw("VIDEO_ENCODER")
        video_encoder = (
            _normalize_video_encoder(raw_venc)
            if raw_venc
            else ("off" if rtsp_hwaccel == "off" else "auto")
        )
        door_visit_idle_timeout_sec = max(
            1.0,
            get_float(
                "DOOR_VISIT_IDLE_TIMEOUT_SECONDS",
                get_float("DOOR_VISIT_IDLE_TIMEOUT_SEC", 14400.0),
            ),
        )

        data_dir = Path(get_raw("DATA_DIR") or str(base_dir / "data"))
        models_dir = Path(get_raw("MODELS_DIR") or str(data_dir / "models"))

        return cls(
            rtsp_url=get_raw("RTSP_URL"),
            tg_bot_token=get_raw("TG_BOT_TOKEN"),
            tg_chat_ids=chat_ids,
            data_dir=data_dir,
            models_dir=models_dir,
            web_host=get_raw("WEB_HOST") or "0.0.0.0",
            web_port=get_int("WEB_PORT", 8090),
            infer_fps=get_float("INFER_FPS", 5.0),
            capture_fps=get_float("CAPTURE_FPS", 6.0),
            pre_event_seconds=get_float("PRE_EVENT_SECONDS", 5.0),
            post_event_seconds=get_float("POST_EVENT_SECONDS", 4.0),
            pose_model=get_raw("POSE_MODEL") or "yolo11s-pose.pt",
            device=get_raw("DEVICE") or "auto",
            pose_conf=get_float("POSE_CONF", 0.48),
            bed_lock_seconds=get_float("BED_LOCK_SECONDS", 20.0),
            initial_bed_lock_seconds=get_float("INITIAL_BED_LOCK_SECONDS", 5.0),
            sit_confirm_seconds=get_float("SIT_CONFIRM_SECONDS", 1.8),
            leave_confirm_seconds=get_float("LEAVE_CONFIRM_SECONDS", 2.0),
            fall_confirm_seconds=get_float("FALL_CONFIRM_SECONDS", 1.8),
            missing_confirm_seconds=get_float("MISSING_CONFIRM_SECONDS", 12.0),
            alert_cooldown_seconds=get_float("ALERT_COOLDOWN_SECONDS", 45.0),
            repeat_fall_alert_seconds=get_float("REPEAT_FALL_ALERT_SECONDS", 15.0),
            max_fall_repeats=get_int("MAX_FALL_REPEATS", 20),
            suppress_sitting_when_caregiver=get_bool("SUPPRESS_SITTING_WHEN_CAREGIVER", True),
            face_match_threshold=get_float("FACE_MATCH_THRESHOLD", 0.38),
            tg_bot_username=bot_username,
            public_webapp_url=public_webapp_url,
            frame_rotate=frame_rotate,
            frame_flip_h=frame_flip_h,
            low_light_enhance=low_light_enhance,
            rtsp_hwaccel=rtsp_hwaccel,
            stream_h264_enabled=stream_h264_enabled,
            video_encoder=video_encoder,
            door_visit_idle_timeout_sec=door_visit_idle_timeout_sec,
        )
