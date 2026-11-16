from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import cv2
import numpy as np

logger = logging.getLogger("fallguard.capture")

VALID_ROTATIONS = (0, 90, 180, 270)
VALID_HWACCEL_MODES = ("auto", "nvdec", "off")
LOW_LIGHT_LUMA_THRESHOLD = 55.0
MAX_CAPTURE_LONG_SIDE = 1920
WINDOWS_FALLGUARD_FFMPEG = Path(r"C:\FallGuard\bin\ffmpeg.exe")
CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)

BACKEND_NVDEC = "NVDEC (h264_cuvid)"
BACKEND_CPU = "CPU (FFmpeg)"

# Подавляем некритичный спам декодера FFmpeg при старте потока до прихода ключевого IDR-кадра
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

_RTSP_CRED_RE = re.compile(r"(rtsps?://)([^/@\s]+@)", re.IGNORECASE)


def _redact_url(text: str) -> str:
    """Маскирует логин/пароль RTSP в любых диагностических сообщениях."""
    return _RTSP_CRED_RE.sub(r"\1***@", text)


def find_ffmpeg_bin() -> str | None:
    """Находит исполняемый файл ffmpeg через FFMPEG_BIN, PATH или C:\\FallGuard\\bin\\ffmpeg.exe."""
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    found = shutil.which("ffmpeg")
    if found:
        return found
    if WINDOWS_FALLGUARD_FFMPEG.is_file():
        return str(WINDOWS_FALLGUARD_FFMPEG)
    return None


def _is_rtsp_url(url: str) -> bool:
    low = url.strip().lower()
    return (not url.strip().isdigit()) and (low.startswith("rtsp://") or low.startswith("rtsps://"))


def _normalize_hwaccel(value: object) -> str:
    cleaned = str(value or "auto").strip().lower()
    if cleaned in VALID_HWACCEL_MODES:
        return cleaned
    return "auto"


def _normalize_rotate(value: object, fallback: int = 0) -> int:
    """Приводит угол поворота к одному из допустимых значений {0, 90, 180, 270}."""
    if isinstance(value, bool):
        return fallback
    try:
        angle = int(value) % 360  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    if angle in VALID_ROTATIONS:
        return angle
    snapped = (int(round(angle / 90.0)) * 90) % 360
    return snapped if snapped in VALID_ROTATIONS else fallback


def _build_cuvid_ffmpeg_cmd(ffmpeg_bin: str, url: str, pix_fmt: str = "nv12") -> list[str]:
    return [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-c:v",
        "h264_cuvid",
        "-timeout",
        "5000000",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-an",
        "-sn",
        "pipe:1",
    ]


@dataclass(frozen=True)
class BufferedFrame:
    timestamp: float
    jpeg_bytes: bytes


class _CuvidSession:
    """Управляет дочерним процессом FFmpeg h264_cuvid (NV12/BGR24), дренированием stderr и watchdog-таймаутом."""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        width: int,
        height: int,
        pix_fmt: str = "nv12",
        read_timeout_sec: float = 12.0,
    ) -> None:
        self.proc = proc
        self.width = int(width)
        self.height = int(height)
        self.pix_fmt = pix_fmt.lower()
        if self.pix_fmt == "nv12":
            self.frame_nbytes = (self.width * self.height * 3) // 2
        else:
            self.frame_nbytes = self.width * self.height * 3
        self._buf = bytearray(self.frame_nbytes)
        self.stderr_tail: deque[str] = deque(maxlen=20)
        self._closed = threading.Event()
        self._last_io_mono = time.monotonic()
        self._read_timeout_sec = max(3.0, float(read_timeout_sec))

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="ffmpeg-cuvid-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="ffmpeg-cuvid-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _drain_stderr(self) -> None:
        err_stream = self.proc.stderr
        if err_stream is None:
            return
        try:
            while not self._closed.is_set():
                line = err_stream.readline()
                if not line:
                    break
                decoded = _redact_url(line.decode("utf-8", errors="replace").strip())
                if decoded:
                    self.stderr_tail.append(decoded)
        except Exception:
            pass

    def _watchdog_loop(self) -> None:
        while not self._closed.wait(1.0):
            if self.proc.poll() is not None:
                return
            if (time.monotonic() - self._last_io_mono) > self._read_timeout_sec:
                logger.warning("Таймаут чтения кадра FFmpeg CUVID (>%.1f с), принудительный рестарт", self._read_timeout_sec)
                self.close()
                return

    def read_raw_packet(self, abort_event_1: threading.Event, abort_event_2: threading.Event) -> bool:
        """Вычитывает ровно один сырой кадр из stdout в self._buf без аллокаций памяти."""
        stdout: IO[bytes] | None = self.proc.stdout
        if stdout is None or self._closed.is_set():
            return False

        mv = memoryview(self._buf)
        offset = 0
        total = self.frame_nbytes
        try:
            while offset < total:
                if abort_event_1.is_set() or abort_event_2.is_set() or self._closed.is_set():
                    return False
                read_fn = getattr(stdout, "readinto", None)
                if callable(read_fn):
                    n = read_fn(mv[offset:])
                else:
                    chunk = stdout.read(total - offset)
                    if not chunk:
                        return False
                    n = len(chunk)
                    mv[offset : offset + n] = chunk
                if not n:
                    return False
                offset += int(n)
                self._last_io_mono = time.monotonic()
        except Exception:
            return False
        return True

    def decode_current_bgr(self) -> np.ndarray:
        """Преобразует текущий буфер (NV12 или BGR24) в BGR-кадр OpenCV."""
        if self.pix_fmt == "nv12":
            nv12_plane = np.frombuffer(self._buf, dtype=np.uint8).reshape(
                ((self.height * 3) // 2, self.width)
            )
            return cv2.cvtColor(nv12_plane, cv2.COLOR_YUV2BGR_NV12)
        arr = np.frombuffer(self._buf, dtype=np.uint8).reshape((self.height, self.width, 3))
        return arr.copy()

    def read_frame(self, abort_event_1: threading.Event, abort_event_2: threading.Event) -> np.ndarray | None:
        if not self.read_raw_packet(abort_event_1, abort_event_2):
            return None
        return self.decode_current_bgr()

    def last_error_hint(self) -> str:
        if self.stderr_tail:
            return self.stderr_tail[-1]
        rc = self.proc.poll()
        return f"код возврата FFmpeg={rc}" if rc is not None else "обрыв пайпа FFmpeg"

    def close(self) -> None:
        self._closed.set()
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        try:
            self.proc.wait(timeout=2.0)
        except Exception:
            pass


class VideoStreamCapture:
    """
    Потоковый захват RTSP с аппаратным ускорением NVDEC (h264_cuvid) и бесшовным
    автоматическим откатом на CPU (cv2.VideoCapture TCP -> UDP), защитой от битых
    H.264/HEVC референс-кадров, трансформацией кадра на лету и кольцевым буфером предзаписи.
    """

    def __init__(
        self,
        rtsp_url: str,
        capture_fps: float = 6.0,
        pre_event_seconds: float = 5.0,
        settings_path: Path | None = None,
        default_rotate: int = 0,
        default_flip_h: bool = False,
        default_enhance: bool = False,
        hwaccel: str = "auto",
    ) -> None:
        self._rtsp_url = rtsp_url.strip()
        self._capture_fps = max(2.0, float(capture_fps))
        self._max_buffer = max(12, int(self._capture_fps * (pre_event_seconds + 2.0)))
        self._settings_path: Path | None = Path(settings_path) if settings_path is not None else None
        self._hwaccel: str = _normalize_hwaccel(hwaccel)

        self._lock = threading.RLock()
        self._proc_lock = threading.Lock()
        self._rotate: int = _normalize_rotate(default_rotate, fallback=0)
        self._flip_h: bool = bool(default_flip_h)
        self._enhance: bool = bool(default_enhance)
        self._load_transform_settings()

        self._raw_base_frame: np.ndarray | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._resolution: list[int] = [0, 0]
        self._active_transport: str = "tcp"
        self._decoder_backend: str = BACKEND_CPU
        self._ring: deque[BufferedFrame] = deque(maxlen=self._max_buffer)

        self._cuvid_session: _CuvidSession | None = None
        self._gpu_available_cache: bool | None = None

        self._connected = False
        self._last_error: str = ""
        self._started = False
        self._stop_event = threading.Event()
        self._reload_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _load_transform_settings(self) -> None:
        if self._settings_path is None or not self._settings_path.exists():
            return
        try:
            raw = json.loads(self._settings_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if "rotate" in raw:
                    self._rotate = _normalize_rotate(raw["rotate"], fallback=self._rotate)
                if "flip_h" in raw:
                    self._flip_h = bool(raw["flip_h"])
                if "enhance" in raw:
                    self._enhance = bool(raw["enhance"])
        except Exception as exc:
            logger.warning("Не удалось прочитать настройки камеры из %s: %s", self._settings_path, exc)

    def _save_transform_settings(self) -> None:
        if self._settings_path is None:
            return
        payload = {
            "rotate": self._rotate,
            "flip_h": self._flip_h,
            "enhance": self._enhance,
        }
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._settings_path.with_suffix(self._settings_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(self._settings_path)
        except OSError as exc:
            logger.warning("Не удалось сохранить настройки камеры в %s: %s", self._settings_path, exc)

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected and (time.monotonic() - self._latest_ts) < 10.0

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def active_transport(self) -> str:
        with self._lock:
            return self._active_transport

    @property
    def decoder_backend(self) -> str:
        with self._lock:
            return self._decoder_backend

    @property
    def hwaccel(self) -> str:
        with self._lock:
            return self._hwaccel

    @property
    def seconds_since_last_frame(self) -> float:
        with self._lock:
            if self._latest_ts <= 0.0:
                return 1e9
            return max(0.0, time.monotonic() - self._latest_ts)

    def get_transform(self) -> dict:
        """Возвращает текущие параметры трансформации кадра и актуальное разрешение [w, h]."""
        with self._lock:
            return {
                "rotate": self._rotate,
                "flip_h": self._flip_h,
                "enhance": self._enhance,
                "resolution": list(self._resolution),
            }

    def set_transform(
        self,
        rotate: int | None = None,
        flip_h: bool | None = None,
        enhance: bool | None = None,
    ) -> dict:
        """
        Мгновенно меняет ориентацию и улучшение кадра на лету,
        сохраняет настройки в settings_path и пересчитывает последний кадр.
        """
        with self._lock:
            if rotate is not None:
                self._rotate = _normalize_rotate(rotate, fallback=self._rotate)
            if flip_h is not None:
                self._flip_h = bool(flip_h)
            if enhance is not None:
                self._enhance = bool(enhance)

            if self._raw_base_frame is not None:
                transformed = self._apply_transform(
                    self._raw_base_frame,
                    rotate=self._rotate,
                    flip_h=self._flip_h,
                    enhance=self._enhance,
                )
                self._latest_frame = transformed
                self._resolution = [int(transformed.shape[1]), int(transformed.shape[0])]
                self._latest_ts = time.monotonic()

            self._save_transform_settings()
            return self.get_transform()

    def _close_active_cuvid(self) -> None:
        with self._proc_lock:
            sess = self._cuvid_session
            self._cuvid_session = None
        if sess is not None:
            sess.close()
        with self._lock:
            self._decoder_backend = BACKEND_CPU

    def reload_url(self, new_rtsp_url: str) -> None:
        """Безопасно переключает RTSP URL на лету и инициирует переподключение."""
        cleaned = new_rtsp_url.strip()
        with self._lock:
            self._rtsp_url = cleaned
            self._connected = False
            self._last_error = "" if cleaned else "RTSP_URL не задан в .env"
            self._reload_event.set()
            should_spawn = (
                self._started
                and bool(cleaned)
                and not self._stop_event.is_set()
                and (self._thread is None or not self._thread.is_alive())
            )
        self._close_active_cuvid()
        if should_spawn:
            self.start()

    def start(self) -> None:
        with self._lock:
            self._started = True
            if not self._rtsp_url:
                self._last_error = "RTSP_URL не задан в .env"
                return
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._reload_event.clear()
            self._thread = threading.Thread(target=self._loop, name="rtsp-capture", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._reload_event.set()
        self._close_active_cuvid()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=4.0)

    def get_latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_ts

    def push_annotated_to_ring(self, annotated_bgr: np.ndarray, ts: float) -> None:
        """Сохраняет уже размеченный кадр в кольцевой RAM-буфер для склейки MP4 со скелетом."""
        h, w = annotated_bgr.shape[:2]
        work = annotated_bgr
        if w > 960:
            scale = 960.0 / float(w)
            work = cv2.resize(annotated_bgr, (960, int(h * scale)))
        ok, enc = cv2.imencode(".jpg", work, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            if self._ring and (ts - self._ring[-1].timestamp) < (0.85 / self._capture_fps):
                return
            self._ring.append(BufferedFrame(timestamp=ts, jpeg_bytes=enc.tobytes()))

    def snapshot_ring_since(self, min_ts: float) -> list[BufferedFrame]:
        with self._lock:
            return [bf for bf in self._ring if bf.timestamp >= min_ts]

    def _find_ffmpeg_bin(self) -> str | None:
        env_bin = os.environ.get("FFMPEG_BIN", "").strip()
        if env_bin and Path(env_bin).is_file():
            return env_bin
        found = shutil.which("ffmpeg")
        if found:
            return found
        if WINDOWS_FALLGUARD_FFMPEG.is_file():
            return str(WINDOWS_FALLGUARD_FFMPEG)
        return None

    def _has_nvidia_gpu(self) -> bool:
        if self._gpu_available_cache is not None:
            return self._gpu_available_cache

        detected = False
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            try:
                cuda_mod = getattr(torch_mod, "cuda", None)
                if cuda_mod is not None and bool(cuda_mod.is_available()):
                    detected = True
            except Exception:
                pass

        if not detected:
            smi_bin = shutil.which("nvidia-smi")
            if not smi_bin and os.name == "nt":
                win_smi = Path(r"C:\Windows\System32\nvidia-smi.exe")
                if win_smi.is_file():
                    smi_bin = str(win_smi)
            if smi_bin:
                try:
                    res = subprocess.run(
                        [smi_bin, "-L"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        timeout=2.5,
                        check=False,
                        creationflags=CREATE_NO_WINDOW,
                    )
                    if res.returncode == 0 and b"GPU " in res.stdout:
                        detected = True
                except Exception:
                    pass

        self._gpu_available_cache = detected
        return detected

    def _probe_rtsp_dimensions(self, url: str, ffmpeg_bin: str) -> tuple[int, int] | None:
        """
        Определяет ширину и высоту видеопотока один раз при подключении:
        сначала через ffprobe (если доступен), иначе пробным чтением одного кадра cv2.VideoCapture.
        """
        ff_path = Path(ffmpeg_bin)
        probe_candidates: list[str] = []
        sibling_name = "ffprobe.exe" if ff_path.suffix.lower() == ".exe" else "ffprobe"
        sibling = ff_path.with_name(sibling_name)
        if sibling.is_file():
            probe_candidates.append(str(sibling))
        which_probe = shutil.which("ffprobe")
        if which_probe and which_probe not in probe_candidates:
            probe_candidates.append(which_probe)

        for probe_bin in probe_candidates:
            if self._stop_event.is_set() or self._reload_event.is_set():
                return None
            try:
                res = subprocess.run(
                    [
                        probe_bin,
                        "-v",
                        "error",
                        "-timeout",
                        "5000000",
                        "-rtsp_transport",
                        "tcp",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height,codec_name",
                        "-of",
                        "json",
                        url,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    check=False,
                    creationflags=CREATE_NO_WINDOW,
                )
                if res.returncode == 0 and res.stdout:
                    info = json.loads(res.stdout.decode("utf-8", errors="ignore"))
                    streams = info.get("streams") or []
                    if streams:
                        codec = str(streams[0].get("codec_name") or "").lower()
                        if codec and codec not in ("h264", "avc", "avc1"):
                            logger.info(
                                "Поток камеры имеет кодек %s (не H.264) — используем штатный декодер",
                                codec,
                            )
                            return None
                        w = int(streams[0].get("width") or 0)
                        h = int(streams[0].get("height") or 0)
                        if w >= 32 and h >= 32:
                            return w, h
            except Exception:
                pass

        # Запасной способ определения размеров через короткое пробное открытие cv2.VideoCapture
        cap = self._open_with_transport(url, "tcp")
        try:
            if not cap.isOpened():
                return None
            for _ in range(8):
                if self._stop_event.is_set() or self._reload_event.is_set():
                    return None
                ok, frm = cap.read()
                if ok and self._is_usable_frame(frm):
                    return int(frm.shape[1]), int(frm.shape[0])
        finally:
            cap.release()
        return None

    def _open_cuvid_session(self, url: str) -> tuple[_CuvidSession | None, np.ndarray | None]:
        ffmpeg_bin = self._find_ffmpeg_bin()
        if not ffmpeg_bin:
            return None, None

        dims = self._probe_rtsp_dimensions(url, ffmpeg_bin)
        if dims is None:
            return None, None
        width, height = dims
        pix_fmt = "nv12" if (width % 2 == 0 and height % 2 == 0) else "bgr24"

        cmd = _build_cuvid_ffmpeg_cmd(ffmpeg_bin, url, pix_fmt=pix_fmt)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception as exc:
            logger.warning("Не удалось запустить процесс FFmpeg CUVID: %s", exc)
            return None, None

        session = _CuvidSession(proc=proc, width=width, height=height, pix_fmt=pix_fmt)
        with self._proc_lock:
            self._cuvid_session = session

        first_frame: np.ndarray | None = None
        for _ in range(12):
            if self._stop_event.is_set() or self._reload_event.is_set():
                break
            frm = session.read_frame(self._stop_event, self._reload_event)
            if frm is None:
                break
            if self._is_usable_frame(frm):
                first_frame = frm
                break

        if first_frame is None:
            hint = session.last_error_hint()
            logger.warning("FFmpeg h264_cuvid не выдал стартовый кадр (%s)", hint)
            self._close_active_cuvid()
            return None, None

        with self._lock:
            self._decoder_backend = BACKEND_NVDEC
            self._active_transport = "tcp"
        logger.info("Видеопоток открыт через %s (%dx%d)", BACKEND_NVDEC, width, height)
        return session, first_frame

    @staticmethod
    def _normalize_frame_size(frame: np.ndarray) -> np.ndarray:
        """Ограничивает сверхвысокие потоки (>1920px по длинной стороне) для экономии CPU и RAM."""
        h, w = frame.shape[:2]
        long_side = max(w, h)
        if long_side <= MAX_CAPTURE_LONG_SIDE:
            return frame
        scale = float(MAX_CAPTURE_LONG_SIDE) / float(long_side)
        new_w = max(2, (int(round(w * scale)) // 2) * 2)
        new_h = max(2, (int(round(h * scale)) // 2) * 2)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _apply_transform(
        frame: np.ndarray,
        rotate: int,
        flip_h: bool,
        enhance: bool,
    ) -> np.ndarray:
        """
        Применяет поворот (0/90/180/270), зеркальное отражение и мягкое адаптивное
        вытягивание теней CLAHE в LAB при enhance=True или низкой яркости (< 55).
        """
        out = frame
        if rotate == 90:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            out = cv2.rotate(out, cv2.ROTATE_180)
        elif rotate == 270:
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if flip_h:
            out = cv2.flip(out, 1)

        need_clahe = enhance
        if not need_clahe:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            mean_luma = float(cv2.mean(gray)[0])
            if mean_luma < LOW_LIGHT_LUMA_THRESHOLD:
                need_clahe = True

        if need_clahe:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_eq = clahe.apply(l_ch)
            out = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        return out

    @staticmethod
    def _is_usable_frame(frame: np.ndarray | None) -> bool:
        """Отсекает пустые пакеты и повреждённые H.264/HEVC буферы при потере референс-кадра."""
        if frame is None or not isinstance(frame, np.ndarray):
            return False
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            return False
        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return False
        bottom = frame[int(h * 0.75) :, ::4, :]
        top = frame[: int(h * 0.25), ::4, :]
        if bottom.shape[0] >= 4 and top.shape[0] >= 4:
            row_diff_bottom = float(np.mean(np.abs(np.diff(bottom.astype(np.int16), axis=0))))
            row_diff_top = float(np.mean(np.abs(top.astype(np.int16)[1:] - top.astype(np.int16)[:-1])))
            if row_diff_top > 4.0 and row_diff_bottom < 0.05:
                mean_b, mean_g, mean_r = (
                    float(np.mean(bottom[:, :, 0])),
                    float(np.mean(bottom[:, :, 1])),
                    float(np.mean(bottom[:, :, 2])),
                )
                is_corrupt_green = mean_g > 110.0 and mean_r < 25.0 and mean_b < 25.0
                is_corrupt_gray = (
                    abs(mean_r - 128.0) < 4.0
                    and abs(mean_g - 128.0) < 4.0
                    and abs(mean_b - 128.0) < 4.0
                )
                if is_corrupt_green or is_corrupt_gray:
                    return False
        return True

    def _open_with_transport(self, url: str, transport: str) -> cv2.VideoCapture:
        if url.isdigit():
            cap = cv2.VideoCapture(int(url), cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        is_rtsp = _is_rtsp_url(url)
        if is_rtsp:
            if transport == "udp":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;udp|stimeout;5000000|max_delay;500000"
                    "|fflags;nobuffer+discardcorrupt|flags;low_delay"
                )
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|stimeout;5000000"
                    "|fflags;nobuffer+discardcorrupt|flags;low_delay"
                )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _connect_auto_transport(self, url: str) -> tuple[cv2.VideoCapture | None, np.ndarray | None, str]:
        """
        Пробует открыть поток сначала поверх TCP, а при отказе или пустом RTP-потоке —
        автоматически переключается на UDP и дожидается первого чистого опорного кадра.
        """
        is_rtsp = _is_rtsp_url(url)
        transports = ("tcp", "udp") if is_rtsp else ("file",)

        for transport in transports:
            if self._stop_event.is_set() or self._reload_event.is_set():
                return None, None, transport
            cap = self._open_with_transport(url, transport)
            if not cap.isOpened():
                cap.release()
                continue

            first_frame: np.ndarray | None = None
            for _ in range(12):
                if self._stop_event.is_set() or self._reload_event.is_set():
                    break
                ok, frm = cap.read()
                if ok and self._is_usable_frame(frm):
                    first_frame = frm
                    break
            if first_frame is not None:
                logger.info("Видеопоток открыт через %s (транспорт=%s)", BACKEND_CPU, transport)
                return cap, first_frame, transport

            cap.release()

        return None, None, transports[0]

    def _store_fresh_frame(self, raw_frame: np.ndarray) -> None:
        base = self._normalize_frame_size(raw_frame)
        with self._lock:
            rotate = self._rotate
            flip_h = self._flip_h
            enhance = self._enhance

        transformed = self._apply_transform(base, rotate=rotate, flip_h=flip_h, enhance=enhance)
        now = time.monotonic()
        with self._lock:
            self._raw_base_frame = base
            self._latest_frame = transformed
            self._resolution = [int(transformed.shape[1]), int(transformed.shape[0])]
            self._latest_ts = now
            self._connected = True
            self._last_error = ""

    def _can_attempt_cuvid(self, url: str, fallback_latched: bool) -> bool:
        if fallback_latched:
            return False
        if self._hwaccel not in ("auto", "nvdec"):
            return False
        if not _is_rtsp_url(url):
            return False
        if self._find_ffmpeg_bin() is None:
            return False
        return self._has_nvidia_gpu()

    def _run_cuvid_stream(
        self,
        session: _CuvidSession,
        first_frame: np.ndarray,
        min_decode_interval: float,
    ) -> int:
        """
        Читает кадры из активной сессии FFmpeg CUVID до остановки, перезагрузки URL или обрыва пайпа.
        Возвращает количество успешно принятых кадров (для детекции ранних падений CUVID).
        """
        self._store_fresh_frame(first_frame)
        last_decode_ts = time.monotonic()
        frames_ok = 1
        corrupt_streak = 0

        while not self._stop_event.is_set() and not self._reload_event.is_set():
            if not session.read_raw_packet(self._stop_event, self._reload_event):
                break
            now_mono = time.monotonic()
            # Вычитываем каждый сырой NV12 пакет из пайпа для нулевой задержки,
            # а конвертацию YUV->BGR и CLAHE выполняем строго с частотой capture_fps
            if (now_mono - last_decode_ts) < min_decode_interval:
                frames_ok += 1
                continue
            frm = session.decode_current_bgr()
            if not self._is_usable_frame(frm):
                corrupt_streak += 1
                if corrupt_streak >= 10:
                    break
                continue
            corrupt_streak = 0
            frames_ok += 1
            last_decode_ts = now_mono
            self._store_fresh_frame(frm)

        return frames_ok

    def _loop(self) -> None:
        cap: cv2.VideoCapture | None = None
        backoff = 1.5
        consecutive_failures = 0
        cuvid_failures = 0
        cuvid_fallback_latched = False
        last_decode_ts = 0.0
        min_decode_interval = 1.0 / max(2.0, self._capture_fps * 1.35)

        try:
            while not self._stop_event.is_set():
                if self._reload_event.is_set():
                    self._reload_event.clear()
                    self._close_active_cuvid()
                    if cap is not None:
                        cap.release()
                        cap = None
                    consecutive_failures = 0
                    cuvid_failures = 0
                    cuvid_fallback_latched = False
                    backoff = 1.5

                with self._lock:
                    current_url = self._rtsp_url

                if not current_url:
                    self._close_active_cuvid()
                    if cap is not None:
                        cap.release()
                        cap = None
                    with self._lock:
                        self._connected = False
                        self._decoder_backend = BACKEND_CPU
                        self._last_error = "RTSP_URL не задан в .env"
                    self._stop_event.wait(1.0)
                    continue

                # Ветка 1: Аппаратное декодирование NVDEC (h264_cuvid) с прозрачным откатом после 2 сбоев
                if cap is None and self._can_attempt_cuvid(current_url, cuvid_fallback_latched):
                    session, cuvid_first = self._open_cuvid_session(current_url)
                    if session is not None and cuvid_first is not None:
                        backoff = 1.5
                        frames_read = self._run_cuvid_stream(session, cuvid_first, min_decode_interval)
                        self._close_active_cuvid()
                        if self._stop_event.is_set() or self._reload_event.is_set():
                            continue
                        if frames_read >= 30:
                            # Стабильная долгая сессия (обычный сетевой реконнект камеры)
                            cuvid_failures = 0
                            with self._lock:
                                self._connected = False
                                self._last_error = "Поток NVDEC прерван, переподключение..."
                            self._stop_event.wait(1.0)
                            continue
                        cuvid_failures += 1
                    else:
                        self._close_active_cuvid()
                        cuvid_failures += 1

                    if cuvid_failures >= 2:
                        cuvid_fallback_latched = True
                        logger.warning(
                            "h264_cuvid завершился ошибкой %d раз(а) подряд — прозрачный переход на %s",
                            cuvid_failures,
                            BACKEND_CPU,
                        )
                    else:
                        self._stop_event.wait(0.25)
                        continue

                # Ветка 2: Штатный декодер cv2.VideoCapture (CPU FFmpeg, автоподбор TCP -> UDP)
                if cap is None or not cap.isOpened():
                    try:
                        if cap is not None:
                            cap.release()
                        with self._lock:
                            self._decoder_backend = BACKEND_CPU
                        cap, first_frame, transport = self._connect_auto_transport(current_url)
                        if cap is None or first_frame is None:
                            with self._lock:
                                self._connected = False
                                self._last_error = (
                                    "Не удалось открыть видеопоток камеры по TCP/UDP "
                                    "(проверьте RTSP_URL и сеть)"
                                )
                            self._stop_event.wait(backoff)
                            backoff = min(10.0, backoff * 1.5)
                            continue

                        with self._lock:
                            self._active_transport = transport
                            self._decoder_backend = BACKEND_CPU
                        self._store_fresh_frame(first_frame)
                        last_decode_ts = time.monotonic()
                        consecutive_failures = 0
                        backoff = 1.5
                    except Exception as exc:
                        with self._lock:
                            self._connected = False
                            self._last_error = f"Ошибка открытия потока: {_redact_url(str(exc))}"
                        self._stop_event.wait(backoff)
                        backoff = min(10.0, backoff * 1.5)
                        continue

                grabbed = cap.grab()
                if not grabbed:
                    consecutive_failures += 1
                    if consecutive_failures < 8:
                        self._stop_event.wait(0.05)
                        continue
                    with self._lock:
                        self._connected = False
                        self._last_error = "Поток прерван, переподключение..."
                    cap.release()
                    cap = None
                    consecutive_failures = 0
                    self._stop_event.wait(1.5)
                    continue

                now_mono = time.monotonic()
                if (now_mono - last_decode_ts) < min_decode_interval:
                    continue

                ok, frame = cap.retrieve()
                if not ok or not self._is_usable_frame(frame):
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        with self._lock:
                            self._connected = False
                            self._last_error = "Серия повреждённых кадров, переоткрытие потока..."
                        cap.release()
                        cap = None
                        consecutive_failures = 0
                        self._stop_event.wait(1.0)
                    continue

                consecutive_failures = 0
                last_decode_ts = now_mono
                assert frame is not None
                self._store_fresh_frame(frame)
        finally:
            self._close_active_cuvid()
            if cap is not None:
                cap.release()
