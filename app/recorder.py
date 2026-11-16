from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Sequence

from .capture import CREATE_NO_WINDOW, BufferedFrame, find_ffmpeg_bin

logger = logging.getLogger("fallguard.recorder")

ENCODER_NVENC = "NVENC (h264_nvenc)"
ENCODER_CPU = "CPU (libx264)"


def _build_nvenc_args() -> list[str]:
    return [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]


def _build_cpu_x264_args() -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]


class ClipRecorder:
    """
    Сборщик тревожных MP4-клипов с аппаратным кодированием NVIDIA NVENC (h264_nvenc),
    автоматической пробой драйвера, бесшовным CPU-откатом (libx264), защитой от окон
    консоли на Windows (CREATE_NO_WINDOW) и замером времени кодирования last_encode_ms.
    """

    def __init__(
        self,
        hwaccel: str = "auto",
        ffmpeg_bin: str | None = None,
    ) -> None:
        mode = str(hwaccel or "auto").strip().lower()
        self._hwaccel = mode if mode in ("auto", "nvenc", "nvdec", "off") else "auto"
        self._explicit_ffmpeg = ffmpeg_bin
        self._lock = threading.RLock()
        self._nvenc_supported: bool | None = False if self._hwaccel == "off" else None
        self._last_encode_ms: float = 0.0

    def _resolve_ffmpeg(self) -> str | None:
        if self._explicit_ffmpeg:
            return self._explicit_ffmpeg
        return find_ffmpeg_bin()

    def probe_nvenc(self, force: bool = False) -> bool:
        """
        Выполняет быструю однократную пробу открытия аппаратной сессии h264_nvenc (1 кадр 256×144 -> null)
        и кэширует результат.
        """
        with self._lock:
            if self._hwaccel == "off":
                self._nvenc_supported = False
                return False
            if not force and self._nvenc_supported is not None:
                return self._nvenc_supported

        ffmpeg_bin = self._resolve_ffmpeg()
        if not ffmpeg_bin:
            with self._lock:
                self._nvenc_supported = False
            return False

        probe_cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=256x144:r=10",
            "-frames:v",
            "1",
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "null",
            "-",
        ]
        supported = False
        try:
            res = subprocess.run(
                probe_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=3.5,
                check=False,
                creationflags=CREATE_NO_WINDOW,
            )
            supported = res.returncode == 0
        except Exception:
            supported = False

        with self._lock:
            self._nvenc_supported = supported
        if supported:
            logger.info("Аппаратный кодировщик клипов активирован: %s", ENCODER_NVENC)
        return supported

    @property
    def encoder_backend(self) -> str:
        with self._lock:
            if self._hwaccel == "off":
                return ENCODER_CPU
            cached = self._nvenc_supported
        if cached is None:
            cached = self.probe_nvenc()
        return ENCODER_NVENC if cached else ENCODER_CPU

    @property
    def last_encode_ms(self) -> float:
        with self._lock:
            return self._last_encode_ms

    def encode(
        self,
        frames: Sequence[BufferedFrame],
        fps: float = 6.0,
        timeout_sec: float = 25.0,
    ) -> bytes | None:
        """
        Кодирует последовательность JPEG-кадров в совместимый с iOS/Android MP4 (H.264 yuv420p +faststart):
        сначала через h264_nvenc (если поддерживается), с прозрачным откатом на libx264 при ошибке.
        """
        if len(frames) < 4:
            return None
        ffmpeg_bin = self._resolve_ffmpeg()
        if not ffmpeg_bin:
            logger.warning("ffmpeg не найден в PATH / C:\\FallGuard\\bin — сборка видеоклипа пропущена")
            return None

        eff_fps = max(3.0, min(15.0, float(fps)))
        if len(frames) >= 2:
            span = frames[-1].timestamp - frames[0].timestamp
            if span > 0.5:
                eff_fps = max(3.0, min(15.0, len(frames) / span))

        use_nvenc = self.probe_nvenc()
        attempts: list[tuple[bool, list[str]]] = []
        if use_nvenc:
            attempts.append((True, _build_nvenc_args()))
        attempts.append((False, _build_cpu_x264_args()))

        stream_blob = b"".join(f.jpeg_bytes for f in frames)

        with tempfile.TemporaryDirectory(prefix="fallguard_clip_") as tmpdir:
            out_mp4 = Path(tmpdir) / "event.mp4"
            for is_hw, codec_args in attempts:
                if out_mp4.exists():
                    try:
                        out_mp4.unlink()
                    except OSError:
                        pass

                cmd = [
                    ffmpeg_bin,
                    "-nostdin",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "image2pipe",
                    "-framerate",
                    f"{eff_fps:.2f}",
                    "-vcodec",
                    "mjpeg",
                    "-i",
                    "-",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    *codec_args,
                    str(out_mp4),
                ]

                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        cmd,
                        input=stream_blob,
                        capture_output=True,
                        timeout=timeout_sec,
                        check=False,
                        creationflags=CREATE_NO_WINDOW,
                    )
                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    if proc.returncode == 0 and out_mp4.is_file():
                        mp4_data = out_mp4.read_bytes()
                        if mp4_data:
                            with self._lock:
                                self._last_encode_ms = elapsed_ms
                                if not is_hw and use_nvenc:
                                    self._nvenc_supported = False
                            return mp4_data

                    err_msg = proc.stderr.decode("utf-8", errors="replace")[:260]
                    if is_hw:
                        logger.warning(
                            "Сбой кодирования клипа через h264_nvenc (код %s: %s) — откат на %s",
                            proc.returncode,
                            err_msg,
                            ENCODER_CPU,
                        )
                        with self._lock:
                            self._nvenc_supported = False
                    else:
                        logger.error("Ошибка сборки MP4 через libx264 (код %s): %s", proc.returncode, err_msg)
                except subprocess.TimeoutExpired:
                    logger.error("Таймаут ffmpeg (%ss) при кодировании клипа", timeout_sec)
                    if is_hw:
                        with self._lock:
                            self._nvenc_supported = False
                        continue
                    return None
                except Exception as exc:
                    logger.error("Исключение при сборке MP4: %s", exc)
                    if not is_hw:
                        return None
        return None

    def build_mp4(
        self,
        frames: Sequence[BufferedFrame],
        fps: float = 6.0,
        timeout_sec: float = 25.0,
    ) -> bytes | None:
        return self.encode(frames, fps=fps, timeout_sec=timeout_sec)


default_clip_recorder = ClipRecorder(hwaccel="auto")


def get_encoder_backend() -> str:
    return default_clip_recorder.encoder_backend


def get_last_encode_ms() -> float:
    return default_clip_recorder.last_encode_ms


def build_mp4_from_jpegs(
    frames: Sequence[BufferedFrame],
    fps: float = 6.0,
    timeout_sec: float = 25.0,
) -> bytes | None:
    """Обратно-совместимая обёртка над глобальным ClipRecorder (NVENC -> libx264)."""
    return default_clip_recorder.encode(frames, fps=fps, timeout_sec=timeout_sec)
