from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app.capture import CREATE_NO_WINDOW, BufferedFrame
from app.recorder import (
    ENCODER_CPU,
    ENCODER_NVENC,
    ClipRecorder,
    build_mp4_from_jpegs,
)
from app.stream_hub import (
    H264StreamHub,
    IsoBmffStreamParser,
    build_passthrough_fmp4_cmd,
    is_fragment_keyframe,
)


def _generate_real_fmp4_stream(width: int = 320, height: int = 180) -> bytes:
    """Генерирует настоящий H.264 High Profile fragmented MP4 поток через системный ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    assert ffmpeg_bin is not None, "В тестовом окружении должен быть установлен ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate=10",
        "-frames:v",
        "16",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "6",
        "-keyint_min",
        "6",
        "-sc_threshold",
        "0",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
        "-frag_duration",
        "200000",
        "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=15.0, check=True, creationflags=CREATE_NO_WINDOW)
    assert len(res.stdout) > 1024
    return res.stdout


def _make_buffered_jpegs(count: int = 8, width: int = 320, height: int = 180) -> list[BufferedFrame]:
    frames: list[BufferedFrame] = []
    t0 = 1000.0
    for idx in range(count):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = ((idx * 25) % 255, 80, 140)
        cv2.rectangle(img, (20 + idx * 8, 30), (90 + idx * 8, 120), (40, 220, 120), -1)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        assert ok
        frames.append(BufferedFrame(timestamp=t0 + idx * 0.16, jpeg_bytes=enc.tobytes()))
    return frames


def test_isobmff_parser_avcc_extraction_and_decodable_gop_cache(tmp_path: Path) -> None:
    raw_fmp4 = _generate_real_fmp4_stream(width=320, height=180)

    hub = H264StreamHub(rtsp_url="")
    emitted_all = []
    # Подаём поток мелкими невыровненными кусками по 97 байт для проверки склейки границ боксов
    step = 97
    for pos in range(0, len(raw_fmp4), step):
        emitted_all.extend(hub.feed_bytes(raw_fmp4[pos : pos + step]))

    assert hub.is_ready is True
    assert hub.init_segment is not None
    assert hub.init_segment[:8].endswith(b"ftyp")
    assert b"moov" in hub.init_segment

    init_msg = hub.get_init_message()
    assert init_msg["type"] == "init"
    # High Profile (0x64 = 100), Level 4.0 (0x28 = 40) -> avc1.64..28
    assert init_msg["codecs"].startswith("avc1.64")
    assert init_msg["codecs"].endswith("28")
    assert init_msg["mime"] == f'video/mp4; codecs="{init_msg["codecs"]}"'
    assert init_msg["width"] == 320
    assert init_msg["height"] == 180

    # Проверяем наличие как ключевых (IDR), так и неключевых фрагментов при frag_duration=200ms и GOP=6
    assert len(emitted_all) >= 3
    assert any(f.is_keyframe for f in emitted_all)

    gop = hub.gop_cache
    assert len(gop) >= 1
    # Первый фрагмент в GOP-кэше обязан начинаться с ключевого кадра
    first_gop_boxes = IsoBmffStreamParser().feed(gop[0])
    moof_bytes = next(b for t, b in first_gop_boxes if t == b"moof")
    mdat_bytes = next(b for t, b in first_gop_boxes if t == b"mdat")
    assert is_fragment_keyframe(moof_bytes, mdat_bytes) is True

    # Критическая проверка для MSE/PWA: склейка init_segment + gop_cache должна чисто декодироваться!
    playable_mp4 = tmp_path / "instant_start.mp4"
    playable_mp4.write_bytes(hub.init_segment + b"".join(gop))
    cap = cv2.VideoCapture(str(playable_mp4))
    try:
        assert cap.isOpened()
        ok, decoded_bgr = cap.read()
        assert ok and decoded_bgr is not None
        assert decoded_bgr.shape == (180, 320, 3)
    finally:
        cap.release()


def test_stream_hub_async_subscribers_gop_burst_and_slow_client_drop() -> None:
    raw_fmp4 = _generate_real_fmp4_stream(width=320, height=180)
    hub = H264StreamHub(rtsp_url="", queue_maxsize=6)

    # Сначала прогреваем хаб начальной частью потока (чтобы заполнить init_segment и GOP-кэш)
    boxes = IsoBmffStreamParser().feed(raw_fmp4)
    # Отправляем ftyp + moov + первые 2 фрагмента (moof+mdat)
    mdat_seen = 0
    split_idx = 0
    for idx, (btype, full_box) in enumerate(boxes):
        hub.feed_bytes(full_box)
        if btype == b"mdat":
            mdat_seen += 1
            if mdat_seen >= 2:
                split_idx = idx + 1
                break

    assert hub.is_ready is True
    pre_gop_count = len(hub.gop_cache)
    assert pre_gop_count >= 1

    async def _scenario() -> None:
        async with hub.subscribe(include_gop=True) as sub1, hub.subscribe(include_gop=False) as sub2:
            assert hub.stream_clients == 2
            # sub1 мгновенно получил весь текущий GOP в очередь (< 1 мс без ожидания следующего IDR!)
            assert sub1.queue.qsize() == pre_gop_count
            assert sub2.queue.qsize() == 0

            # Досылаем оставшиеся боксы из «живого» потока
            for _, rem_box in boxes[split_idx:]:
                hub.feed_bytes(rem_box)
            await asyncio.sleep(0.02)

            chunk_1 = await asyncio.wait_for(sub1.get(), timeout=1.0)
            chunk_2 = await asyncio.wait_for(sub2.get(), timeout=1.0)
            assert chunk_1 is not None and b"moof" in chunk_1 and b"mdat" in chunk_1
            assert chunk_2 is not None and b"moof" in chunk_2 and b"mdat" in chunk_2

        assert hub.stream_clients == 0

    asyncio.run(_scenario())


def test_stream_hub_passthrough_cmd_and_clean_process_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = build_passthrough_fmp4_cmd(
        "/usr/bin/ffmpeg",
        "rtsp://admin:TopSecret@192.168.1.50:554/Streaming/Channels/101",
        frag_duration_us=200_000,
    )
    assert "-c:v" in cmd and "copy" in cmd
    assert "-rtsp_transport" in cmd and "tcp" in cmd
    assert "-timeout" in cmd and "5000000" in cmd
    assert "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset" in cmd

    raw_fmp4 = _generate_real_fmp4_stream(width=320, height=180)

    class _FakeFmp4Proc:
        def __init__(self) -> None:
            self._hold = asyncio.Event()
            self._bio = io.BytesIO(raw_fmp4)
            self.stderr = io.BytesIO(b"[rtsp] warn rtsp://admin:TopSecret@192.168.1.50:554/ch1\n")
            self.killed = False
            self.waited = False
            self._rc: int | None = None
            self.stdout = self

        def readinto(self, b: Any) -> int:
            if self.killed:
                return 0
            mv = memoryview(b)
            chunk = self._bio.read(min(len(mv), 4096))
            if chunk:
                mv[: len(chunk)] = chunk
                return len(chunk)
            time.sleep(0.05)
            return 0

        def close(self) -> None:
            pass

        def poll(self) -> int | None:
            return self._rc

        def kill(self) -> None:
            self.killed = True
            self._rc = -9

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            return -9

    spawned: list[_FakeFmp4Proc] = []

    def fake_popen(args: list[str], **kwargs: Any) -> _FakeFmp4Proc:
        assert kwargs.get("creationflags") == CREATE_NO_WINDOW
        p = _FakeFmp4Proc()
        spawned.append(p)
        return p

    monkeypatch.setattr("app.stream_hub.subprocess.Popen", fake_popen)
    hub = H264StreamHub(
        rtsp_url="rtsp://admin:TopSecret@192.168.1.50:554/Streaming/Channels/101",
        ffmpeg_bin="/usr/bin/ffmpeg",
    )
    hub.start()
    try:
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline and not hub.is_ready:
            time.sleep(0.03)
        assert hub.is_ready is True
        assert hub.width == 320 and hub.height == 180
    finally:
        hub.stop()

    assert len(spawned) >= 1
    assert spawned[0].killed is True
    assert spawned[0].waited is True
    assert "TopSecret" not in hub.last_error


def test_clip_recorder_cpu_fallback_and_nvenc_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _make_buffered_jpegs(count=8)

    # 1. Принудительный CPU-режим (libx264) и автоопределение (CPU в контейнере / NVENC на RTX GPU)
    auto_rec = ClipRecorder(hwaccel="auto")
    assert auto_rec.encoder_backend in (ENCODER_CPU, ENCODER_NVENC)
    auto_mp4 = auto_rec.encode(frames, fps=8.0)
    assert auto_mp4 is not None and len(auto_mp4) > 512

    cpu_rec = ClipRecorder(hwaccel="off")
    assert cpu_rec.encoder_backend == ENCODER_CPU
    mp4_bytes = cpu_rec.encode(frames, fps=8.0)
    assert mp4_bytes is not None and len(mp4_bytes) > 512
    assert cpu_rec.last_encode_ms > 0.0

    out_file = tmp_path / "clip_cpu.mp4"
    out_file.write_bytes(mp4_bytes)
    cap = cv2.VideoCapture(str(out_file))
    try:
        assert cap.isOpened()
        ok, frm = cap.read()
        assert ok and frm is not None
    finally:
        cap.release()

    # Проверяем обратную совместимость build_mp4_from_jpegs
    compat_mp4 = build_mp4_from_jpegs(frames, fps=8.0)
    assert compat_mp4 is not None and len(compat_mp4) > 512

    # 2. Эмуляция активного h264_nvenc + прозрачного отката при runtime-сбое NVENC
    real_run = subprocess.run
    recorded_cmds: list[list[str]] = []
    fail_nvenc_encode = False

    def intercept_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert kwargs.get("creationflags") == CREATE_NO_WINDOW
        recorded_cmds.append(list(cmd))
        if "lavfi" in cmd and "h264_nvenc" in cmd:
            # Успешная проба драйвера NVENC
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        if "h264_nvenc" in cmd:
            if fail_nvenc_encode:
                return subprocess.CompletedProcess(cmd, 1, b"", b"Out of NVENC sessions")
            # Подменяем h264_nvenc на реальный libx264, чтобы проверить создание валидного MP4
            patched = list(cmd)
            idx = patched.index("h264_nvenc")
            patched[idx : idx + 5] = ["libx264", "-preset", "ultrafast", "-crf", "25"]
            return real_run(patched, *args, **kwargs)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr("app.recorder.subprocess.run", intercept_run)

    nv_rec = ClipRecorder(hwaccel="auto")
    assert nv_rec.probe_nvenc(force=True) is True
    assert nv_rec.encoder_backend == ENCODER_NVENC

    recorded_cmds.clear()
    nv_mp4 = nv_rec.encode(frames, fps=8.0)
    assert nv_mp4 is not None and len(nv_mp4) > 512
    assert nv_rec.last_encode_ms > 0.0
    assert any("h264_nvenc" in c and "p4" in c and "26" in c for c in recorded_cmds)

    # А теперь имитируем внезапный сбой сессии NVENC во время реальной тревоги -> прозрачный откат на libx264!
    fail_nvenc_encode = True
    recorded_cmds.clear()
    fallback_mp4 = nv_rec.encode(frames, fps=8.0)
    assert fallback_mp4 is not None and len(fallback_mp4) > 512
    assert any("h264_nvenc" in c for c in recorded_cmds)
    assert any("libx264" in c for c in recorded_cmds)
    assert nv_rec.encoder_backend == ENCODER_CPU
