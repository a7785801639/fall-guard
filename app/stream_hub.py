from __future__ import annotations

import asyncio
import logging
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import IO, Any, AsyncIterator, Callable

from .capture import CREATE_NO_WINDOW, _is_rtsp_url, _redact_url, find_ffmpeg_bin

logger = logging.getLogger("fallguard.stream_hub")

MAX_BOX_BYTES = 16 * 1024 * 1024
MAX_GOP_FRAGMENTS = 30
MAX_GOP_BYTES = 4 * 1024 * 1024
DEFAULT_CODEC_STRING = "avc1.640028"


def build_passthrough_fmp4_cmd(
    ffmpeg_bin: str,
    rtsp_url: str,
    frag_duration_us: int = 200_000,
) -> list[str]:
    """Формирует команду прямого RTSP -> fMP4 ремультиплексирования без перекодирования (-c:v copy)."""
    return [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "5000000",
        "-i",
        rtsp_url,
        "-an",
        "-sn",
        "-c:v",
        "copy",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
        "-frag_duration",
        str(max(40_000, int(frag_duration_us))),
        "pipe:1",
    ]


def build_transcode_fmp4_cmd(
    ffmpeg_bin: str,
    rtsp_url: str,
    target_size: tuple[int, int],
    fps_limit: int = 12,
    bitrate_kbps: int = 1200,
    use_nvenc: bool = True,
    frag_duration_us: int = 200_000,
) -> list[str]:
    """Формирует команду FFmpeg для масштабирования RTSP-потока в лёгкий fMP4 (NVENC / libx264)."""
    w = max(2, int(target_size[0]) & ~1)
    h = max(2, int(target_size[1]) & ~1)
    cmd: list[str] = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "5000000",
        "-i",
        rtsp_url,
        "-an",
        "-sn",
        "-vf",
        f"scale={w}:{h}",
    ]
    if fps_limit > 0:
        cmd.extend(["-r", str(int(fps_limit))])
    if use_nvenc:
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "27"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-crf", "26"])
    if bitrate_kbps > 0:
        kb = int(bitrate_kbps)
        cmd.extend(["-maxrate", f"{kb}k", "-bufsize", f"{kb * 2}k"])
    cmd.extend(
        [
            "-g",
            "15",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mp4",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
            "-frag_duration",
            str(max(40_000, int(frag_duration_us))),
            "pipe:1",
        ]
    )
    return cmd


def _iter_box_spans(buf: bytes | bytearray, start: int, end: int) -> list[tuple[bytes, int, int, int]]:
    """Возвращает (box_type, box_start, body_start, box_end) для прямых дочерних боксов в диапазоне."""
    out: list[tuple[bytes, int, int, int]] = []
    pos = start
    while pos + 8 <= end:
        size32 = struct.unpack_from(">I", buf, pos)[0]
        btype = bytes(buf[pos + 4 : pos + 8])
        if size32 == 1:
            if pos + 16 > end:
                break
            bsize = struct.unpack_from(">Q", buf, pos + 8)[0]
            hdr = 16
        elif size32 == 0:
            bsize = end - pos
            hdr = 8
        else:
            bsize = size32
            hdr = 8
        if bsize < hdr or pos + bsize > end:
            break
        out.append((btype, pos, pos + hdr, pos + bsize))
        pos += bsize
    return out


def rebase_moof_tfdt(frag_bytes: bytes, base_tfdt: int | None) -> tuple[bytes, int]:
    """
    Вычитает base_tfdt из baseMediaDecodeTime (tfdt) во всех traf-боксах фрагмента moof.
    Длина бокса не меняется (offset в trun сохраняется), а незакрытый fMP4-сегмент начинается с 0:00.
    """
    if not frag_bytes:
        return frag_bytes, 0 if base_tfdt is None else int(base_tfdt)
    buf = bytearray(frag_bytes)
    eff_base: int | None = int(base_tfdt) if base_tfdt is not None else None
    changed = False

    for top_type, _, top_body, top_end in _iter_box_spans(buf, 0, len(buf)):
        if top_type != b"moof":
            continue
        for m_type, _, m_body, m_end in _iter_box_spans(buf, top_body, top_end):
            if m_type != b"traf":
                continue
            for t_type, _, t_body, t_end in _iter_box_spans(buf, m_body, m_end):
                if t_type != b"tfdt" or t_body + 8 > t_end:
                    continue
                version = buf[t_body]
                if version == 0:
                    orig = struct.unpack_from(">I", buf, t_body + 4)[0]
                    if eff_base is None:
                        eff_base = orig
                    new_val = max(0, orig - eff_base)
                    struct.pack_into(">I", buf, t_body + 4, new_val & 0xFFFFFFFF)
                    changed = True
                elif version == 1 and t_body + 12 <= t_end:
                    orig = struct.unpack_from(">Q", buf, t_body + 4)[0]
                    if eff_base is None:
                        eff_base = orig
                    new_val = max(0, orig - eff_base)
                    struct.pack_into(">Q", buf, t_body + 4, new_val)
                    changed = True

    final_base = 0 if eff_base is None else eff_base
    return (bytes(buf) if changed else frag_bytes), final_base


def _iter_direct_child_boxes(payload: bytes) -> list[tuple[bytes, int, bytes]]:
    """Возвращает прямые дочерние ISO-BMFF боксы внутри контейнера: (box_type, header_len, full_box)."""
    out: list[tuple[bytes, int, bytes]] = []
    pos = 0
    total = len(payload)
    while pos + 8 <= total:
        size32 = struct.unpack_from(">I", payload, pos)[0]
        btype = payload[pos + 4 : pos + 8]
        if size32 == 1:
            if pos + 16 > total:
                break
            bsize = struct.unpack_from(">Q", payload, pos + 8)[0]
            hdr = 16
        elif size32 == 0:
            bsize = total - pos
            hdr = 8
        else:
            bsize = size32
            hdr = 8
        if bsize < hdr or pos + bsize > total:
            break
        out.append((btype, hdr, payload[pos : pos + bsize]))
        pos += bsize
    return out


def parse_moov_metadata(moov_box: bytes) -> dict[str, Any]:
    """
    Извлекает точную строку кодека RFC 6381 (avc1.%02X%02X%02X из avcC)
    и разрешение кадра (width, height) из инициализационного бокса moov.
    """
    codec_str = ""
    width = 0
    height = 0
    tkhd_w = 0
    tkhd_h = 0

    if len(moov_box) >= 8 and moov_box[4:8] == b"moov":
        hdr0 = 16 if struct.unpack_from(">I", moov_box, 0)[0] == 1 else 8
        moov_body = moov_box[hdr0:]
    else:
        moov_body = moov_box

    containers = {b"trak", b"mdia", b"minf", b"stbl"}
    stack: list[bytes] = [moov_body]

    while stack:
        body = stack.pop()
        for btype, hdr, full_box in _iter_direct_child_boxes(body):
            inner = full_box[hdr:]
            if btype in containers:
                stack.append(inner)
            elif btype == b"tkhd" and len(inner) >= 80:
                version = inner[0]
                # В tkhd (version 0: 80B, version 1: 92B) последние 8 байт — width и height в 16.16 fixed-point
                wh_offset = 76 if version == 0 else 88
                if len(inner) >= wh_offset + 8:
                    fw, fh = struct.unpack_from(">II", inner, wh_offset)
                    if fw > 0 and fh > 0:
                        tkhd_w = fw >> 16
                        tkhd_h = fh >> 16
            elif btype == b"stsd" and len(inner) >= 16:
                # FullBox (4B version+flags) + entry_count (4B) = 8B до VisualSampleEntry
                for entry_type, entry_hdr, entry_box in _iter_direct_child_boxes(inner[8:]):
                    entry_body = entry_box[entry_hdr:]
                    if entry_type in (b"avc1", b"avc3", b"hev1", b"hvc1") and len(entry_body) >= 78:
                        # VisualSampleEntry: 6B reserved + 2B ref_idx + 16B pre_defined = 24B до width/height
                        ew, eh = struct.unpack_from(">HH", entry_body, 24)
                        if ew > 0 and eh > 0:
                            width, height = int(ew), int(eh)
                        # Дочерние боксы (avcC, hvcC, pasp, btrt) начинаются со смещения 78 внутри entry_body
                        for cfg_type, cfg_hdr, cfg_box in _iter_direct_child_boxes(entry_body[78:]):
                            cfg_body = cfg_box[cfg_hdr:]
                            if cfg_type == b"avcC" and len(cfg_body) >= 4:
                                prof, compat, level = cfg_body[1], cfg_body[2], cfg_body[3]
                                codec_str = f"avc1.{prof:02X}{compat:02X}{level:02X}"
                            elif cfg_type == b"hvcC" and not codec_str:
                                codec_str = "hvc1.1.6.L93.B0"

    # Резервный поиск бокса avcC по сигнатуре, если структура дерева нестандартна
    if not codec_str:
        idx = moov_box.find(b"avcC")
        if idx >= 4 and idx + 8 <= len(moov_box):
            box_sz = struct.unpack_from(">I", moov_box, idx - 4)[0]
            if 11 <= box_sz <= len(moov_box) - (idx - 4):
                prof = moov_box[idx + 5]
                compat = moov_box[idx + 6]
                level = moov_box[idx + 7]
                codec_str = f"avc1.{prof:02X}{compat:02X}{level:02X}"

    if width <= 0 or height <= 0:
        width, height = tkhd_w, tkhd_h

    final_codec = codec_str or DEFAULT_CODEC_STRING
    return {
        "codecs": final_codec,
        "mime": f'video/mp4; codecs="{final_codec}"',
        "width": int(width),
        "height": int(height),
    }


def is_fragment_keyframe(moof_box: bytes, mdat_box: bytes = b"") -> bool:
    """
    Определяет по флагам ISO-BMFF (moof -> traf -> tfhd/trun) и резервно по AVCC NAL (type=5 IDR),
    начинается ли фрагмент с ключевого кадра (Sync Sample).
    """
    if len(moof_box) >= 8 and moof_box[4:8] == b"moof":
        moof_hdr = 16 if struct.unpack_from(">I", moof_box, 0)[0] == 1 else 8
        moof_body = moof_box[moof_hdr:]
        for btype, hdr, full_box in _iter_direct_child_boxes(moof_body):
            if btype != b"traf":
                continue
            default_flags: int | None = None
            sample_flags: int | None = None
            for tb_type, tb_hdr, tb_box in _iter_direct_child_boxes(full_box[hdr:]):
                tb_body = tb_box[tb_hdr:]
                if tb_type == b"tfhd" and len(tb_body) >= 8:
                    tf_flags = struct.unpack_from(">I", tb_body, 0)[0] & 0x00FFFFFF
                    pos = 8  # version/flags (4B) + track_ID (4B)
                    if tf_flags & 0x000001:
                        pos += 8
                    if tf_flags & 0x000002:
                        pos += 4
                    if tf_flags & 0x000008:
                        pos += 4
                    if tf_flags & 0x000010:
                        pos += 4
                    if (tf_flags & 0x000020) and pos + 4 <= len(tb_body):
                        default_flags = struct.unpack_from(">I", tb_body, pos)[0]
                elif tb_type == b"trun" and len(tb_body) >= 8:
                    tr_flags = struct.unpack_from(">I", tb_body, 0)[0] & 0x00FFFFFF
                    sample_count = struct.unpack_from(">I", tb_body, 4)[0]
                    pos = 8
                    if tr_flags & 0x000001:
                        pos += 4  # data_offset
                    if tr_flags & 0x000004:
                        if pos + 4 <= len(tb_body):
                            sample_flags = struct.unpack_from(">I", tb_body, pos)[0]
                    elif sample_count > 0 and (tr_flags & 0x000400):
                        if tr_flags & 0x000100:
                            pos += 4  # sample_duration
                        if tr_flags & 0x000200:
                            pos += 4  # sample_size
                        if pos + 4 <= len(tb_body):
                            sample_flags = struct.unpack_from(">I", tb_body, pos)[0]

            eff_flags = sample_flags if sample_flags is not None else default_flags
            if eff_flags is not None and eff_flags != 0:
                depends_on = (eff_flags >> 24) & 0x03
                is_non_sync = bool((eff_flags >> 16) & 0x01)
                if depends_on == 2:
                    return True
                if is_non_sync or depends_on == 1:
                    return False
                return True

    # Резервный разбор первых NAL-юнитов внутри mdat (4-байтовый AVCC length prefix)
    if len(mdat_box) > 13 and mdat_box[4:8] == b"mdat":
        mdat_hdr = 16 if struct.unpack_from(">I", mdat_box, 0)[0] == 1 else 8
        pos = mdat_hdr
        limit = min(len(mdat_box), mdat_hdr + 65536)
        checked = 0
        while pos + 5 <= limit and checked < 12:
            nal_len = struct.unpack_from(">I", mdat_box, pos)[0]
            if nal_len <= 0 or pos + 4 + nal_len > len(mdat_box):
                break
            nal_type = mdat_box[pos + 4] & 0x1F
            if nal_type == 5:  # IDR slice
                return True
            if nal_type == 1:  # non-IDR slice
                return False
            pos += 4 + nal_len
            checked += 1

    return False


class IsoBmffStreamParser:
    """Потоковый парсер боксов ISO-BMFF, устойчивый к произвольной нарезке байтовых чанков пайпа."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[tuple[bytes, bytes]]:
        if not chunk:
            return []
        self._buf.extend(chunk)
        boxes: list[tuple[bytes, bytes]] = []

        while len(self._buf) >= 8:
            size32 = struct.unpack_from(">I", self._buf, 0)[0]
            btype = bytes(self._buf[4:8])
            if not all(32 <= c <= 126 for c in btype):
                raise ValueError(f"Повреждённый заголовок бокса ISO-BMFF: {btype!r}")

            if size32 == 1:
                if len(self._buf) < 16:
                    break
                box_size = struct.unpack_from(">Q", self._buf, 8)[0]
                min_hdr = 16
            else:
                box_size = size32
                min_hdr = 8

            if box_size < min_hdr or box_size > MAX_BOX_BYTES:
                raise ValueError(f"Некорректный размер бокса {btype!r}: {box_size}")

            if len(self._buf) < box_size:
                break

            full_box = bytes(self._buf[:box_size])
            del self._buf[:box_size]
            boxes.append((btype, full_box))

        return boxes


@dataclass(frozen=True)
class StreamFragment:
    seq: int
    timestamp: float
    data: bytes
    is_keyframe: bool


class StreamSubscription:
    """
    Клиентская подписка на H264StreamHub с собственной asyncio.Queue.
    Поддерживает чтение через `async for chunk in sub:` или напрямую через `await sub.get()`.
    """

    def __init__(
        self,
        hub: H264StreamHub,
        loop: asyncio.AbstractEventLoop,
        queue_maxsize: int = 24,
    ) -> None:
        self._hub = hub
        self.loop = loop
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=max(4, queue_maxsize))
        self._closed = False

    def __enter__(self) -> StreamSubscription:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> StreamSubscription:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _enqueue_in_loop(self, frag: StreamFragment | None) -> None:
        if self._closed:
            return
        if frag is None:
            while self.queue.full():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            return

        if self.queue.full():
            # При переполнении очереди медленного клиента сбрасываем устаревший хвост на свежем IDR-кадре
            if frag.is_keyframe:
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            else:
                # Пропускаем P-фрагмент, чтобы не раздувать память
                return
        try:
            self.queue.put_nowait(frag.data)
        except asyncio.QueueFull:
            pass

    def push_threadsafe(self, frag: StreamFragment | None) -> None:
        if self._closed or self.loop.is_closed():
            return
        try:
            self.loop.call_soon_threadsafe(self._enqueue_in_loop, frag)
        except RuntimeError:
            pass

    async def get(self) -> bytes | None:
        if self._closed and self.queue.empty():
            return None
        return await self.queue.get()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter_chunks()

    async def _iter_chunks(self) -> AsyncIterator[bytes]:
        try:
            while not self._closed:
                chunk = await self.queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub._unsubscribe(self)
        self.push_threadsafe(None)


class H264StreamHub:
    """
    Единый мультиплексор прямого H.264 fMP4 потока без перекодирования (-c:v copy, 0% CPU/GPU):
    - Держит 1 процесс FFmpeg RTSP -> fragmented MP4;
    - Выделяет init_segment (ftyp + moov) и RFC 6381 строку кодека avc1.XXXXXX;
    - Кэширует последний ключевой GOP (moof + mdat от свежего IDR-кадра) для старта < 100 мс;
    - Вещает фрагменты всем подключённым WebSocket / HTTP клиентам через asyncio.Queue.
    """

    def __init__(
        self,
        rtsp_url: str = "",
        frag_duration_us: int = 200_000,
        queue_maxsize: int = 24,
        ffmpeg_bin: str | None = None,
        read_timeout_sec: float = 12.0,
        fallback_rtsp_url: str = "",
        target_size: tuple[int, int] | None = None,
        fps_limit: int = 0,
        bitrate_kbps: int = 0,
        hwaccel: str = "auto",
        on_demand: bool = False,
        idle_stop_sec: float = 15.0,
        quality_label: str = "1080p",
    ) -> None:
        self._rtsp_url = rtsp_url.strip()
        self._fallback_rtsp_url = fallback_rtsp_url.strip()
        self._frag_duration_us = max(40_000, int(frag_duration_us))
        self._queue_maxsize = max(6, int(queue_maxsize))
        self._explicit_ffmpeg = ffmpeg_bin
        self._read_timeout_sec = max(3.0, float(read_timeout_sec))
        self._target_size = target_size
        self._fps_limit = max(0, int(fps_limit))
        self._bitrate_kbps = max(0, int(bitrate_kbps))
        mode = str(hwaccel or "auto").strip().lower()
        self._hwaccel = mode if mode in ("auto", "nvenc", "nvdec", "off") else "auto"
        self._on_demand = bool(on_demand)
        self._idle_stop_sec = max(2.0, float(idle_stop_sec))
        self._quality_label = str(quality_label or "1080p")

        self._lock = threading.RLock()
        self._proc_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._last_io_mono: float = 0.0

        self._init_segment: bytes | None = None
        self._codec_string: str = DEFAULT_CODEC_STRING
        self._mime_type: str = f'video/mp4; codecs="{DEFAULT_CODEC_STRING}"'
        self._width: int = 0
        self._height: int = 0

        self._gop_cache: list[StreamFragment] = []
        self._gop_bytes: int = 0
        self._seq: int = 0
        self._last_frag_mono: float = 0.0
        self._last_error: str = ""
        self._ready_event = threading.Event()
        self._wake_event = threading.Event()

        self._using_fallback = False
        self._nvenc_failed = self._hwaccel == "off"
        self._active_requested = False
        self._last_consumer_mono: float = 0.0

        # Временное состояние сборщика боксов
        self._parser = IsoBmffStreamParser()
        self._pending_ftyp: bytes = b""
        self._pending_prefix: bytearray = bytearray()
        self._pending_raw_moof: bytes | None = None

        self._subscribers: set[StreamSubscription] = set()
        self._sinks: set[Callable[[bytes | None, StreamFragment, list[StreamFragment]], None]] = set()
        self._started = False
        self._stop_event = threading.Event()
        self._reload_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._init_segment is not None

    @property
    def is_running(self) -> bool:
        with self._lock:
            fresh = self._last_frag_mono > 0.0 and (time.monotonic() - self._last_frag_mono) < 15.0
            return self._init_segment is not None and (fresh or self._thread is not None)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    @property
    def clients_count(self) -> int:
        return self.client_count

    @property
    def stream_clients(self) -> int:
        return self.client_count

    @property
    def codec_string(self) -> str:
        with self._lock:
            return self._codec_string

    @property
    def mime_type(self) -> str:
        with self._lock:
            return self._mime_type

    @property
    def width(self) -> int:
        with self._lock:
            return self._width

    @property
    def height(self) -> int:
        with self._lock:
            return self._height

    @property
    def resolution(self) -> list[int]:
        with self._lock:
            return [self._width, self._height]

    @property
    def init_segment(self) -> bytes | None:
        with self._lock:
            return self._init_segment

    @property
    def gop_cache(self) -> list[bytes]:
        with self._lock:
            return [frag.data for frag in self._gop_cache]

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def get_init_message(self) -> dict[str, Any]:
        """Возвращает стартовый JSON-пакет для WebSocket /ws/stream."""
        with self._lock:
            return {
                "type": "init",
                "mime": self._mime_type,
                "codecs": self._codec_string,
                "width": self._width,
                "height": self._height,
                "quality": self._quality_label,
            }

    def add_fragment_sink(
        self,
        cb: Callable[[bytes | None, StreamFragment, list[StreamFragment]], None],
    ) -> None:
        with self._lock:
            self._sinks.add(cb)
            self._active_requested = True
            self._last_consumer_mono = time.monotonic()
            self._wake_event.set()

    def remove_fragment_sink(
        self,
        cb: Callable[[bytes | None, StreamFragment, list[StreamFragment]], None],
    ) -> None:
        with self._lock:
            self._sinks.discard(cb)
            if not self._subscribers and not self._sinks:
                self._active_requested = False
                self._last_consumer_mono = time.monotonic()

    def ensure_active(self) -> None:
        """Будит on_demand-поток при обращении клиента или приёмника фрагментов."""
        with self._lock:
            self._active_requested = True
            self._last_consumer_mono = time.monotonic()
            self._wake_event.set()
            should_start = (
                self._started
                and not self._stop_event.is_set()
                and (_is_rtsp_url(self._rtsp_url) or _is_rtsp_url(self._fallback_rtsp_url))
                and (self._thread is None or not self._thread.is_alive())
            )
        if should_start:
            self.start()

    def reset_stream_state(self) -> None:
        with self._lock:
            self._parser.reset()
            self._pending_ftyp = b""
            self._pending_prefix.clear()
            self._pending_raw_moof = None

    def feed_bytes(self, chunk: bytes | bytearray | memoryview) -> list[StreamFragment]:
        """
        Разбирает очередной кусок fMP4-потока, обновляет init_segment / GOP-кэш
        и рассылает готовые фрагменты moof+mdat активным подписчикам и DVR-приёмникам.
        """
        emitted: list[StreamFragment] = []
        boxes = self._parser.feed(chunk)

        for btype, full_box in boxes:
            if btype == b"ftyp":
                self._pending_ftyp = full_box
                self._pending_prefix.clear()
                self._pending_raw_moof = None
            elif btype == b"moov":
                meta = parse_moov_metadata(full_box)
                init_bytes = self._pending_ftyp + full_box
                with self._lock:
                    self._init_segment = init_bytes
                    self._codec_string = str(meta["codecs"])
                    self._mime_type = str(meta["mime"])
                    if int(meta["width"]) > 0 and int(meta["height"]) > 0:
                        self._width = int(meta["width"])
                        self._height = int(meta["height"])
                    self._last_error = ""
                    self._ready_event.set()
            elif btype in (b"styp", b"sidx"):
                self._pending_prefix.extend(full_box)
            elif btype == b"moof":
                self._pending_prefix.extend(full_box)
                self._pending_raw_moof = full_box
            elif btype == b"mdat" and self._pending_raw_moof is not None:
                frag_bytes = bytes(self._pending_prefix) + full_box
                raw_moof = self._pending_raw_moof
                self._pending_prefix.clear()
                self._pending_raw_moof = None

                is_kf = is_fragment_keyframe(raw_moof, full_box)
                now_mono = time.monotonic()

                with self._lock:
                    self._seq += 1
                    frag = StreamFragment(
                        seq=self._seq,
                        timestamp=now_mono,
                        data=frag_bytes,
                        is_keyframe=is_kf,
                    )
                    self._last_frag_mono = now_mono
                    if is_kf:
                        self._gop_cache = [frag]
                        self._gop_bytes = len(frag_bytes)
                    elif self._gop_cache and (
                        len(self._gop_cache) < MAX_GOP_FRAGMENTS
                        and (self._gop_bytes + len(frag_bytes)) <= MAX_GOP_BYTES
                    ):
                        self._gop_cache.append(frag)
                        self._gop_bytes += len(frag_bytes)

                    subs_snapshot = list(self._subscribers)
                    sinks_snapshot = list(self._sinks)
                    init_snapshot = self._init_segment
                    gop_snapshot = list(self._gop_cache)

                for sub in subs_snapshot:
                    sub.push_threadsafe(frag)
                for sink in sinks_snapshot:
                    try:
                        sink(init_snapshot, frag, gop_snapshot)
                    except Exception:
                        pass
                emitted.append(frag)

        return emitted

    def subscribe(
        self,
        include_gop: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> StreamSubscription:
        """
        Создаёт клиентскую подписку с мгновенным предзаполнением очереди последним ключевым GOP.
        Должна вызываться внутри работающего asyncio event loop (или с явной передачей loop).
        """
        ev_loop = loop if loop is not None else asyncio.get_running_loop()
        with self._lock:
            gop_snapshot = list(self._gop_cache) if include_gop else []
            eff_maxsize = max(self._queue_maxsize, len(gop_snapshot) + 8)
            sub = StreamSubscription(self, ev_loop, queue_maxsize=eff_maxsize)
            for frag in gop_snapshot:
                try:
                    sub.queue.put_nowait(frag.data)
                except asyncio.QueueFull:
                    break
            self._subscribers.add(sub)
            self._active_requested = True
            self._last_consumer_mono = time.monotonic()
            self._wake_event.set()
        return sub

    def _unsubscribe(self, sub: StreamSubscription) -> None:
        with self._lock:
            self._subscribers.discard(sub)
            if not self._subscribers and not self._sinks:
                self._active_requested = False
                self._last_consumer_mono = time.monotonic()

    async def wait_ready(self, timeout: float = 5.0) -> bool:
        """Асинхронно ожидает готовности init_segment (ftyp + moov) до timeout секунд."""
        self.ensure_active()
        if self.is_ready:
            return True
        deadline = time.monotonic() + max(0.05, float(timeout))
        while time.monotonic() < deadline:
            if self.is_ready or self._stop_event.is_set():
                return self.is_ready
            await asyncio.sleep(0.05)
        return self.is_ready

    async def iter_fmp4_stream(self, wait_timeout: float = 5.0) -> AsyncIterator[bytes]:
        """Готовый генератор непрерывного HTTP fMP4 потока (init_segment + GOP + live) для GET /api/stream.mp4."""
        ready = await self.wait_ready(timeout=wait_timeout)
        init_seg = self.init_segment
        if not ready or not init_seg:
            return
        async with self.subscribe(include_gop=True) as sub:
            yield init_seg
            async for frag_bytes in sub:
                yield frag_bytes

    def _kill_active_proc(self) -> None:
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass

    def reload_url(self, new_rtsp_url: str, fallback_rtsp_url: str | None = None) -> None:
        cleaned = new_rtsp_url.strip()
        with self._lock:
            self._rtsp_url = cleaned
            if fallback_rtsp_url is not None:
                self._fallback_rtsp_url = fallback_rtsp_url.strip()
            self._using_fallback = False
            self._init_segment = None
            self._gop_cache.clear()
            self._gop_bytes = 0
            self._ready_event.clear()
            self._reload_event.set()
            self._wake_event.set()
            should_spawn = (
                self._started
                and (_is_rtsp_url(cleaned) or _is_rtsp_url(self._fallback_rtsp_url))
                and not self._stop_event.is_set()
                and (self._thread is None or not self._thread.is_alive())
            )
        self._kill_active_proc()
        if should_spawn:
            self.start()

    def start(self) -> None:
        with self._lock:
            self._started = True
            has_any_url = _is_rtsp_url(self._rtsp_url) or _is_rtsp_url(self._fallback_rtsp_url)
            if not has_any_url:
                self._last_error = "RTSP_URL не задан для прямого H.264 потока"
                return
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._reload_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name=f"h264-stream-hub-{self._quality_label}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._reload_event.set()
        self._wake_event.set()
        self._kill_active_proc()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        with self._lock:
            subs = list(self._subscribers)
            self._subscribers.clear()
        for sub in subs:
            sub.close()

    def _resolve_ffmpeg(self) -> str | None:
        if self._explicit_ffmpeg:
            return self._explicit_ffmpeg
        return find_ffmpeg_bin()

    def _should_sleep_on_demand_locked(self) -> bool:
        if not self._on_demand:
            return False
        if self._subscribers or self._sinks:
            return False
        if self._active_requested and (time.monotonic() - self._last_consumer_mono) <= self._idle_stop_sec:
            return False
        return True

    def _run_loop(self) -> None:
        backoff = 1.5
        read_buf = bytearray(65536)
        consecutive_no_frag = 0

        try:
            while not self._stop_event.is_set():
                if self._reload_event.is_set():
                    self._reload_event.clear()
                    backoff = 1.5
                    consecutive_no_frag = 0

                with self._lock:
                    if self._should_sleep_on_demand_locked():
                        if self._init_segment is not None:
                            self._init_segment = None
                            self._gop_cache.clear()
                            self._gop_bytes = 0
                            self._ready_event.clear()
                        self._active_requested = False
                        self._wake_event.clear()
                        need_sleep = True
                    else:
                        need_sleep = False
                if need_sleep:
                    self._wake_event.wait(0.5)
                    continue

                with self._lock:
                    if self._using_fallback and _is_rtsp_url(self._fallback_rtsp_url):
                        url = self._fallback_rtsp_url
                        eff_target = self._target_size or (640, 360)
                        eff_fps = self._fps_limit or 10
                        eff_bitrate = self._bitrate_kbps or 400
                    elif _is_rtsp_url(self._rtsp_url):
                        url = self._rtsp_url
                        eff_target = self._target_size
                        eff_fps = self._fps_limit
                        eff_bitrate = self._bitrate_kbps
                    elif _is_rtsp_url(self._fallback_rtsp_url):
                        url = self._fallback_rtsp_url
                        eff_target = self._target_size or (640, 360)
                        eff_fps = self._fps_limit or 10
                        eff_bitrate = self._bitrate_kbps or 400
                    else:
                        url = ""
                        eff_target = self._target_size
                        eff_fps = self._fps_limit
                        eff_bitrate = self._bitrate_kbps

                if not _is_rtsp_url(url):
                    self._stop_event.wait(1.0)
                    continue

                ffmpeg_bin = self._resolve_ffmpeg()
                if not ffmpeg_bin:
                    with self._lock:
                        self._last_error = "ffmpeg не найден для H264StreamHub"
                    self._stop_event.wait(5.0)
                    continue

                self.reset_stream_state()
                tried_nvenc = bool(eff_target is not None and not self._nvenc_failed)
                if eff_target is None:
                    cmd = build_passthrough_fmp4_cmd(
                        ffmpeg_bin=ffmpeg_bin,
                        rtsp_url=url,
                        frag_duration_us=self._frag_duration_us,
                    )
                else:
                    cmd = build_transcode_fmp4_cmd(
                        ffmpeg_bin=ffmpeg_bin,
                        rtsp_url=url,
                        target_size=eff_target,
                        fps_limit=eff_fps,
                        bitrate_kbps=eff_bitrate,
                        use_nvenc=tried_nvenc,
                        frag_duration_us=self._frag_duration_us,
                    )

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
                    with self._lock:
                        self._last_error = f"Ошибка старта FFmpeg ({self._quality_label}): {_redact_url(str(exc))}"
                    self._stop_event.wait(backoff)
                    backoff = min(10.0, backoff * 1.5)
                    continue

                with self._proc_lock:
                    self._proc = proc
                self._last_io_mono = time.monotonic()

                stderr_tail: deque[str] = deque(maxlen=15)
                closed_flag = threading.Event()
                idle_killed = threading.Event()

                def _drain_err(err_io: IO[bytes] | None) -> None:
                    if err_io is None:
                        return
                    try:
                        while not closed_flag.is_set():
                            line = err_io.readline()
                            if not line:
                                break
                            txt = _redact_url(line.decode("utf-8", errors="replace").strip())
                            if txt:
                                stderr_tail.append(txt)
                    except Exception:
                        pass

                def _watchdog(target_proc: subprocess.Popen[bytes]) -> None:
                    while not closed_flag.wait(1.0):
                        if target_proc.poll() is not None:
                            return
                        with self._lock:
                            sleep_idle = self._should_sleep_on_demand_locked()
                        if sleep_idle:
                            idle_killed.set()
                            self._kill_active_proc()
                            return
                        if (time.monotonic() - self._last_io_mono) > self._read_timeout_sec:
                            logger.warning(
                                "Таймаут H264StreamHub [%s] (>%.1f с без пакетов), реконнект",
                                self._quality_label,
                                self._read_timeout_sec,
                            )
                            self._kill_active_proc()
                            return

                threading.Thread(target=_drain_err, args=(proc.stderr,), daemon=True).start()
                threading.Thread(target=_watchdog, args=(proc,), daemon=True).start()

                got_any_fragment = False
                stdout = proc.stdout
                mv = memoryview(read_buf)
                try:
                    while stdout is not None and not self._stop_event.is_set() and not self._reload_event.is_set():
                        read_fn = getattr(stdout, "readinto", None)
                        if callable(read_fn):
                            n = read_fn(mv)
                        else:
                            raw = stdout.read(len(read_buf))
                            n = len(raw) if raw else 0
                            if n:
                                mv[:n] = raw
                        if not n:
                            break
                        self._last_io_mono = time.monotonic()
                        frags = self.feed_bytes(mv[: int(n)])
                        if frags:
                            got_any_fragment = True
                            consecutive_no_frag = 0
                            backoff = 1.5
                except Exception as exc:
                    with self._lock:
                        self._last_error = f"Сбой чтения fMP4 пайпа: {_redact_url(str(exc))}"
                finally:
                    closed_flag.set()
                    self._kill_active_proc()

                if self._stop_event.is_set() or self._reload_event.is_set() or idle_killed.is_set():
                    continue

                if stderr_tail:
                    with self._lock:
                        self._last_error = stderr_tail[-1]
                if not got_any_fragment:
                    if tried_nvenc:
                        self._nvenc_failed = True
                        logger.warning(
                            "h264_nvenc недоступен для профиля %s, переключение на CPU libx264",
                            self._quality_label,
                        )
                        self._stop_event.wait(0.2)
                        continue
                    consecutive_no_frag += 1
                    if (
                        consecutive_no_frag >= 3
                        and self._fallback_rtsp_url
                        and _is_rtsp_url(self._fallback_rtsp_url)
                        and not self._using_fallback
                    ):
                        self._using_fallback = True
                        consecutive_no_frag = 0
                        logger.warning(
                            "Субпоток %s недоступен, переключение на масштабирование основного потока",
                            self._quality_label,
                        )
                        continue
                    self._stop_event.wait(backoff)
                    backoff = min(10.0, backoff * 1.5)
                else:
                    self._stop_event.wait(0.8)
        finally:
            self._kill_active_proc()
