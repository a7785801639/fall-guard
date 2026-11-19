from __future__ import annotations

import asyncio
import base64
import json
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, derive_rtsp_sub_url
from app.dvr import DvrRecorderStore
from app.engine import FallGuardEngine
from app.geometry import PoseObservation
from app.motion_log import MotionLogStore
from app.state_machine import AlertEvent, AlertSeverity, PatientState
from app.stream_hub import (
    H264StreamHub,
    StreamFragment,
    build_transcode_fmp4_cmd,
    parse_moov_metadata,
    rebase_moof_tfdt,
)
from app.timeline import TimelineStore
from app.web import create_web_app


def _iso_box(btype: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), btype) + body


def _make_synthetic_init_segment(width: int = 640, height: int = 360) -> bytes:
    ftyp = _iso_box(b"ftyp", b"isom\x00\x00\x02\x00isomiso5")
    avcc = _iso_box(b"avcC", b"\x01\x64\x00\x1f\xff\xe1\x00\x04\x67\x64\x00\x1f\x01\x00\x02\x68\xee")
    # VisualSampleEntry body: 24B + width(2B) + height(2B) + 50B padding = 78B перед avcC
    vse_body = (
        b"\x00" * 24
        + struct.pack(">HH", width, height)
        + b"\x00" * 50
        + avcc
    )
    avc1 = _iso_box(b"avc1", vse_body)
    stsd = _iso_box(b"stsd", b"\x00\x00\x00\x00\x00\x00\x00\x01" + avc1)
    stbl = _iso_box(b"stbl", stsd)
    minf = _iso_box(b"minf", stbl)
    mdia = _iso_box(b"mdia", minf)
    trak = _iso_box(b"trak", mdia)
    moov = _iso_box(b"moov", trak)
    return ftyp + moov


def _make_synthetic_fragment(
    tfdt_val: int,
    *,
    is_keyframe: bool = True,
    payload_size: int = 64,
    v1: bool = False,
) -> bytes:
    # tfhd (tf_flags=0x020000 default_sample_flags present, version=0)
    tfhd = _iso_box(b"tfhd", struct.pack(">II", 0x020000, 1))
    if v1:
        tfdt = _iso_box(b"tfdt", struct.pack(">IQ", 0x01000000, tfdt_val))
    else:
        tfdt = _iso_box(b"tfdt", struct.pack(">II", 0, tfdt_val))
    # trun с first_sample_flags (0x000004): depends_on=2 (sync) -> 0x02000000; P-frame -> 0x01010000
    s_flags = 0x02000000 if is_keyframe else 0x01010000
    trun = _iso_box(b"trun", struct.pack(">III", 0x000004, 1, s_flags))
    traf = _iso_box(b"traf", tfhd + tfdt + trun)
    moof = _iso_box(b"moof", traf)
    nal_type = 0x65 if is_keyframe else 0x41
    nal = struct.pack(">I", max(4, payload_size)) + bytes([nal_type]) + (b"\xAB" * max(3, payload_size - 1))
    mdat = _iso_box(b"mdat", nal)
    return moof + mdat


def _build_test_settings(tmp_path: Path, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        rtsp_url="",
        tg_bot_token="123456:TEST_ISOLATED_TOKEN",
        tg_chat_ids=("100001",),
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        dvr_dir=tmp_path / "dvr_root",
        web_host="127.0.0.1",
        web_port=8090,
        infer_fps=5.0,
        capture_fps=6.0,
        pre_event_seconds=5.0,
        post_event_seconds=4.0,
        pose_model="yolo11s-pose.pt",
        device="cpu",
        pose_conf=0.48,
        bed_lock_seconds=20.0,
        initial_bed_lock_seconds=5.0,
        sit_confirm_seconds=1.8,
        leave_confirm_seconds=2.0,
        fall_confirm_seconds=1.8,
        missing_confirm_seconds=12.0,
        alert_cooldown_seconds=45.0,
        repeat_fall_alert_seconds=15.0,
        max_fall_repeats=3,
        suppress_sitting_when_caregiver=True,
        face_match_threshold=0.38,
        tg_bot_username="NoFallenAi",
        frame_rotate=180,
        frame_flip_h=False,
        low_light_enhance=False,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def test_derive_rtsp_sub_url_and_settings_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Автовывод субпотока 102 из 101 с сохранением query-параметров
    main_u = "rtsp://camuser:campass@192.0.2.10:554/Streaming/Channels/101?transportmode=unicast"
    assert (
        derive_rtsp_sub_url(main_u)
        == "rtsp://camuser:campass@192.0.2.10:554/Streaming/Channels/102?transportmode=unicast"
    )
    # 2. Явный RTSP_SUB_URL перекрывает автовывод
    explicit = "rtsp://camuser:campass@192.0.2.10:554/ch2/sub"
    assert derive_rtsp_sub_url(main_u, explicit) == explicit
    # 3. Нестандартный URL без паттерна возвращает ""
    assert derive_rtsp_sub_url("rtsp://camuser:campass@192.0.2.10:554/h264/live") == ""

    # 4. Загрузка DVR_DIR и квот из .env
    env_file = tmp_path / ".env"
    dvr_target = tmp_path / "custom_dvr"
    env_file.write_text(
        "\n".join(
            [
                f"RTSP_URL={main_u}",
                "TG_BOT_TOKEN=111222:FAKE_TOKEN_FOR_TEST",
                f"DVR_DIR={dvr_target}",
                "DVR_RETENTION_DAYS=7",
                "DVR_MIN_FREE_GB=5.0",
                "DVR_MAX_GB=95.0",
            ]
        ),
        encoding="utf-8",
    )
    for k in ("RTSP_URL", "RTSP_SUB_URL", "TG_BOT_TOKEN", "DVR_DIR", "DVR_RETENTION_DAYS", "DVR_MIN_FREE_GB", "DVR_MAX_GB"):
        monkeypatch.delenv(k, raising=False)
    loaded = Settings.from_env(dotenv_path=env_file)
    assert loaded.dvr_dir == dvr_target
    assert loaded.dvr_retention_days == pytest.approx(7.0)
    assert loaded.dvr_min_free_gb == pytest.approx(5.0)
    assert loaded.dvr_max_gb == pytest.approx(95.0)
    assert "/Streaming/Channels/102" in loaded.rtsp_sub_url


def test_rebase_moof_tfdt_and_transcode_cmd_and_ondemand(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd_nv = build_transcode_fmp4_cmd(
        "/usr/bin/ffmpeg",
        "rtsp://cam:pw@192.0.2.10/Streaming/Channels/101",
        target_size=(1280, 720),
        fps_limit=15,
        bitrate_kbps=1400,
        use_nvenc=True,
    )
    assert "-vf" in cmd_nv and "scale=1280:720" in cmd_nv
    assert "h264_nvenc" in cmd_nv
    assert "-g" in cmd_nv and "15" in cmd_nv

    cmd_cpu = build_transcode_fmp4_cmd(
        "/usr/bin/ffmpeg",
        "rtsp://cam:pw@192.0.2.10/Streaming/Channels/101",
        target_size=(640, 360),
        use_nvenc=False,
    )
    assert "libx264" in cmd_cpu and "scale=640:360" in cmd_cpu

    # Перебазирование tfdt v0 и v1 (первый фрагмент становится 0, второй хранит дельту)
    f1 = _make_synthetic_fragment(900_000, is_keyframe=True, v1=False)
    f2 = _make_synthetic_fragment(945_000, is_keyframe=False, v1=False)
    p1, base = rebase_moof_tfdt(f1, None)
    p2, _ = rebase_moof_tfdt(f2, base)
    assert base == 900_000
    assert p1 != f1
    # В p1 tfdt -> 0, в p2 tfdt -> 45_000
    idx1 = p1.find(b"tfdt")
    idx2 = p2.find(b"tfdt")
    assert struct.unpack_from(">I", p1, idx1 + 8)[0] == 0
    assert struct.unpack_from(">I", p2, idx2 + 8)[0] == 45_000

    # Проверка v1 (uint64)
    f_v1 = _make_synthetic_fragment(12_000_000_000, is_keyframe=True, v1=True)
    p_v1, base_v1 = rebase_moof_tfdt(f_v1, None)
    assert base_v1 == 12_000_000_000
    idx_v1 = p_v1.find(b"tfdt")
    assert struct.unpack_from(">Q", p_v1, idx_v1 + 8)[0] == 0

    # Жизненный цикл on_demand=True: без wait_ready / subscribe FFmpeg не запускается
    spawn_calls: list[list[str]] = []

    class _DummyProc:
        def __init__(self) -> None:
            self.stderr = None
            self.stdout = None
            self.killed = False

        def poll(self) -> int | None:
            return None if not self.killed else 0

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(args: list[str], **kwargs: Any) -> _DummyProc:
        spawn_calls.append(args)
        return _DummyProc()

    monkeypatch.setattr("app.stream_hub.subprocess.Popen", fake_popen)
    hub_od = H264StreamHub(
        rtsp_url="rtsp://cam:pw@192.0.2.10:554/Streaming/Channels/101",
        target_size=(1280, 720),
        ffmpeg_bin="/usr/bin/ffmpeg",
        on_demand=True,
        idle_stop_sec=2.0,
        quality_label="720p",
    )
    hub_od.start()
    try:
        time.sleep(0.15)
        assert len(spawn_calls) == 0  # спящий режим без зрителей
        hub_od.ensure_active()
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not spawn_calls:
            time.sleep(0.03)
        assert len(spawn_calls) >= 1
    finally:
        hub_od.stop()


def test_motion_log_store_adaptive_rate_phantoms_buckets_and_prune(tmp_path: Path) -> None:
    ml = MotionLogStore(tmp_path / "dvr", retention_days=7.0, flush_interval_sec=10.0, flush_batch_size=100)
    now = time.time()

    # 1. В полном покое (optical_motion < 0.010, IN_BED) шаг >= 5.0 с
    assert ml.record(
        ts=now - 20.0,
        state="in_bed",
        status_reason="спокойно",
        optical_motion=0.002,
        patient_in_bed_latched=True,
        tracks_ui=[{"track_id": 1, "role": "patient", "reason": "bed_sleep_lock", "bbox": [0.2, 0.3, 0.6, 0.7], "keypoints": [], "angle_deg": 75.0}],
    )
    # Через 1.5 с при неизменном покое запись пропускается
    assert not ml.record(
        ts=now - 18.5,
        state="in_bed",
        optical_motion=0.003,
        patient_in_bed_latched=True,
        tracks_ui=[{"track_id": 1, "role": "patient", "reason": "bed_sleep_lock", "bbox": [0.2, 0.3, 0.6, 0.7], "keypoints": [], "angle_deg": 75.0}],
    )
    # Через 5.5 с — пишется heartbeat
    assert ml.record(
        ts=now - 14.5,
        state="in_bed",
        optical_motion=0.003,
        patient_in_bed_latched=True,
        tracks_ui=[{"track_id": 1, "role": "patient", "reason": "bed_sleep_lock", "bbox": [0.2, 0.3, 0.6, 0.7], "keypoints": [], "angle_deg": 75.0}],
    )

    # 2. Вспышка движения + отсеянный фантом в raw_observations (ign) -> пишется немедленно и с высоким рейтом (0.25 с)
    phantom_obs = PoseObservation(
        track_id=99,
        bbox_norm=(0.55, 0.35, 0.68, 0.72),
        keypoints_norm=tuple([(0.6, 0.4 + i * 0.01, 0.6) for i in range(17)]),
        conf=0.61,
    )
    assert ml.record(
        ts=now - 14.0,
        state="sitting_up",
        status_reason="приподнялась",
        optical_motion=0.032,
        patient_in_bed_latched=True,
        timers={"sit": 0.9, "lv": 0.0},
        tracks_ui=[{"track_id": 1, "role": "patient", "reason": "bed_sleep_lock", "bbox": [0.2, 0.3, 0.6, 0.7], "keypoints": [], "angle_deg": 31.5}],
        raw_observations=[phantom_obs],
    )
    ml.flush()

    recs = ml.query_range(now - 30.0, now)
    assert len(recs) == 3
    assert any(r.get("ign") and r["ign"][0]["id"] == 99 for r in recs)

    buckets = ml.motion_buckets(now - 3600.0, now + 60.0, bucket_sec=300.0)
    assert len(buckets) >= 1
    assert any(b["ign_hits"] >= 1 and b["max_opt"] >= 0.03 for b in buckets)

    # 3. Очистка старше retention_days (папка 9 дней назад удаляется, сегодняшняя остаётся)
    old_day = datetime.fromtimestamp(now - 9 * 86400.0).strftime("%Y-%m-%d")
    old_dir = ml.root_dir / old_day
    old_dir.mkdir(parents=True, exist_ok=True)
    (old_dir / "12.jsonl.gz").write_bytes(b"old")
    removed_dirs = ml.prune(retention_days=7.0, now_ts=now)
    assert removed_dirs >= 1
    assert not old_dir.exists()
    today_dir = ml.root_dir / datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    assert today_dir.exists()


def test_dvr_recorder_adaptive_hi_retention_and_free_space_guard(tmp_path: Path) -> None:
    ml = MotionLogStore(tmp_path / "dvr")
    free_state = {"free_bytes": 20 * (1024 ** 3)}

    store = DvrRecorderStore(
        dvr_dir=tmp_path / "dvr",
        segment_seconds=60,
        retention_days=7.0,
        max_gb=10.0,
        min_free_gb=5.0,
        motion_hold_seconds=10.0,
        motion_log=ml,
        enable_remux=False,
        disk_free_fn=lambda _p: free_state["free_bytes"],
    )
    store.start()
    try:
        init_low = _make_synthetic_init_segment(640, 360)
        init_hi = _make_synthetic_init_segment(1920, 1080)

        t0 = time.time() - 300.0
        mono0 = 1000.0
        # Запись непрерывного low-сегмента
        f_low1 = StreamFragment(seq=1, timestamp=mono0, data=_make_synthetic_fragment(500_000, is_keyframe=True), is_keyframe=True)
        f_low2 = StreamFragment(seq=2, timestamp=mono0 + 2.0, data=_make_synthetic_fragment(590_000, is_keyframe=False), is_keyframe=False)
        store.on_fragment("low", init_low, f_low1, [f_low1], width=640, height=360, wall_ts=t0, mono_ts=mono0)
        store.on_fragment("low", init_low, f_low2, [f_low1, f_low2], width=640, height=360, wall_ts=t0 + 2.0, mono_ts=mono0 + 2.0)

        # Пока нет активности — hi НЕ записывается
        f_hi_calm = StreamFragment(seq=10, timestamp=mono0 + 3.0, data=_make_synthetic_fragment(100_000, is_keyframe=True), is_keyframe=True)
        store.on_fragment("hi", init_hi, f_hi_calm, [f_hi_calm], width=1920, height=1080, wall_ts=t0 + 3.0, mono_ts=mono0 + 3.0)
        assert store.resolve_best_segment(t0 + 1.0, prefer_tier="auto")["tier"] == "low"

        # Появилось движение -> notify_activity() -> hi стартует с захватом pre-motion gop_cache!
        store.notify_activity(wall_ts=t0 + 5.0, hold_sec=12.0)
        pre_kf = StreamFragment(seq=11, timestamp=mono0 + 4.0, data=_make_synthetic_fragment(200_000, is_keyframe=True, payload_size=2048), is_keyframe=True)
        live_p = StreamFragment(seq=12, timestamp=mono0 + 5.5, data=_make_synthetic_fragment(245_000, is_keyframe=False, payload_size=2048), is_keyframe=False)
        store.on_fragment("hi", init_hi, live_p, [pre_kf, live_p], width=1920, height=1080, wall_ts=t0 + 5.5, mono_ts=mono0 + 5.5)

        best_auto = store.resolve_best_segment(t0 + 5.2, prefer_tier="auto")
        assert best_auto is not None and best_auto["tier"] == "hi"
        best_low = store.resolve_best_segment(t0 + 5.2, prefer_tier="low")
        assert best_low is not None and best_low["tier"] == "low"
    finally:
        store.stop()

    # Проверяем перегрузку порога свободного места: имитируем 4.0 ГБ свободно < min_free_gb=5.0 ГБ
    # В одном часе лежит и hi (~4 КБ) и low (< 1 КБ): сначала должен удалиться hi, затем low
    days_before = store.list_days()["days"]
    assert any(d["hi_count"] >= 1 and d["low_count"] >= 1 for d in days_before)

    free_state["free_bytes"] = 4 * (1024 ** 3)  # < 5 ГБ, reclaimed от 4 КБ не закроет 1 ГБ, поэтому удалит всё закрытое
    deleted = store.enforce_retention()
    assert deleted >= 2
    assert store.list_segments(t0 - 100, t0 + 500) == []


def test_timeline_7_days_and_web_api_quality_dvr_and_flag(tmp_path: Path) -> None:
    settings = _build_test_settings(tmp_path)
    engine = FallGuardEngine(settings)
    try:
        # 1. Расширение TimelineStore до 7 дней (604800 с)
        assert engine.timeline.window_sec == pytest.approx(604800.0)
        now = time.time()
        engine.timeline.record("in_bed", True, now_ts=now - 5 * 86400.0)
        engine.timeline.record("sitting_up", True, now_ts=now - 100.0)
        engine.timeline.record("in_bed", True, now_ts=now - 10.0)
        sl = engine.timeline.slice_range(now - 5 * 86400.0 - 60, now - 4 * 86400.0)
        assert len(sl["spans"]) >= 1

        # 2. Кадр и масштабирование JPEG / MJPEG / WS по ?quality=auto|360p|720p|1080p
        frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_bgr[:, :] = (40, 110, 60)
        ok_raw, enc_raw = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        assert ok_raw
        with engine._lock:
            engine._latest_raw_frame = frame_bgr
            engine._latest_jpeg = enc_raw.tobytes()

        # Подкладываем отдельные потоки fMP4 в stream_hub (1080p) и stream_hub_low (360p)
        init_1080 = _make_synthetic_init_segment(1920, 1080)
        frag_1080 = _make_synthetic_fragment(1000, is_keyframe=True, payload_size=32)
        engine.stream_hub.feed_bytes(init_1080 + frag_1080)

        init_360 = _make_synthetic_init_segment(640, 360)
        frag_360 = _make_synthetic_fragment(1000, is_keyframe=True, payload_size=24)
        engine.stream_hub_low.feed_bytes(init_360 + frag_360)

        # Пишем DVR сегмент и лог движения за момент now - 5.0
        engine.motion_log.record(
            ts=now - 5.0,
            state="left_bed",
            status_reason="встала с кровати",
            optical_motion=0.041,
            timers={"lv": 1.2},
            tracks_ui=[{"track_id": 7, "role": "patient", "reason": "reid", "bbox": [0.3, 0.2, 0.5, 0.8], "keypoints": [], "angle_deg": 8.0}],
            force=True,
        )
        engine.motion_log.flush()
        f_dvr1 = StreamFragment(seq=1, timestamp=10.0, data=_make_synthetic_fragment(100_000, is_keyframe=True, payload_size=512), is_keyframe=True)
        f_dvr2 = StreamFragment(seq=2, timestamp=12.0, data=_make_synthetic_fragment(145_000, is_keyframe=False, payload_size=512), is_keyframe=False)
        engine.dvr.on_fragment("low", init_360, f_dvr1, [f_dvr1], width=640, height=360, wall_ts=now - 6.0, mono_ts=10.0)
        engine.dvr.on_fragment("low", init_360, f_dvr2, [f_dvr1, f_dvr2], width=640, height=360, wall_ts=now - 2.0, mono_ts=12.0)

        app = create_web_app(engine)
        client = TestClient(app)

        # Проверка /api/status?window_sec=604800
        st_r = client.get("/api/status?window_sec=604800")
        assert st_r.status_code == 200
        st_json = st_r.json()
        assert st_json["timeline"]["window_sec"] == 604800
        assert st_json["video_qualities"] == ["auto", "360p", "720p", "1080p"]
        assert st_json["dvr"]["enabled"] is True

        # Проверка масштабирования /api/snapshot.jpg для auto (Android -> 360p, Desktop -> 1080p)
        snap_mob = client.get("/api/snapshot.jpg?quality=auto", headers={"User-Agent": "Mozilla/5.0 (Linux; Android 14)"})
        assert snap_mob.status_code == 200
        assert snap_mob.headers.get("x-video-quality") == "360p"
        mob_img = cv2.imdecode(np.frombuffer(snap_mob.content, np.uint8), cv2.IMREAD_COLOR)
        assert mob_img is not None and mob_img.shape[:2] == (360, 640)

        snap_720 = client.get("/api/snapshot.jpg?quality=720p")
        assert snap_720.status_code == 200
        assert snap_720.headers.get("x-video-quality") == "720p"
        mid_img = cv2.imdecode(np.frombuffer(snap_720.content, np.uint8), cv2.IMREAD_COLOR)
        assert mid_img is not None and mid_img.shape[:2] == (720, 1280)

        snap_pc = client.get("/api/snapshot.jpg?quality=auto", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64)"})
        assert snap_pc.status_code == 200
        assert snap_pc.headers.get("x-video-quality") == "1080p"
        pc_img = cv2.imdecode(np.frombuffer(snap_pc.content, np.uint8), cv2.IMREAD_COLOR)
        assert pc_img is not None and pc_img.shape[:2] == (1080, 1920)

        # Проверка /ws/stream?quality=360p (отдаёт width=640) и ?quality=1080p (отдаёт width=1920)
        with client.websocket_connect("/ws/stream?quality=360p") as ws360:
            init_msg = ws360.receive_json()
            assert init_msg["width"] == 640 and init_msg["height"] == 360
            assert init_msg["quality"] == "360p"
            b_init = ws360.receive_bytes()
            assert b"moov" in b_init

        with client.websocket_connect("/ws/stream?quality=1080p") as ws1080:
            init_msg_hi = ws1080.receive_json()
            assert init_msg_hi["width"] == 1920 and init_msg_hi["height"] == 1080

        # Проверка DVR API (/days, /day, /resolve, /segments с HTTP Range 206, /motion, /flag)
        days_r = client.get("/api/dvr/days")
        assert days_r.status_code == 200
        assert len(days_r.json()["days"]) >= 7

        today_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        day_r = client.get(f"/api/dvr/day?date={today_str}")
        assert day_r.status_code == 200
        day_j = day_r.json()
        assert len(day_j["segments"]) >= 1
        assert len(day_j["motion_buckets"]) >= 1

        res_r = client.get(f"/api/dvr/resolve?ts={now - 4.0:.2f}")
        assert res_r.status_code == 200
        seg = res_r.json()["segment"]
        assert seg is not None and seg["tier"] == "low"

        seg_url = seg["url"]
        range_r = client.get(seg_url, headers={"Range": "bytes=0-31"})
        assert range_r.status_code == 206
        assert len(range_r.content) == 32

        # Защита от path traversal
        assert client.get("/api/dvr/segments/..%5Cconfig.py.mp4").status_code == 404

        mot_r = client.get(f"/api/dvr/motion?start_ts={now - 60:.1f}&end_ts={now + 5:.1f}")
        assert mot_r.status_code == 200
        assert any(fr["st"] == "left_bed" for fr in mot_r.json()["frames"])

        # Отметка ошибки/фантома с кадром на моменте архива -> пакет в data/feedback/dvr_*/
        snap_b64 = base64.b64encode(enc_raw.tobytes()).decode("ascii")
        flag_r = client.post(
            "/api/dvr/flag",
            json={
                "ts": now - 4.0,
                "kind": "phantom_visitor",
                "note": "Фантом у двери",
                "snapshot_b64": f"data:image/jpeg;base64,{snap_b64}",
            },
        )
        assert flag_r.status_code == 200
        flag_j = flag_r.json()
        assert flag_j["ok"] is True
        fid = flag_j["flag"]["alert_id"]
        assert fid.startswith("dvr_")
        pkg_dir = settings.data_dir / "feedback" / fid
        assert (pkg_dir / "context.json").is_file()
        assert (pkg_dir / "motion_slice.json").is_file()
        assert (pkg_dir / "annotated.jpg").is_file()

        # Флаг отражается в /api/dvr/day -> flags и в /api/feedback
        day2_r = client.get(f"/api/dvr/day?date={today_str}")
        assert any(f["alert_id"] == fid for f in day2_r.json()["flags"])

        # Стандартный report_false_alert также сохраняет motion_slice.json
        ev = AlertEvent(
            state=PatientState.LEFT_BED,
            severity=AlertSeverity.WARNING,
            title="Встала",
            description="Тест",
            timestamp=time.monotonic(),
            patient_track_id=7,
        )
        engine.feedback.register_recent_alert(
            "alert_motion_test",
            ev,
            annotated_jpeg=enc_raw.tobytes(),
            raw_jpeg=enc_raw.tobytes(),
            context={"motion_history": engine.motion_log.recent_slice(25.0)},
        )
        fb_res = engine.report_false_alert("alert_motion_test")
        assert fb_res["feedback"]["has_motion_log"] is True
        assert (settings.data_dir / "feedback" / "alert_motion_test" / "motion_slice.json").is_file()
    finally:
        engine.stop()
