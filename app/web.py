from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import FallGuardEngine
from .patient_id import Role
from .tg_auth import (
    DEFAULT_SESSION_TTL_SEC,
    WEBAPP_SESSION_COOKIE,
    create_session_token,
    is_external_proxy_request,
    verify_session_token,
    verify_telegram_init_data,
)


class ZonesUpdatePayload(BaseModel):
    bed_polygon: list[list[float]] | None = None
    door_polygon: list[list[float]] | None = None
    ignore_polygons: list[list[list[float]]] | None = None
    floor_polygon: list[list[float]] | None = None
    floor_width_cm: float | None = None
    floor_depth_cm: float | None = None
    mattress_height_cm: float | None = None


class TimelineSanitizePayload(BaseModel):
    min_caregiver_sec: float = 3.5
    cutoff_ts: float | None = None


class PausePayload(BaseModel):
    seconds: float = 900.0


class AssignTrackPayload(BaseModel):
    track_id: int


class CameraTransformPayload(BaseModel):
    rotate: int | None = None
    flip_h: bool | None = None
    enhance: bool | None = None


class EnrollFromCameraPayload(BaseModel):
    track_id: int | None = None


class SubscriberTogglePayload(BaseModel):
    enabled: bool | None = None


class TrackRolePayload(BaseModel):
    role: str


class FalseAlarmReportPayload(BaseModel):
    reporter: str = "Веб-пульт"


class TelegramAuthPayload(BaseModel):
    init_data: str = ""


_PUBLIC_SHELL_GET_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/manifest.webmanifest",
        "/icon.svg",
        "/sw.js",
        "/telegram-web-app.js",
        "/api/health",
        "/api/tg/session",
    }
)

_PUBLIC_AUTH_POST_PATHS: frozenset[str] = frozenset(
    {
        "/api/tg/auth",
        "/api/tg/logout",
    }
)


def _resolve_safe_face_photo(faces_root: Path, role: str, filename: str) -> Path | None:
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        return None
    try:
        role_dir = (faces_root / role).resolve()
        candidate = (role_dir / filename).resolve()
    except OSError:
        return None
    if candidate.parent != role_dir or candidate.is_symlink() or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        return None
    return candidate


def create_web_app(engine: FallGuardEngine) -> FastAPI:
    app = FastAPI(title="FallGuard Control Panel", version="1.7.0")
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _is_https_request(headers: Mapping[str, str], scheme: str) -> bool:
        fwd_proto = (headers.get("x-forwarded-proto") or "").split(",", 1)[0].strip().lower()
        return fwd_proto == "https" or scheme.lower() == "https"

    def _authenticate_external_credentials(
        headers: Mapping[str, str],
        cookies: Mapping[str, str],
        query_params: Mapping[str, str],
    ) -> dict[str, Any] | None:
        if not engine.settings.telegram_configured:
            return None
        bot_token = engine.settings.tg_bot_token.strip()
        if not bot_token:
            return None

        token_candidates: list[str] = []
        raw_auth = (headers.get("authorization") or "").strip()
        if raw_auth.lower().startswith("bearer "):
            bearer_tok = raw_auth[7:].strip()
            if bearer_tok:
                token_candidates.append(bearer_tok)

        query_tok = (query_params.get("token") or "").strip()
        if query_tok:
            token_candidates.append(query_tok)

        cookie_tok = (cookies.get(WEBAPP_SESSION_COOKIE) or "").strip()
        if cookie_tok:
            token_candidates.append(cookie_tok)

        for cand in token_candidates:
            parsed = verify_session_token(cand, bot_token)
            if parsed is not None and engine.notifier.is_authorized_webapp_user(parsed["chat_id"]):
                return parsed

        raw_init_data = (headers.get("x-telegram-init-data") or "").strip()
        if raw_init_data:
            verified = verify_telegram_init_data(raw_init_data, bot_token)
            if verified is not None and engine.notifier.is_authorized_webapp_user(
                verified["chat_id"]
            ):
                return {
                    "chat_id": verified["chat_id"],
                    "name": verified["name"],
                    "username": verified.get("username", ""),
                }
        return None

    def _is_public_shell_route(method: str, path: str) -> bool:
        m = method.upper()
        if m in ("GET", "HEAD") and path in _PUBLIC_SHELL_GET_PATHS:
            return True
        if m == "POST" and path in _PUBLIC_AUTH_POST_PATHS:
            return True
        return False

    def _authorize_websocket(websocket: WebSocket) -> bool:
        client_host = websocket.client.host if websocket.client is not None else None
        if not is_external_proxy_request(websocket.headers, client_host):
            return True
        return (
            _authenticate_external_credentials(
                websocket.headers,
                websocket.cookies,
                websocket.query_params,
            )
            is not None
        )

    @app.middleware("http")
    async def external_telegram_auth_guard(
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        client_host = request.client.host if request.client is not None else None
        if is_external_proxy_request(request.headers, client_host):
            if not _is_public_shell_route(request.method, request.url.path):
                auth_user = _authenticate_external_credentials(
                    request.headers,
                    request.cookies,
                    request.query_params,
                )
                if auth_user is None:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "ok": False,
                            "detail": "Требуется вход через Telegram Mini App",
                        },
                    )
        return await call_next(request)

    @app.api_route("/", methods=["GET", "HEAD"])
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.api_route("/manifest.webmanifest", methods=["GET", "HEAD"])
    async def pwa_manifest() -> FileResponse:
        path = static_dir / "manifest.webmanifest"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Манифест не найден")
        return FileResponse(path, media_type="application/manifest+json")

    @app.api_route("/sw.js", methods=["GET", "HEAD"])
    async def pwa_service_worker() -> FileResponse:
        path = static_dir / "sw.js"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Service Worker не найден")
        return FileResponse(
            path,
            media_type="application/javascript",
            headers={
                "Service-Worker-Allowed": "/",
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            },
        )

    @app.api_route("/icon.svg", methods=["GET", "HEAD"])
    async def pwa_icon() -> FileResponse:
        path = static_dir / "icon.svg"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Иконка не найдена")
        return FileResponse(path, media_type="image/svg+xml")

    @app.api_route("/telegram-web-app.js", methods=["GET", "HEAD"])
    async def telegram_webapp_sdk() -> FileResponse:
        path = static_dir / "telegram-web-app.js"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Telegram WebApp SDK не найден")
        return FileResponse(
            path,
            media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.api_route("/api/health", methods=["GET", "HEAD"])
    async def health() -> dict:
        return {"ok": True, "state": engine.fsm.state.value}

    @app.api_route("/api/tg/session", methods=["GET", "HEAD"])
    async def tg_session(request: Request) -> dict[str, Any]:
        client_host = request.client.host if request.client is not None else None
        is_external = is_external_proxy_request(request.headers, client_host)
        bot_uname = engine.notifier.bot_username
        webapp_url = engine.settings.public_webapp_url
        if not is_external:
            return {
                "ok": True,
                "authenticated": True,
                "mode": "lan",
                "session_token": "",
                "user": None,
                "bot_username": bot_uname,
                "webapp_url": webapp_url,
            }

        auth_user = _authenticate_external_credentials(
            request.headers,
            request.cookies,
            request.query_params,
        )
        if auth_user is None:
            return {
                "ok": True,
                "authenticated": False,
                "mode": "none",
                "session_token": "",
                "user": None,
                "bot_username": bot_uname,
                "webapp_url": webapp_url,
            }

        fresh_token = create_session_token(
            chat_id=auth_user["chat_id"],
            name=auth_user["name"],
            bot_token=engine.settings.tg_bot_token,
            ttl_sec=DEFAULT_SESSION_TTL_SEC,
            username=auth_user.get("username", ""),
        )
        return {
            "ok": True,
            "authenticated": True,
            "mode": "telegram",
            "session_token": fresh_token,
            "user": {
                "chat_id": auth_user["chat_id"],
                "name": auth_user["name"],
                "username": auth_user.get("username", ""),
            },
            "bot_username": bot_uname,
            "webapp_url": webapp_url,
        }

    @app.post("/api/tg/auth")
    async def tg_auth(
        request: Request,
        payload: TelegramAuthPayload | None = Body(default=None),
    ) -> JSONResponse:
        if not engine.settings.telegram_configured:
            return JSONResponse(
                status_code=401,
                content={
                    "ok": False,
                    "detail": "Токен Telegram-бота не настроен на сервере",
                },
            )
        raw_init = (payload.init_data if payload is not None else "").strip()
        verified = verify_telegram_init_data(raw_init, engine.settings.tg_bot_token)
        if verified is None:
            return JSONResponse(
                status_code=401,
                content={
                    "ok": False,
                    "detail": "Недействительная или устаревшая подпись Telegram Mini App",
                },
            )

        chat_id = verified["chat_id"]
        bot_uname = engine.notifier.bot_username or "NoFallenAi"
        if not engine.notifier.is_authorized_webapp_user(chat_id):
            return JSONResponse(
                status_code=403,
                content={
                    "ok": False,
                    "detail": f"Аккаунт не подключён к боту @{bot_uname}. Отправьте /start боту.",
                },
            )

        session_token = create_session_token(
            chat_id=chat_id,
            name=verified["name"],
            bot_token=engine.settings.tg_bot_token,
            ttl_sec=DEFAULT_SESSION_TTL_SEC,
            username=verified.get("username", ""),
        )
        user_payload = {
            "chat_id": chat_id,
            "name": verified["name"],
            "username": verified.get("username", ""),
        }
        resp = JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "authenticated": True,
                "mode": "telegram",
                "session_token": session_token,
                "user": user_payload,
                "start_param": verified.get("start_param", ""),
            },
        )
        is_https = _is_https_request(request.headers, request.url.scheme)
        resp.set_cookie(
            key=WEBAPP_SESSION_COOKIE,
            value=session_token,
            max_age=DEFAULT_SESSION_TTL_SEC,
            path="/",
            httponly=True,
            secure=is_https,
            samesite="none" if is_https else "lax",
        )
        return resp

    @app.post("/api/tg/logout")
    async def tg_logout(request: Request) -> JSONResponse:
        resp = JSONResponse(
            status_code=200,
            content={"ok": True, "authenticated": False},
        )
        is_https = _is_https_request(request.headers, request.url.scheme)
        resp.delete_cookie(
            key=WEBAPP_SESSION_COOKIE,
            path="/",
            httponly=True,
            secure=is_https,
            samesite="none" if is_https else "lax",
        )
        return resp

    @app.get("/api/status")
    async def status(window_sec: float = Query(default=21600.0, ge=60.0, le=86400.0)) -> dict:
        return engine.get_status(timeline_window_sec=window_sec)

    @app.get("/api/prefs")
    async def get_prefs() -> dict:
        return {
            "ok": True,
            "prefs": engine.get_prefs(),
            "care": {
                "diagnostics": engine.get_care_diagnostics(),
                "summary_24h": engine.timeline.summary_stats(window_sec=86400.0),
            },
        }

    @app.post("/api/prefs")
    async def update_prefs(payload: dict[str, Any] = Body(...)) -> dict:
        updated = engine.update_prefs(payload)
        return {
            "ok": True,
            "prefs": updated,
            "care": {
                "diagnostics": engine.get_care_diagnostics(),
                "summary_24h": engine.timeline.summary_stats(window_sec=86400.0),
            },
        }

    @app.post("/api/recalibrate-baseline")
    async def recalibrate_baseline() -> dict:
        diag = engine.recalibrate_baseline()
        return {"ok": True, "diagnostics": diag}

    @app.post("/api/reset-anchor")
    async def reset_anchor() -> dict:
        res = engine.reset_appearance_anchor()
        return {"ok": True, **res}

    @app.get("/api/timeline")
    async def get_timeline(window_sec: float = Query(default=21600.0, ge=60.0, le=86400.0)) -> dict:
        tl = engine.timeline.to_dict(view_window_sec=window_sec)
        clips = [
            c
            for c in engine.archive.list_clips(limit=60)
            if float(c.get("ts") or 0.0) >= float(tl["start_ts"])
        ]
        thumbs = (
            engine.thumbs.list_index(since_ts=float(tl["start_ts"]))
            if getattr(engine, "thumbs", None) is not None
            else []
        )
        return {
            "ok": True,
            "timeline": tl,
            "clips": clips,
            "thumbs": thumbs,
            "summary_24h": engine.timeline.summary_stats(window_sec=86400.0),
        }

    @app.post("/api/timeline/sanitize")
    async def sanitize_timeline(
        payload: TimelineSanitizePayload | None = Body(default=None),
    ) -> dict:
        eff_min_sec = payload.min_caregiver_sec if payload is not None else 3.5
        eff_cutoff = payload.cutoff_ts if payload is not None else None
        clip_ts = [
            float(c["ts"])
            for c in engine.archive.list_clips(limit=200)
            if c.get("ts") is not None
        ]
        removed = engine.timeline.sanitize_glitches(
            min_caregiver_sec=eff_min_sec,
            replace_caregiver_after_ts=eff_cutoff,
            keep_clip_timestamps=clip_ts,
        )
        return {
            "ok": True,
            "removed": removed,
            "timeline": engine.timeline.to_dict(),
            "summary_24h": engine.timeline.summary_stats(window_sec=86400.0),
        }

    @app.get("/api/timeline/thumb")
    async def get_timeline_thumb_nearest(ts: float = Query(...)) -> FileResponse:
        if getattr(engine, "thumbs", None) is None:
            raise HTTPException(status_code=404, detail="Хранилище мини-снимков недоступно")
        found = engine.thumbs.get_nearest(ts)
        if found is None:
            raise HTTPException(
                status_code=404,
                detail="Мини-снимок хроники за указанное время не найден",
            )
        thumb_path, entry = found
        resolved = Path(thumb_path)
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="Файл мини-снимка не найден")
        entry_dict = entry if isinstance(entry, dict) else {}
        return FileResponse(
            resolved,
            media_type="image/jpeg",
            headers={
                "X-Thumb-Ts": str(entry_dict.get("ts", ts)),
                "X-Thumb-State": str(entry_dict.get("state", "")),
                "Cache-Control": "no-cache",
            },
        )

    @app.get("/api/timeline/thumb/{thumb_id}.jpg")
    async def get_timeline_thumb_by_id(thumb_id: str) -> FileResponse:
        if (
            not thumb_id
            or "/" in thumb_id
            or "\\" in thumb_id
            or ".." in thumb_id
            or getattr(engine, "thumbs", None) is None
        ):
            raise HTTPException(status_code=404, detail="Мини-снимок не найден")
        thumb_path = engine.thumbs.get_by_id(thumb_id)
        if thumb_path is None:
            raise HTTPException(status_code=404, detail="Мини-снимок не найден")
        resolved = Path(thumb_path)
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="Файл мини-снимка не найден")
        return FileResponse(
            resolved,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/report")
    async def get_report() -> dict:
        return {
            "ok": True,
            "text": engine.get_care_report_text(),
            "summary_24h": engine.timeline.summary_stats(window_sec=86400.0),
            "diagnostics": engine.get_care_diagnostics(),
        }

    @app.get("/api/telegram-qr.svg")
    async def get_telegram_qr() -> Response:
        svg = engine.get_telegram_qr_svg()
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/clips")
    async def list_clips(
        limit: int = Query(default=50, ge=1, le=200),
        state: str | None = Query(default=None),
    ) -> dict:
        return {
            "ok": True,
            "clips": engine.archive.list_clips(limit=limit, state_filter=state),
            "stats": engine.archive.storage_stats(),
        }

    @app.get("/api/clips/{clip_id}.jpg")
    async def get_clip_jpg(clip_id: str) -> FileResponse:
        media_path = engine.archive.resolve_media_path(clip_id, "jpg")
        if media_path is None:
            raise HTTPException(status_code=404, detail="Снимок события не найден")
        return FileResponse(media_path, media_type="image/jpeg")

    @app.get("/api/clips/{clip_id}.mp4")
    async def get_clip_mp4(clip_id: str, download: int = Query(default=0)) -> FileResponse:
        media_path = engine.archive.resolve_media_path(clip_id, "mp4")
        if media_path is None:
            raise HTTPException(status_code=404, detail="Видеоклип события не найден")
        disposition = "attachment" if download == 1 else "inline"
        return FileResponse(
            media_path,
            media_type="video/mp4",
            filename=f"{clip_id}.mp4",
            content_disposition_type=disposition,
        )

    @app.get("/api/clips/{clip_id}")
    async def get_clip_meta(clip_id: str) -> dict:
        meta = engine.archive.get_clip(clip_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="Событие в архиве не найдено")
        return {"ok": True, "clip": meta}

    @app.delete("/api/clips/{clip_id}")
    async def delete_clip(clip_id: str) -> dict:
        ok = engine.archive.delete_clip(clip_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Клип не найден")
        engine._log_event("info", f"Удалена запись архива {clip_id}", category="system")
        return {
            "ok": True,
            "clips": engine.archive.list_clips(limit=50),
            "stats": engine.archive.storage_stats(),
        }

    @app.delete("/api/clips")
    async def clear_clips() -> dict:
        removed = engine.archive.clear_all()
        engine._log_event("info", f"Очищен видеоархив ({removed} записей)", category="system")
        return {
            "ok": True,
            "removed": removed,
            "clips": [],
            "stats": engine.archive.storage_stats(),
        }

    @app.get("/api/feedback")
    async def list_feedback(
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict:
        return {
            "ok": True,
            "items": engine.feedback.list_feedback(limit=limit),
            "stats": engine.feedback.stats(),
        }

    @app.post("/api/feedback/{alert_id}/false_alarm")
    async def report_false_alarm(
        alert_id: str,
        payload: FalseAlarmReportPayload | None = Body(default=None),
    ) -> dict:
        reporter = (payload.reporter if payload is not None else "Веб-пульт").strip() or "Веб-пульт"
        try:
            return engine.report_false_alert(alert_id, reporter_name=reporter)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/camera")
    async def get_camera() -> dict:
        cam = engine.get_camera_transform()
        return {"ok": True, **cam, "camera": cam}

    @app.post("/api/camera")
    async def update_camera(payload: CameraTransformPayload) -> dict:
        if payload.rotate is not None and payload.rotate not in (0, 90, 180, 270):
            raise HTTPException(
                status_code=422,
                detail="Угол поворота должен принимать одно из значений: 0, 90, 180 или 270",
            )
        cam = engine.set_camera_transform(
            rotate=payload.rotate,
            flip_h=payload.flip_h,
            enhance=payload.enhance,
        )
        return {"ok": True, **cam, "camera": cam}

    @app.get("/api/zones")
    async def get_zones() -> dict:
        return engine.zones.get().to_dict()

    @app.post("/api/zones")
    async def save_zones(payload: ZonesUpdatePayload) -> dict:
        if payload.floor_polygon is not None and len(payload.floor_polygon) not in (0, 4):
            raise HTTPException(
                status_code=422,
                detail="Полигон пола для 2.5D-калибровки должен содержать ровно 4 угловые точки (или пустой список)",
            )
        updated = engine.zones.update(
            bed_polygon=payload.bed_polygon,
            door_polygon=payload.door_polygon,
            ignore_polygons=payload.ignore_polygons,
            floor_polygon=payload.floor_polygon,
            floor_width_cm=payload.floor_width_cm,
            floor_depth_cm=payload.floor_depth_cm,
            mattress_height_cm=payload.mattress_height_cm,
        )
        engine.on_zones_updated()
        snap = engine.get_telemetry_snapshot()
        return {
            "ok": True,
            "zones": updated.to_dict(),
            "homography_calibrated": bool(snap.get("homography_calibrated", False)),
        }

    @app.delete("/api/zones/ignore")
    async def clear_ignore_zones() -> dict:
        updated = engine.clear_ignore_zones()
        return {"ok": True, "zones": updated.to_dict()}

    @app.post("/api/ignore-track")
    async def ignore_track(payload: AssignTrackPayload) -> dict:
        ok, updated = engine.ignore_track(payload.track_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Трек не найден в текущем кадре")
        return {"ok": True, "zones": updated.to_dict()}

    @app.post("/api/pause")
    async def pause_guard(payload: PausePayload) -> dict:
        if payload.seconds <= 0:
            engine.resume()
        else:
            engine.pause(payload.seconds)
        return {"ok": True, "state": engine.fsm.state.value}

    @app.post("/api/resume")
    async def resume_guard() -> dict:
        engine.resume()
        return {"ok": True, "state": engine.fsm.state.value}

    @app.post("/api/door_visit/clear")
    async def clear_door_visit() -> dict:
        return engine.clear_door_visit()

    @app.post("/api/bed_presence/clear")
    async def clear_bed_presence() -> dict:
        return engine.clear_bed_presence()

    @app.post("/api/assign-patient")
    async def assign_patient(payload: AssignTrackPayload) -> dict:
        res = engine.assign_track_role(payload.track_id, Role.PATIENT)
        if not res.get("ok"):
            raise HTTPException(
                status_code=404,
                detail=str(res.get("detail") or "Трек не найден в текущем кадре"),
            )
        return res

    @app.post("/api/assign-visitor")
    async def assign_visitor(payload: AssignTrackPayload) -> dict:
        res = engine.assign_track_role(payload.track_id, Role.VISITOR)
        if not res.get("ok"):
            raise HTTPException(
                status_code=404,
                detail=str(res.get("detail") or "Трек не найден в текущем кадре"),
            )
        return res

    @app.post("/api/tracks/{track_id}/role")
    async def assign_track_role_by_path(track_id: int, payload: TrackRolePayload) -> dict:
        role_raw = (payload.role or "").strip().lower()
        if role_raw not in (Role.PATIENT.value, Role.VISITOR.value):
            raise HTTPException(status_code=400, detail="Роль должна быть patient или visitor")
        res = engine.assign_track_role(track_id, Role(role_raw))
        if not res.get("ok"):
            raise HTTPException(
                status_code=404,
                detail=str(res.get("detail") or "Трек не найден в текущем кадре"),
            )
        return res

    @app.post("/api/test-telegram")
    async def test_telegram() -> dict:
        snap = engine.get_latest_jpeg()
        ok, msg = engine.notifier.send_test_snapshot(snap)
        return {"ok": ok, "message": msg}

    @app.get("/api/subscribers")
    async def get_subscribers() -> dict:
        uname = engine.notifier.bot_username
        subs = engine.list_subscribers_normalized()
        return {
            "ok": True,
            "bot_username": uname,
            "bot_url": f"https://t.me/{uname}",
            "subscribers": subs,
            "telegram_subscribers": subs,
        }

    @app.post("/api/telegram/subscribers/{chat_id}/toggle")
    @app.post("/api/subscribers/{chat_id}/toggle")
    async def toggle_subscriber(
        chat_id: str,
        payload: SubscriberTogglePayload | None = Body(default=None),
    ) -> dict:
        explicit_enabled = payload.enabled if payload is not None else None
        updated_sub = engine.toggle_subscriber_enabled(chat_id, enabled=explicit_enabled)
        if updated_sub is None:
            raise HTTPException(status_code=404, detail="Подписчик не найден")
        subs = engine.list_subscribers_normalized()
        return {
            "ok": True,
            "subscriber": updated_sub,
            "enabled": bool(updated_sub.get("enabled", True)),
            "telegram_subscribers": subs,
            "subscribers": subs,
            "telegram_recipients_count": len(engine.notifier.get_recipient_chat_ids()),
        }

    @app.delete("/api/subscribers/{chat_id}")
    async def delete_subscriber(chat_id: str) -> dict:
        ok, msg = engine.notifier.remove_subscriber(chat_id)
        if not ok and "в .env" in msg:
            raise HTTPException(status_code=400, detail=msg)
        if not ok:
            raise HTTPException(status_code=404, detail=msg)
        subs = engine.list_subscribers_normalized()
        return {
            "ok": True,
            "message": msg,
            "subscribers": subs,
            "telegram_subscribers": subs,
        }

    @app.get("/api/faces/{role}/photo/{filename}")
    async def get_face_photo(role: str, filename: str) -> FileResponse:
        if role not in (Role.PATIENT.value, Role.VISITOR.value):
            raise HTTPException(status_code=400, detail="Роль должна быть patient или visitor")
        faces_root = engine.settings.data_dir / "faces"
        photo_path = _resolve_safe_face_photo(faces_root, role, filename)
        if photo_path is None:
            raise HTTPException(status_code=404, detail="Миниатюра эталона не найдена")
        return FileResponse(photo_path)

    @app.post("/api/faces/{role}/from-camera")
    async def enroll_face_from_camera(
        role: str,
        payload: EnrollFromCameraPayload | None = Body(default=None),
    ) -> dict:
        if role not in (Role.PATIENT.value, Role.VISITOR.value):
            raise HTTPException(status_code=400, detail="Роль должна быть patient или visitor")
        track_id = payload.track_id if payload is not None else None
        ok, info = engine.enroll_face_from_camera(Role(role), track_id=track_id)
        if not ok:
            raise HTTPException(status_code=422, detail=info)
        return {"ok": True, "filename": info, "faces": engine.faces.list_photos()}

    @app.post("/api/faces/{role}")
    async def upload_face(role: str, file: UploadFile = File(...)) -> dict:
        if role not in (Role.PATIENT.value, Role.VISITOR.value):
            raise HTTPException(status_code=400, detail="Роль должна быть patient или visitor")
        raw = await file.read()
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Файл больше 15 МБ")
        ok, info = engine.faces.add_photo(Role(role), file.filename or "photo.jpg", raw)
        if not ok:
            raise HTTPException(status_code=422, detail=info)
        role_ru = "бабушка" if role == Role.PATIENT.value else "свой"
        engine._log_event("info", f"Загружено эталонное фото ({role_ru})", category="system")
        return {"ok": True, "filename": info, "faces": engine.faces.list_photos()}

    @app.delete("/api/faces/{role}/{filename}")
    async def delete_face(role: str, filename: str) -> dict:
        if role not in (Role.PATIENT.value, Role.VISITOR.value):
            raise HTTPException(status_code=400, detail="Роль должна быть patient или visitor")
        engine.faces.delete_photo(Role(role), filename)
        engine._log_event("info", f"Удалён эталон лица ({filename})", category="system")
        return {"ok": True, "faces": engine.faces.list_photos()}

    @app.get("/api/snapshot.jpg")
    async def snapshot_jpg(overlay: int = Query(default=1)) -> Response:
        jpeg = engine.get_latest_clean_jpeg() if overlay == 0 else engine.get_latest_jpeg()
        if not jpeg:
            raise HTTPException(status_code=503, detail="Кадр ещё не готов")
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    def _build_mjpeg_response(use_overlay: bool, max_frames: int = 0) -> StreamingResponse:
        async def _gen() -> AsyncGenerator[bytes, None]:
            sent = 0
            while True:
                jpeg = engine.get_latest_jpeg() if use_overlay else engine.get_latest_clean_jpeg()
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                    sent += 1
                    if max_frames > 0 and sent >= max_frames:
                        break
                await asyncio.sleep(0.12)

        return StreamingResponse(
            _gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/api/mjpeg")
    async def stream_mjpeg_clean(
        overlay: int = Query(default=0),
        frames: int = Query(default=0, ge=0, le=1000),
    ) -> StreamingResponse:
        return _build_mjpeg_response(use_overlay=(overlay != 0), max_frames=frames)

    @app.get("/api/stream.mjpg")
    async def stream_mjpg(
        overlay: int = Query(default=1),
        frames: int = Query(default=0, ge=0, le=1000),
    ) -> StreamingResponse:
        return _build_mjpeg_response(use_overlay=(overlay != 0), max_frames=frames)

    @app.get("/api/stream.mp4")
    async def stream_fmp4(
        max_fragments: int = Query(default=0, ge=0, le=10000),
    ) -> StreamingResponse:
        wait_t = (
            3.5
            if (engine.settings.camera_configured or engine.stream_hub.is_ready)
            else 0.15
        )
        ready = await engine.stream_hub.wait_ready(timeout=wait_t)
        init_seg = engine.stream_hub.init_segment
        if not ready or not init_seg:
            raise HTTPException(
                status_code=503,
                detail=engine.stream_hub.last_error or "Прямой H.264 поток ещё не готов",
            )

        async def _fmp4_gen() -> AsyncGenerator[bytes, None]:
            sent_frags = 0
            async with engine.stream_hub.subscribe(include_gop=True) as sub:
                yield init_seg
                async for frag_bytes in sub:
                    yield frag_bytes
                    sent_frags += 1
                    if max_fragments > 0 and sent_frags >= max_fragments:
                        break

        return StreamingResponse(
            _fmp4_gen(),
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "X-Stream-Codecs": engine.stream_hub.codec_string,
            },
        )

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(websocket: WebSocket) -> None:
        if not _authorize_websocket(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            async with engine.subscribe_telemetry() as sub:
                await websocket.send_json(engine.get_telemetry_snapshot())
                while True:
                    try:
                        msg = await asyncio.wait_for(sub.get(), timeout=2.0)
                    except asyncio.TimeoutError:
                        await websocket.send_json(engine.get_telemetry_snapshot())
                        continue
                    if msg is None:
                        break
                    await websocket.send_json(msg)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        if not _authorize_websocket(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            wait_t = (
                3.5
                if (engine.settings.camera_configured or engine.stream_hub.is_ready)
                else 0.15
            )
            ready = await engine.stream_hub.wait_ready(timeout=wait_t)
            init_seg = engine.stream_hub.init_segment
            if not ready or not init_seg:
                await websocket.send_json(
                    {
                        "type": "error",
                        "fallback": "/api/mjpeg?overlay=0",
                        "detail": engine.stream_hub.last_error or "Прямой H.264 поток недоступен",
                    }
                )
                await websocket.close(code=1011)
                return

            await websocket.send_json(engine.stream_hub.get_init_message())
            await websocket.send_bytes(init_seg)
            async with engine.stream_hub.subscribe(include_gop=True) as sub:
                async for frag_bytes in sub:
                    await websocket.send_bytes(frag_bytes)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass

    return app
