from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger("fallguard.devices")

DEVICE_COOKIE: str = "fg_device"
DEVICE_TOKEN_PREFIX: str = "fgd_"
PAIR_CODE_TTL_SEC: float = 600.0
PAIR_CODE_ALPHABET: str = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
PAIR_CODE_LEN: int = 8
MAX_FAILED_CLAIMS: int = 10
FAILED_CLAIM_WINDOW_SEC: float = 600.0
MAX_DEVICES: int = 24

ALERT_MODE_ALL = "all"
ALERT_MODE_STANDUP = "standup"
ALERT_MODE_FALLS = "falls"


def normalize_alert_mode(raw: object) -> str:
    val = str(raw or "").strip().lower()
    if val in ("standup", "stand", "stand_only", "standup_only", "getting_up", "left_bed"):
        return ALERT_MODE_STANDUP
    if val in ("falls", "fall", "falls_only", "fall_only"):
        return ALERT_MODE_FALLS
    return ALERT_MODE_ALL


def alert_matches_mode(state: object, severity: object, mode: object) -> bool:
    st = str(getattr(state, "value", state) or "").strip().lower()
    sev = str(getattr(severity, "value", severity) or "").strip().lower()
    m = normalize_alert_mode(mode)
    if m == ALERT_MODE_ALL:
        return True
    if st == "fallen":
        return True
    if m == ALERT_MODE_STANDUP:
        return st == "left_bed" and sev != "critical"
    return False


# --- Дежурства: кому слать тревоги в какие часы ------------------------------------------
DUTY_MODES = ("always", "night", "day", "custom")


def normalize_duty(raw: object) -> dict[str, Any]:
    """{"mode": always|night|day|custom, "from": час, "to": час, "falls_always": bool}. Нет — всегда."""
    d = raw if isinstance(raw, dict) else {}
    mode = str(d.get("mode") or "always").strip().lower()
    if mode not in DUTY_MODES:
        mode = "always"

    def hour(v: object, default: int) -> int:
        try:
            return int(v) % 24  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    return {
        "mode": mode,
        "from": hour(d.get("from"), 22),
        "to": hour(d.get("to"), 7),
        "falls_always": bool(d.get("falls_always", True)),
    }


def _in_hours(hour: int, start: int, end: int) -> bool:
    if start == end:
        return True
    return start <= hour < end if start < end else (hour >= start or hour < end)


def duty_on(duty: object, hour: int, night_start: int = 22, night_end: int = 7) -> bool:
    """Дежурит ли получатель в этот час (местное время сервера). Ночь — из настроек ночного режима."""
    d = normalize_duty(duty)
    if d["mode"] == "always":
        return True
    if d["mode"] == "night":
        return _in_hours(hour, night_start, night_end)
    if d["mode"] == "day":
        return not _in_hours(hour, night_start, night_end)
    return _in_hours(hour, d["from"], d["to"])


def duty_covers(duty: object, state: object, hour: int, night_start: int = 22, night_end: int = 7) -> bool:
    """Получит ли тревогу: дежурит — да; не дежурит — только падение, если «падения — всегда»."""
    st = str(getattr(state, "value", state) or "").strip().lower()
    d = normalize_duty(duty)
    return duty_on(d, hour, night_start, night_end) or (st == "fallen" and d["falls_always"])


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _clean_name(raw: object, fallback: str) -> str:
    name = " ".join(str(raw or "").split())[:48]
    return name or fallback


class DeviceStore:
    """
    Привязанные телефоны (Android-приложение). Токен устройства выдаётся один раз при привязке
    по одноразовому коду и хранится на сервере только в виде SHA-256, поэтому утечка
    data/devices.json не раскрывает действующие токены.
    """

    def __init__(self, path: Path | str, wall_clock: Any = time.time) -> None:
        self._path = Path(path)
        self._clock = wall_clock
        self._lock = threading.RLock()
        self._devices: dict[str, dict[str, Any]] = {}
        self._pair_codes: dict[str, float] = {}
        self._failed_claims: deque[float] = deque(maxlen=MAX_FAILED_CLAIMS * 2)
        self._load()

    # --- хранение -----------------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            items = raw.get("devices", []) if isinstance(raw, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                dev_id = str(item.get("id") or "").strip()
                token_hash = str(item.get("token_hash") or "").strip()
                if not dev_id or len(token_hash) != 64:
                    continue
                self._devices[dev_id] = {
                    "id": dev_id,
                    "name": _clean_name(item.get("name"), "Телефон"),
                    "platform": str(item.get("platform") or "android")[:24],
                    "token_hash": token_hash,
                    "mode": normalize_alert_mode(item.get("mode")),
                    "enabled": bool(item.get("enabled", True)),
                    "duty": normalize_duty(item.get("duty")),
                    "created_at": float(item.get("created_at") or 0.0),
                    "last_seen": float(item.get("last_seen") or 0.0),
                }
        except Exception as exc:
            logger.warning("Не удалось прочитать %s: %s", self._path, exc)

    def _save_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(f"{self._path.name}.tmp.{os.getpid()}")
            tmp.write_text(
                json.dumps({"devices": list(self._devices.values())}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.warning("Не удалось сохранить %s: %s", self._path, exc)

    @staticmethod
    def _public(dev: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": dev["id"],
            "name": dev["name"],
            "platform": dev["platform"],
            "mode": dev["mode"],
            "enabled": dev["enabled"],
            "duty": normalize_duty(dev.get("duty")),
            "created_at": dev["created_at"],
            "last_seen": dev["last_seen"],
        }

    # --- привязка -----------------------------------------------------------------
    def create_pair_code(self) -> tuple[str, float]:
        """Одноразовый код привязки на 10 минут. Возвращает (код, срок действия в секундах)."""
        now = float(self._clock())
        with self._lock:
            self._pair_codes = {c: exp for c, exp in self._pair_codes.items() if exp > now}
            code = "".join(secrets.choice(PAIR_CODE_ALPHABET) for _ in range(PAIR_CODE_LEN))
            self._pair_codes[code] = now + PAIR_CODE_TTL_SEC
            return code, PAIR_CODE_TTL_SEC

    def claims_locked_out(self) -> bool:
        now = float(self._clock())
        with self._lock:
            recent = [t for t in self._failed_claims if (now - t) <= FAILED_CLAIM_WINDOW_SEC]
            return len(recent) >= MAX_FAILED_CLAIMS

    def claim(
        self,
        code: str | None,
        name: str = "",
        platform: str = "android",
        require_code: bool = True,
    ) -> tuple[dict[str, Any], str] | None:
        """
        Обменивает код привязки на постоянный токен устройства. require_code=False допустим
        только для запросов из локальной сети, где пульт и так открыт без авторизации.
        """
        now = float(self._clock())
        clean_code = "".join(ch for ch in str(code or "").upper() if ch.isalnum())
        with self._lock:
            if require_code or clean_code:
                if self.claims_locked_out():
                    return None
                valid = bool(clean_code) and any(
                    hmac.compare_digest(known, clean_code) and exp > now
                    for known, exp in self._pair_codes.items()
                )
                if not valid:
                    self._failed_claims.append(now)
                    return None
                self._pair_codes.pop(clean_code, None)

            if len(self._devices) >= MAX_DEVICES:
                oldest = min(self._devices.values(), key=lambda d: d["last_seen"] or d["created_at"])
                self._devices.pop(oldest["id"], None)

            token = DEVICE_TOKEN_PREFIX + secrets.token_urlsafe(32)
            dev_id = secrets.token_hex(6)
            dev = {
                "id": dev_id,
                "name": _clean_name(name, "Телефон"),
                "platform": str(platform or "android")[:24],
                "token_hash": _hash_token(token),
                "mode": ALERT_MODE_ALL,
                "enabled": True,
                "created_at": round(now, 3),
                "last_seen": round(now, 3),
            }
            self._devices[dev_id] = dev
            self._save_locked()
            return self._public(dev), token

    # --- использование ------------------------------------------------------------
    def authenticate(self, token: str | None) -> dict[str, Any] | None:
        raw = str(token or "").strip()
        if not raw.startswith(DEVICE_TOKEN_PREFIX) or len(raw) > 128:
            return None
        digest = _hash_token(raw)
        now = float(self._clock())
        with self._lock:
            for dev in self._devices.values():
                if hmac.compare_digest(dev["token_hash"], digest):
                    if (now - float(dev["last_seen"])) >= 300.0:
                        dev["last_seen"] = round(now, 3)
                        self._save_locked()
                    return self._public(dev)
        return None

    # Телефон опрашивает сервер постоянно, а last_seen обновляется раз в 5 минут:
    # молчание дольше 15 минут значит, что тревоги до него сейчас не дойдут
    STALE_AFTER_SEC: float = 900.0

    def list(self) -> list[dict[str, Any]]:
        now = float(self._clock())
        with self._lock:
            out = []
            for d in sorted(self._devices.values(), key=lambda d: d["created_at"]):
                pub = self._public(d)
                seen = float(d.get("last_seen") or d.get("created_at") or 0.0)
                pub["offline_sec"] = round(max(0.0, now - seen)) if d["enabled"] else 0
                pub["stale"] = bool(d["enabled"]) and (now - seen) > self.STALE_AFTER_SEC
                out.append(pub)
            return out

    def get(self, dev_id: str) -> dict[str, Any] | None:
        with self._lock:
            dev = self._devices.get(str(dev_id or "").strip())
            return self._public(dev) if dev is not None else None

    def update(
        self,
        dev_id: str,
        name: str | None = None,
        mode: str | None = None,
        enabled: bool | None = None,
        duty: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            dev = self._devices.get(str(dev_id or "").strip())
            if dev is None:
                return None
            if name is not None:
                dev["name"] = _clean_name(name, dev["name"])
            if mode is not None:
                dev["mode"] = normalize_alert_mode(mode)
            if enabled is not None:
                dev["enabled"] = bool(enabled)
            if duty is not None:
                dev["duty"] = normalize_duty(duty)
            self._save_locked()
            return self._public(dev)

    def remove(self, dev_id: str) -> bool:
        with self._lock:
            removed = self._devices.pop(str(dev_id or "").strip(), None)
            if removed is not None:
                self._save_locked()
            return removed is not None


class AlertFeed:
    """
    Лента последних тревог для приложений-клиентов (Android-сервис опрашивает её long-poll'ом).
    seq — миллисекунды Unix-времени, строго возрастающие: переживают перезапуск сервера.
    """

    def __init__(self, max_items: int = 60) -> None:
        self._lock = threading.Lock()
        self._items: deque[dict[str, Any]] = deque(maxlen=max_items)
        self._last_seq = 0

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._last_seq

    def push(
        self,
        alert_id: str,
        state: str,
        severity: str,
        title: str,
        description: str,
        has_image: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            seq = max(int(time.time() * 1000), self._last_seq + 1)
            self._last_seq = seq
            item = {
                "seq": seq,
                "id": str(alert_id),
                "ts": round(time.time(), 3),
                "state": str(state),
                "severity": str(severity),
                "title": str(title),
                "description": str(description),
                "is_fall": str(state) == "fallen",
                "has_image": bool(has_image),
            }
            self._items.append(item)
            return dict(item)

    def last_ts(self, states: tuple[str, ...]) -> float | None:
        """Время последней тревоги с одним из этих состояний — с него и решаем, кто на дежурстве."""
        with self._lock:
            for item in reversed(self._items):
                if str(item.get("state")) in states:
                    return float(item.get("ts") or 0.0) or None
        return None

    def since(self, after_seq: int, mode: str = ALERT_MODE_ALL, max_age_sec: float = 0.0) -> list[dict[str, Any]]:
        eff_mode = normalize_alert_mode(mode)
        now = time.time()
        with self._lock:
            out = []
            for item in self._items:
                if item["seq"] <= after_seq:
                    continue
                if not alert_matches_mode(item["state"], item["severity"], eff_mode):
                    continue
                if max_age_sec > 0.0 and (now - float(item["ts"])) > max_age_sec:
                    continue
                out.append(dict(item))
            return out
