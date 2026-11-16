from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import time
import urllib.parse
from collections.abc import Mapping
from typing import Any

WEBAPP_SESSION_COOKIE: str = "fg_tg_session"
DEFAULT_SESSION_TTL_SEC: int = 2_592_000  # 30 суток
DEFAULT_INIT_DATA_MAX_AGE_SEC: int = 86_400  # 24 часа
CLOCK_SKEW_TOLERANCE_SEC: float = 60.0

_DEFAULT_REVERSE_PROXY_IPS = ("192.168.1.6", "192.168.211.2")


def _load_reverse_proxy_ips() -> frozenset[str]:
    """Адреса обратных прокси, через которые идёт внешний трафик: REVERSE_PROXY_IPS в .env (через запятую)."""
    raw = os.environ.get("REVERSE_PROXY_IPS", "")
    extra = {p.strip() for p in raw.replace(";", ",").split(",") if p.strip()}
    return frozenset(set(_DEFAULT_REVERSE_PROXY_IPS) | extra)


REVERSE_PROXY_IPS: frozenset[str] = _load_reverse_proxy_ips()

_LOCAL_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "testserver",
        "testclient",
    }
)

_LOCAL_DOMAIN_SUFFIXES: tuple[str, ...] = (
    ".local",
    ".lan",
    ".localhost",
    ".internal",
    ".home.arpa",
)


def _derive_webapp_secret_key(bot_token: str) -> bytes:
    return hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()


def _derive_session_secret_key(bot_token: str) -> bytes:
    return hmac.new(
        b"FallGuardWebAppSession",
        bot_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()


def verify_telegram_init_data(
    init_data: str,
    bot_token: str,
    max_age_sec: int = DEFAULT_INIT_DATA_MAX_AGE_SEC,
    now_ts: float | None = None,
) -> dict[str, Any] | None:
    """
    Проверяет криптографическую подпись HMAC-SHA256 строки window.Telegram.WebApp.initData
    по официальному алгоритму Telegram Mini Apps (ключ HMAC-SHA256("WebAppData", bot_token))
    и контролирует свежесть auth_date.
    """
    if not isinstance(init_data, str) or not isinstance(bot_token, str):
        return None
    clean_init = init_data.strip()
    clean_token = bot_token.strip()
    if not clean_init or not clean_token:
        return None

    try:
        pairs = urllib.parse.parse_qsl(
            clean_init,
            keep_blank_values=True,
            strict_parsing=False,
        )
    except ValueError:
        return None

    if not pairs:
        return None

    received_hash = ""
    hash_seen = False
    check_pairs: list[tuple[str, str]] = []
    for key, val in pairs:
        if key == "hash":
            if hash_seen:
                return None
            hash_seen = True
            received_hash = val.strip().lower()
        else:
            check_pairs.append((key, val))

    if not received_hash or len(received_hash) != 64 or not check_pairs:
        return None

    check_pairs.sort(key=lambda item: item[0])
    data_check_string = "\n".join(f"{k}={v}" for k, v in check_pairs)

    secret_key = _derive_webapp_secret_key(clean_token)
    expected_hash = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, received_hash):
        return None

    params = dict(check_pairs)
    raw_auth_date = params.get("auth_date", "").strip()
    try:
        auth_date = int(raw_auth_date)
    except ValueError:
        return None
    if auth_date <= 0:
        return None

    eff_now = float(time.time() if now_ts is None else now_ts)
    age = eff_now - float(auth_date)
    if age < -CLOCK_SKEW_TOLERANCE_SEC or age > float(max_age_sec):
        return None

    raw_user = params.get("user", "").strip()
    if not raw_user:
        return None
    try:
        user_obj = json.loads(raw_user)
    except (ValueError, TypeError):
        return None
    if not isinstance(user_obj, dict):
        return None

    raw_uid = user_obj.get("id")
    if raw_uid is None or isinstance(raw_uid, bool):
        return None
    chat_id = str(raw_uid).strip()
    if not chat_id:
        return None

    first_name = str(user_obj.get("first_name") or "").strip()
    last_name = str(user_obj.get("last_name") or "").strip()
    username = str(user_obj.get("username") or "").strip().lstrip("@")
    person_name = " ".join(p for p in (first_name, last_name) if p).strip()
    display_name = person_name or (f"@{username}" if username else f"Чат {chat_id}")
    start_param = str(params.get("start_param") or "").strip()

    return {
        "chat_id": chat_id,
        "name": display_name,
        "username": username,
        "auth_date": auth_date,
        "start_param": start_param,
    }


def create_session_token(
    chat_id: str,
    name: str,
    bot_token: str,
    ttl_sec: int = DEFAULT_SESSION_TTL_SEC,
    now_ts: float | None = None,
    username: str = "",
) -> str:
    """
    Выпускает компактный подписанный сессионный токен Mini App:
    urlsafe_b64(payload_json) + "." + hmac_sha256_hex.
    """
    clean_cid = str(chat_id or "").strip()
    clean_token = str(bot_token or "").strip()
    if not clean_cid:
        raise ValueError("chat_id не может быть пустым")
    if not clean_token:
        raise ValueError("bot_token не может быть пустым")

    iat = int(time.time() if now_ts is None else now_ts)
    exp = iat + max(1, int(ttl_sec))
    payload_dict: dict[str, Any] = {
        "chat_id": clean_cid,
        "name": str(name or "").strip() or f"Чат {clean_cid}",
        "iat": iat,
        "exp": exp,
    }
    clean_uname = str(username or "").strip().lstrip("@")
    if clean_uname:
        payload_dict["username"] = clean_uname

    payload_bytes = json.dumps(
        payload_dict,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    b64_part = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    sig_hex = hmac.new(
        _derive_session_secret_key(clean_token),
        b64_part.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{b64_part}.{sig_hex}"


def verify_session_token(
    token: str,
    bot_token: str,
    now_ts: float | None = None,
) -> dict[str, Any] | None:
    """Проверяет подпись и срок действия (exp) сессионного токена Mini App."""
    if not isinstance(token, str) or not isinstance(bot_token, str):
        return None
    clean_tok = token.strip()
    clean_secret = bot_token.strip()
    if not clean_tok or not clean_secret or "." not in clean_tok:
        return None

    parts = clean_tok.split(".")
    if len(parts) != 2:
        return None
    b64_part, sig_part = parts[0].strip(), parts[1].strip().lower()
    if not b64_part or len(sig_part) != 64:
        return None
    try:
        b64_bytes = b64_part.encode("ascii")
    except UnicodeEncodeError:
        return None

    expected_sig = hmac.new(
        _derive_session_secret_key(clean_secret),
        b64_bytes,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, sig_part):
        return None

    padded_b64 = b64_part + ("=" * (-len(b64_part) % 4))
    try:
        raw_json = base64.urlsafe_b64decode(padded_b64.encode("ascii"))
        data = json.loads(raw_json.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    raw_exp = data.get("exp")
    if isinstance(raw_exp, bool) or not isinstance(raw_exp, (int, float)):
        return None
    exp = int(raw_exp)
    eff_now = float(time.time() if now_ts is None else now_ts)
    if eff_now >= float(exp):
        return None

    raw_cid = data.get("chat_id") or data.get("cid")
    if raw_cid is None or isinstance(raw_cid, bool):
        return None
    chat_id = str(raw_cid).strip()
    if not chat_id:
        return None

    raw_iat = data.get("iat", 0)
    iat = int(raw_iat) if isinstance(raw_iat, (int, float)) and not isinstance(raw_iat, bool) else 0
    return {
        "chat_id": chat_id,
        "name": str(data.get("name") or f"Чат {chat_id}"),
        "username": str(data.get("username") or data.get("uname") or "").strip().lstrip("@"),
        "iat": iat,
        "exp": exp,
    }


def _parse_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    clean = value.strip()
    if not clean:
        return None
    if clean.startswith("[") and clean.endswith("]"):
        clean = clean[1:-1].strip()
    if "%" in clean:
        clean = clean.split("%", 1)[0].strip()
    try:
        ip_obj = ipaddress.ip_address(clean)
    except ValueError:
        return None
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        return ip_obj.ipv4_mapped
    return ip_obj


def _is_private_or_local_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip_obj.is_loopback
        or ip_obj.is_private
        or ip_obj.is_link_local
        or ip_obj.is_unspecified
    )


def _extract_hostname_without_port(raw_header: str) -> str:
    if not raw_header:
        return ""
    first = raw_header.split(",", 1)[0].strip().lower()
    if not first:
        return ""
    if first.startswith("["):
        end_idx = first.find("]")
        if end_idx > 1:
            return first[1:end_idx].strip().rstrip(".")
        return first.strip("[]").rstrip(".")
    if first.count(":") == 1:
        host_only, _ = first.split(":", 1)
        return host_only.strip().rstrip(".")
    return first.rstrip(".")


def _is_public_domain_or_ip(hostname: str) -> bool:
    if not hostname:
        return False
    if hostname in _LOCAL_HOSTNAMES or hostname.endswith(_LOCAL_DOMAIN_SUFFIXES):
        return False
    ip_obj = _parse_ip(hostname)
    if ip_obj is not None:
        return not _is_private_or_local_ip(ip_obj)
    return "." in hostname


def is_external_proxy_request(
    headers: Mapping[str, str] | None,
    client_host: str | None,
) -> bool:
    """
    Определяет, пришёл ли запрос снаружи через обратный прокси / публичный домен:
    - True для публичных доменов (например nofall.example.com), прокси 192.168.1.6 / 192.168.211.2
      и любых внешних публичных IP;
    - False для прямого доступа из LAN/VPN (192.168.1.50:8090, localhost, TestClient).
    """
    low_headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    for hdr_name in ("x-forwarded-host", "host"):
        hostname = _extract_hostname_without_port(low_headers.get(hdr_name, ""))
        if _is_public_domain_or_ip(hostname):
            return True

    # Запрос прошёл через прокси: смотрим, откуда он пришёл на самом деле. Берём ПОСЛЕДНИЙ адрес
    # X-Forwarded-For — его дописал ближайший прокси, подделать его клиент не может. Если прокси
    # не передал адрес клиента вовсе — считаем запрос внешним (безопаснее потребовать вход).
    forwarded_markers = ("x-forwarded-for", "x-real-ip", "forwarded", "x-forwarded-proto", "x-forwarded-host")
    if any(low_headers.get(h, "").strip() for h in forwarded_markers):
        hops = [h.strip() for h in low_headers.get("x-forwarded-for", "").split(",") if h.strip()]
        origin = hops[-1] if hops else low_headers.get("x-real-ip", "").strip()
        origin_ip = _parse_ip(origin) if origin else None
        if origin_ip is None or not _is_private_or_local_ip(origin_ip):
            return True

    clean_client = (client_host or "").strip().lower()
    if not clean_client or clean_client in _LOCAL_HOSTNAMES:
        return False

    ip_obj = _parse_ip(clean_client)
    if ip_obj is not None:
        if str(ip_obj) in REVERSE_PROXY_IPS:
            return True
        return not _is_private_or_local_ip(ip_obj)

    if clean_client in REVERSE_PROXY_IPS:
        return True
    return _is_public_domain_or_ip(clean_client)
