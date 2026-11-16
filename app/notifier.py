from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx

from .state_machine import AlertEvent, AlertSeverity

logger = logging.getLogger("fallguard.notifier")
# Подавляем INFO-логи httpx/httpcore, чтобы URL с токеном бота никогда не попадал в логи
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

BTN_MUTE_TEXT = "🔕 Отключить уведомления"
BTN_UNMUTE_TEXT = "🔔 Включить уведомления"
BTN_FALLS_ONLY_TEXT = "🚨 Только падения"
BTN_ALL_EVENTS_TEXT = "📣 Все события"
BTN_PHOTO_TEXT = "📸 Снимок"

ALERT_MODE_ALL = "all"
ALERT_MODE_FALLS = "falls"


def normalize_alert_mode(raw: object) -> str:
    """Режим получателя: «all» — все события, «falls» — только падение / сползание."""
    val = str(raw or "").strip().lower()
    return ALERT_MODE_FALLS if val in ("falls", "fall", "falls_only", "fall_only") else ALERT_MODE_ALL


def is_fall_event(event: object) -> bool:
    state = getattr(event, "state", None)
    return str(getattr(state, "value", state) or "").lower() == "fallen"
BTN_STATUS_TEXT = "📊 Статус"

BOT_COMMANDS_RU: list[dict[str, str]] = [
    {"command": "status", "description": "Текущий кадр и статус охраны"},
    {"command": "photo", "description": "Свежий снимок с камеры"},
    {"command": "report", "description": "Сводка сна и подъёмов за 24 ч"},
    {"command": "clip", "description": "Последний записанный видеоклип"},
    {"command": "pause15", "description": "Я подошёл (пауза 15 минут)"},
    {"command": "pause60", "description": "Пауза 1 час"},
    {"command": "resume", "description": "Включить охрану снова"},
    {"command": "fallsonly", "description": "Присылать мне только падение / сползание"},
    {"command": "allevents", "description": "Присылать мне все события"},
    {"command": "mute", "description": "Отключить уведомления себе"},
    {"command": "unmute", "description": "Включить уведомления себе"},
]


def _is_placeholder_token(token: str) -> bool:
    low = token.strip().lower()
    return (
        not low
        or low.startswith("your_")
        or low.startswith("<")
        or "changeme" in low
        or "example" in low
    )


_COMPACT_KEY_SAFE_RE = re.compile(r"^[A-Za-z0-9_-]{1,48}$")


def _make_compact_alert_key(alert_id: str) -> str:
    """
    Гарантирует компактный ASCII-ключ (<= 48 символов), чтобы callback_data
    вида near:<key> и wrong:<key> строго укладывался в лимит 64 байта UTF-8.
    """
    clean = str(alert_id or "").strip()
    if not clean:
        clean = f"alert_{int(time.time() * 1000)}"
    if _COMPACT_KEY_SAFE_RE.fullmatch(clean):
        return clean
    ascii_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", clean).strip("_")[:34] or "alert"
    digest = hashlib.sha1(clean.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{ascii_slug}_{digest}"


def _build_alert_inline_keyboard(
    alert_id: str,
    acked_by: str = "",
    wrong_marked: bool = False,
) -> str:
    """Компактная однорядная клавиатура только для постов тревог (ровно 2 кнопки)."""
    compact_key = _make_compact_alert_key(alert_id)
    near_btn = {
        "text": "✓ Я рядом" if acked_by else "✅ Я рядом",
        "callback_data": f"near:{compact_key}",
    }
    wrong_btn = (
        {"text": "✓ Ошибка записана", "callback_data": "noop"}
        if wrong_marked
        else {"text": "❌ Ты ошибся", "callback_data": f"wrong:{compact_key}"}
    )
    markup = {"inline_keyboard": [[near_btn, wrong_btn]]}
    return json.dumps(markup, ensure_ascii=False)


def _build_inline_keyboard() -> str:
    """Устаревший хелпер инлайн-сетки (оставлен для обратной совместимости внешних тестов)."""
    markup = {
        "inline_keyboard": [
            [
                {"text": "✅ Я подошёл (пауза 15 мин)", "callback_data": "ack:900"},
                {"text": "⏸ Пауза 1 час", "callback_data": "ack:3600"},
            ],
            [
                {"text": "📸 Свежий кадр", "callback_data": "snap"},
                {"text": "▶️ Включить охрану", "callback_data": "resume"},
            ],
            [
                {"text": "📊 Сводка 24ч", "callback_data": "report"},
                {"text": "🎬 Последний клип", "callback_data": "clip"},
            ],
        ]
    }
    return json.dumps(markup, ensure_ascii=False)


def _build_reply_keyboard(enabled: bool = True, mode: str = ALERT_MODE_ALL) -> str:
    toggle_label = BTN_MUTE_TEXT if enabled else BTN_UNMUTE_TEXT
    mode_label = (
        BTN_ALL_EVENTS_TEXT if normalize_alert_mode(mode) == ALERT_MODE_FALLS else BTN_FALLS_ONLY_TEXT
    )
    markup = {
        "keyboard": [
            [{"text": toggle_label}, {"text": mode_label}],
            [{"text": BTN_PHOTO_TEXT}, {"text": BTN_STATUS_TEXT}],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
    }
    return json.dumps(markup, ensure_ascii=False)


def _extract_chat_meta(chat: dict, sender: dict) -> tuple[str, str, str]:
    chat_id = str(chat.get("id", "")).strip()
    title = str(chat.get("title", "")).strip()
    first = str(sender.get("first_name") or chat.get("first_name") or "").strip()
    last = str(sender.get("last_name") or chat.get("last_name") or "").strip()
    person_name = " ".join(p for p in (first, last) if p).strip()
    name = title or person_name or f"Чат {chat_id}"
    username = str(sender.get("username") or chat.get("username") or "").strip().lstrip("@")
    return chat_id, name, username


def _extract_sender_display(sender: dict, fallback_name: str) -> str:
    first = str(sender.get("first_name") or "").strip()
    last = str(sender.get("last_name") or "").strip()
    person = " ".join(p for p in (first, last) if p).strip()
    if person:
        return person
    uname = str(sender.get("username") or "").strip().lstrip("@")
    if uname:
        return f"@{uname}"
    return fallback_name.strip() or "Пользователь"


@dataclass
class _AlertPostState:
    alert_id: str
    compact_key: str
    base_caption_html: str
    status_suffix_html: str = ""
    acked_by: str = ""
    wrong_marked: bool = False
    wrong_by: str = ""
    visitor_face_saved: bool = False
    has_video: bool = False
    messages: dict[str, int] = field(default_factory=dict)


class TelegramNotifier:
    """
    Двухфазный Telegram-оповещатель с персональным управлением подпиской для нескольких родственников:
    1) Мгновенная отправка JPEG со скелетом + 2 инлайн-кнопки («✅ Я рядом» / «❌ Ты ошибся»);
    2) Бесшовная замена фото на MP4-клип в том же сообщении через editMessageMedia (с кэшем file_id);
    3) Цикл будящих повторов каждые N секунд при неподтверждённом падении;
    4) Хранилище подписчиков data/subscribers.json с персональным флагом enabled
       и постоянной нижней клавиатурой ReplyKeyboardMarkup у каждого пользователя.
    """

    MAX_TRACKED_POSTS: int = 128

    def __init__(
        self,
        bot_token: str,
        chat_ids: tuple[str, ...],
        on_pause_cmd: Callable[[float], None],
        on_resume_cmd: Callable[[], None],
        get_snapshot_jpeg: Callable[[], bytes | None],
        is_fall_unacknowledged: Callable[[], bool],
        repeat_interval_sec: float = 15.0,
        max_repeats: int = 20,
        subscribers_path: Path | None = None,
        bot_username: str = "NoFallenAi",
        get_status_text: Callable[[], str] | None = None,
        on_event: Callable[[str, str], None] | None = None,
        get_report_text: Callable[[], str] | None = None,
        get_latest_clip: Callable[[], tuple[bytes, str] | None] | None = None,
        on_false_alert_cmd: Callable[[str, str, str], Any] | None = None,
        webapp_url: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token = bot_token.strip()
        self._env_chat_ids = tuple(
            cid.strip() for cid in chat_ids if cid.strip() and not _is_placeholder_token(cid)
        )
        self._on_pause = on_pause_cmd
        self._on_resume = on_resume_cmd
        self._get_snapshot = get_snapshot_jpeg
        self._is_fall_unack = is_fall_unacknowledged
        self._repeat_interval = max(10.0, repeat_interval_sec)
        self._max_repeats = max_repeats
        self._subscribers_path = subscribers_path
        self._bot_username = bot_username.strip().lstrip("@") or "NoFallenAi"
        self._get_status_text = get_status_text
        self._on_event = on_event
        self._get_report_text = get_report_text
        self._get_latest_clip = get_latest_clip
        self._on_false_alert = on_false_alert_cmd
        self._webapp_url = (webapp_url or "").strip().rstrip("/")
        self._transport = transport

        self._sub_lock = threading.RLock()
        self._subscribers: dict[str, dict[str, Any]] = {}
        self._load_subscribers()
        self._seed_env_subscribers()

        self._posts_lock = threading.RLock()
        self._posts_by_key: OrderedDict[str, _AlertPostState] = OrderedDict()
        self._key_by_alert_id: dict[str, str] = {}
        self._key_by_msg: dict[tuple[str, int], str] = {}

        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._repeat_lock = threading.Lock()
        self._repeat_active = False

    @property
    def enabled(self) -> bool:
        return not _is_placeholder_token(self._token)

    @property
    def bot_username(self) -> str:
        with self._sub_lock:
            return self._bot_username

    @property
    def webapp_url(self) -> str:
        return self._webapp_url

    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self._token}/{method}"

    def _make_client(self, timeout: float) -> httpx.Client:
        if self._transport is not None:
            return httpx.Client(timeout=timeout, transport=self._transport)
        return httpx.Client(timeout=timeout)

    def _load_subscribers(self) -> None:
        if self._subscribers_path is None or not self._subscribers_path.exists():
            return
        try:
            raw = json.loads(self._subscribers_path.read_text(encoding="utf-8"))
            items = raw.get("subscribers", []) if isinstance(raw, dict) else raw
            if not isinstance(items, list):
                return
            loaded: dict[str, dict[str, Any]] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("chat_id", "")).strip()
                if not cid:
                    continue
                raw_enabled = item.get("enabled", True)
                is_enabled = raw_enabled if isinstance(raw_enabled, bool) else bool(raw_enabled)
                loaded[cid] = {
                    "chat_id": cid,
                    "name": str(item.get("name") or f"Чат {cid}"),
                    "username": str(item.get("username") or "").lstrip("@"),
                    "joined_at": str(item.get("joined_at") or ""),
                    "enabled": is_enabled,
                    "mode": normalize_alert_mode(item.get("mode")),
                }
            with self._sub_lock:
                self._subscribers = loaded
        except Exception as exc:
            logger.warning("Не удалось прочитать subscribers.json: %s", exc)

    def _seed_env_subscribers(self) -> None:
        """Гарантирует присутствие чатов из TG_CHAT_IDS в хранилище с enabled=True при первом старте."""
        if not self._env_chat_ids:
            return
        added = False
        with self._sub_lock:
            for cid in self._env_chat_ids:
                if cid not in self._subscribers:
                    self._subscribers[cid] = {
                        "chat_id": cid,
                        "name": f"Чат из .env ({cid})",
                        "username": "",
                        "joined_at": "статический (.env)",
                        "enabled": True,
                        "mode": ALERT_MODE_ALL,
                    }
                    added = True
        if added and self._subscribers_path is not None:
            self._save_subscribers()

    def _save_subscribers(self) -> None:
        if self._subscribers_path is None:
            return
        try:
            self._subscribers_path.parent.mkdir(parents=True, exist_ok=True)
            with self._sub_lock:
                payload = {"subscribers": list(self._subscribers.values())}
            tmp = self._subscribers_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._subscribers_path)
        except Exception as exc:
            logger.error("Ошибка сохранения subscribers.json: %s", exc)

    def _is_known_subscriber(self, chat_id: str) -> bool:
        cid = chat_id.strip()
        if not cid:
            return False
        with self._sub_lock:
            return cid in self._subscribers or cid in self._env_chat_ids

    def is_authorized_webapp_user(self, chat_id: str) -> bool:
        """
        Проверяет право доступа к защищённому видеопульту Telegram Mini App:
        разрешён любому известному подписчику (subscribers.json или TG_CHAT_IDS),
        даже если у него отключены пуш-уведомления (enabled=False).
        """
        return self._is_known_subscriber(str(chat_id or "").strip())

    def is_subscriber_enabled(self, chat_id: str) -> bool:
        cid = chat_id.strip()
        if not cid:
            return False
        with self._sub_lock:
            sub = self._subscribers.get(cid)
            if sub is not None:
                return bool(sub.get("enabled", True))
            return cid in self._env_chat_ids

    def _target_chats(self, for_fall: bool = True) -> list[str]:
        """
        Чаты с активной подпиской (enabled == True). Для событий, не являющихся падением
        (for_fall=False), исключаются получатели в режиме «только падение / сползание».
        """
        with self._sub_lock:
            seen: set[str] = set()
            result: list[str] = []
            for cid in (*self._env_chat_ids, *self._subscribers.keys()):
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                sub = self._subscribers.get(cid)
                is_enabled = bool(sub.get("enabled", True)) if sub is not None else True
                mode = normalize_alert_mode(sub.get("mode")) if sub is not None else ALERT_MODE_ALL
                if is_enabled and (for_fall or mode == ALERT_MODE_ALL):
                    result.append(cid)
            return result

    def get_subscriber_mode(self, chat_id: str) -> str:
        with self._sub_lock:
            sub = self._subscribers.get(str(chat_id or "").strip())
            return normalize_alert_mode(sub.get("mode")) if sub is not None else ALERT_MODE_ALL

    def _user_keyboard(self, chat_id: str) -> str:
        return _build_reply_keyboard(
            self.is_subscriber_enabled(chat_id),
            self.get_subscriber_mode(chat_id),
        )

    def set_subscriber_mode(self, chat_id: str, mode: str) -> dict[str, object] | None:
        """Персональный режим получателя: все события или только падение / сползание."""
        cid = str(chat_id or "").strip()
        target = normalize_alert_mode(mode)
        changed = False
        sub_name = cid
        with self._sub_lock:
            existing = self._subscribers.get(cid)
            if existing is None and cid not in self._env_chat_ids:
                return None
            if existing is None:
                existing = {
                    "chat_id": cid,
                    "name": f"Чат из .env ({cid})",
                    "username": "",
                    "joined_at": "статический (.env)",
                    "enabled": True,
                    "mode": ALERT_MODE_ALL,
                }
                self._subscribers[cid] = existing
                changed = True
            if normalize_alert_mode(existing.get("mode")) != target:
                existing["mode"] = target
                changed = True
            sub_name = str(existing.get("name") or cid)
        if changed:
            self._save_subscribers()
            if self._on_event:
                mode_ru = "только падения" if target == ALERT_MODE_FALLS else "все события"
                self._on_event("info", f"Telegram-чат {sub_name}: режим тревог «{mode_ru}»")
        for item in self.list_subscribers():
            if item.get("chat_id") == cid:
                return item
        return None

    def get_recipient_chat_ids(self) -> list[str]:
        """Возвращает список активных получателей боевых тревог (enabled == True)."""
        return self._target_chats()

    def list_subscribers(self) -> list[dict[str, object]]:
        with self._sub_lock:
            env_set = set(self._env_chat_ids)
            items: list[dict[str, object]] = []
            for cid in self._env_chat_ids:
                sub = self._subscribers.get(cid)
                items.append(
                    {
                        "chat_id": cid,
                        "name": str(sub["name"]) if sub else f"Чат из .env ({cid})",
                        "username": str(sub["username"]) if sub else "",
                        "joined_at": str(sub["joined_at"]) if sub else "статический (.env)",
                        "source": "env",
                        "removable": False,
                        "enabled": bool(sub.get("enabled", True)) if sub is not None else True,
                        "mode": normalize_alert_mode(sub.get("mode")) if sub is not None else ALERT_MODE_ALL,
                    }
                )
            for cid, sub in self._subscribers.items():
                if cid in env_set:
                    continue
                items.append(
                    {
                        "chat_id": cid,
                        "name": str(sub["name"]),
                        "username": str(sub["username"]),
                        "joined_at": str(sub["joined_at"]),
                        "source": "bot",
                        "removable": True,
                        "enabled": bool(sub.get("enabled", True)),
                        "mode": normalize_alert_mode(sub.get("mode")),
                    }
                )
            return items

    def register_subscriber(
        self,
        chat_id: str,
        name: str = "",
        username: str = "",
        force_enable: bool = False,
    ) -> bool:
        """
        Добавляет или обновляет метаданные подписчика в data/subscribers.json.
        Для нового чата ставит enabled=True; для уже существующего сохраняет его текущий флаг
        (или включает при force_enable=True). Возвращает True, если чат подключился впервые.
        """
        cid = chat_id.strip()
        if not cid:
            return False
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._sub_lock:
            existing = self._subscribers.get(cid)
            already_known = existing is not None
            # Если чат был засеян только из .env со стандартным именем, обновим имя реального человека
            clean_name = name.strip() or (str(existing["name"]) if existing else f"Чат {cid}")
            clean_user = username.strip().lstrip("@") or (
                str(existing["username"]) if existing else ""
            )
            joined_at = (
                str(existing["joined_at"])
                if existing and existing.get("joined_at")
                else now_str
            )
            prev_enabled = bool(existing.get("enabled", True)) if existing is not None else True
            next_enabled = True if (not already_known or force_enable) else prev_enabled

            updated_entry: dict[str, Any] = {
                "chat_id": cid,
                "name": clean_name,
                "username": clean_user,
                "joined_at": joined_at,
                "enabled": next_enabled,
                "mode": normalize_alert_mode(existing.get("mode")) if existing else ALERT_MODE_ALL,
            }
            changed = existing != updated_entry
            self._subscribers[cid] = updated_entry

        if changed:
            self._save_subscribers()

        if not already_known:
            label = f"{clean_name} (@{clean_user})" if clean_user else f"{clean_name} ({cid})"
            logger.info("Подключён новый подписчик Telegram: %s", label)
            if self._on_event:
                self._on_event("info", f"Подключён Telegram-чат: {label}")
            return True
        return False

    def set_subscriber_enabled(
        self,
        chat_id: str,
        enabled: bool | None = None,
    ) -> dict[str, object] | None:
        """
        Включает или выключает доставку боевых уведомлений конкретному подписчику.
        Если enabled=None — инвертирует текущее состояние. Возвращает запись из list_subscribers().
        """
        cid = chat_id.strip()
        if not cid:
            return None

        changed = False
        sub_name = cid
        target_state = True

        with self._sub_lock:
            existing = self._subscribers.get(cid)
            if existing is None and cid not in self._env_chat_ids:
                return None
            if existing is None:
                existing = {
                    "chat_id": cid,
                    "name": f"Чат из .env ({cid})",
                    "username": "",
                    "joined_at": "статический (.env)",
                    "enabled": True,
                }
                self._subscribers[cid] = existing
                changed = True

            current_flag = bool(existing.get("enabled", True))
            target_state = (not current_flag) if enabled is None else bool(enabled)
            if current_flag != target_state:
                existing["enabled"] = target_state
                changed = True
            sub_name = str(existing.get("name") or cid)

        if changed:
            self._save_subscribers()
            if self._on_event:
                action_ru = "включил" if target_state else "отключил"
                self._on_event("info", f"Telegram-чат {sub_name} {action_ru} уведомления")

        for item in self.list_subscribers():
            if item.get("chat_id") == cid:
                return item
        return None

    def remove_subscriber(self, chat_id: str) -> tuple[bool, str]:
        cid = chat_id.strip()
        with self._sub_lock:
            if cid in self._env_chat_ids:
                return False, "Этот чат задан в .env и не удаляется через веб-пульт"
            removed = self._subscribers.pop(cid, None)
        if removed is None:
            return False, "Подписчик не найден"
        self._save_subscribers()
        if self._on_event:
            self._on_event("info", f"Удалён Telegram-чат: {removed.get('name', cid)}")
        return True, "Подписчик удалён"

    def sync_bot_profile_and_commands(self, client: httpx.Client | None = None) -> bool:
        """Проверяет getMe и регистрирует официальное русское меню команд setMyCommands."""
        if not self.enabled:
            return False

        def _run(c: httpx.Client) -> bool:
            me_resp = c.get(self._api_url("getMe"))
            me_data = me_resp.json()
            if not me_data.get("ok"):
                logger.error("Ошибка Telegram getMe: %s", me_data)
                return False
            uname = str((me_data.get("result") or {}).get("username") or "").strip()
            if uname:
                with self._sub_lock:
                    self._bot_username = uname

            has_https_webapp = self._webapp_url.startswith("https://")
            commands_list: list[dict[str, str]] = (
                [
                    {"command": "app", "description": "Открыть видеопульт Mini App"},
                    *BOT_COMMANDS_RU,
                ]
                if has_https_webapp
                else list(BOT_COMMANDS_RU)
            )
            for extra in ({}, {"language_code": "ru"}):
                payload = {"commands": commands_list, **extra}
                cmd_resp = c.post(self._api_url("setMyCommands"), json=payload)
                cmd_data = cmd_resp.json()
                if not cmd_data.get("ok"):
                    logger.warning("Не удалось обновить меню команд (%s): %s", extra, cmd_data)

            if has_https_webapp:
                menu_resp = c.post(
                    self._api_url("setChatMenuButton"),
                    json={
                        "menu_button": {
                            "type": "web_app",
                            "text": "Пульт",
                            "web_app": {"url": self._webapp_url},
                        }
                    },
                )
                menu_data = menu_resp.json()
                if not menu_data.get("ok"):
                    logger.warning("Не удалось установить кнопку меню Mini App: %s", menu_data)

            logger.info("Telegram-бот @%s активен, меню команд зарегистрировано", self.bot_username)
            return True

        try:
            if client is not None:
                return _run(client)
            with self._make_client(10.0) as owned_client:
                return _run(owned_client)
        except Exception as exc:
            logger.warning("Сетевая ошибка инициализации Telegram-бота: %s", exc)
            return False

    def start_polling(self) -> None:
        if not self.enabled:
            logger.warning("TG_BOT_TOKEN не задан — поллинг Telegram не запущен")
            return
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_updates_loop, name="tg-poller", daemon=True
        )
        self._poll_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=4.0)

    def _rebuild_post_suffix_locked(self, post: _AlertPostState) -> str:
        lines: list[str] = []
        if post.acked_by:
            lines.append(f"✅ <b>Принято ({html.escape(post.acked_by)})</b> · пауза 15 мин")
        if post.wrong_marked:
            who = post.wrong_by or "Пользователь"
            face_note = " · лицо «Своего» запомнено" if post.visitor_face_saved else ""
            lines.append(
                f"🛠 <b>Ошибка записана ({html.escape(who)})</b> · сохранено для разбора{face_note}"
            )
        post.status_suffix_html = ("\n\n" + "\n".join(lines)) if lines else ""
        return post.status_suffix_html

    def _format_post_caption_locked(self, post: _AlertPostState) -> str:
        if post.has_video:
            return f"{post.base_caption_html}{post.status_suffix_html}"
        return (
            f"{post.base_caption_html}{post.status_suffix_html}\n\n"
            f"<i>⏳ Видеоклип загружается…</i>"
        )

    def _register_alert_post_locked(self, post: _AlertPostState) -> None:
        prev_key = self._key_by_alert_id.get(post.alert_id)
        if prev_key and prev_key != post.compact_key:
            self._posts_by_key.pop(prev_key, None)
        self._posts_by_key.pop(post.compact_key, None)
        self._posts_by_key[post.compact_key] = post
        self._key_by_alert_id[post.alert_id] = post.compact_key
        for cid, mid in post.messages.items():
            self._key_by_msg[(str(cid), int(mid))] = post.compact_key

        while len(self._posts_by_key) > self.MAX_TRACKED_POSTS:
            old_key, old_post = self._posts_by_key.popitem(last=False)
            if self._key_by_alert_id.get(old_post.alert_id) == old_key:
                self._key_by_alert_id.pop(old_post.alert_id, None)
            for cid, mid in old_post.messages.items():
                if self._key_by_msg.get((str(cid), int(mid))) == old_key:
                    self._key_by_msg.pop((str(cid), int(mid)), None)

    def _find_post_locked(
        self,
        key_or_alert_id: str | None = None,
        chat_id: str | None = None,
        message_id: int | None = None,
    ) -> _AlertPostState | None:
        if key_or_alert_id:
            clean = str(key_or_alert_id).strip()
            if clean in self._posts_by_key:
                return self._posts_by_key[clean]
            mapped_key = self._key_by_alert_id.get(clean)
            if mapped_key and mapped_key in self._posts_by_key:
                return self._posts_by_key[mapped_key]
        if chat_id and message_id is not None:
            msg_key = self._key_by_msg.get((str(chat_id), int(message_id)))
            if msg_key and msg_key in self._posts_by_key:
                return self._posts_by_key[msg_key]
        return None

    def _edit_post_captions(
        self,
        client: httpx.Client,
        post: _AlertPostState,
        clicked_chat_id: str = "",
        clicked_msg_id: int | None = None,
    ) -> None:
        with self._posts_lock:
            caption = self._format_post_caption_locked(post)
            reply_markup = _build_alert_inline_keyboard(
                post.compact_key,
                acked_by=post.acked_by,
                wrong_marked=post.wrong_marked,
            )
            targets = list(post.messages.items())

        edited_pairs: set[tuple[str, int]] = set()
        for cid, mid in targets:
            pair = (str(cid), int(mid))
            edited_pairs.add(pair)
            try:
                client.post(
                    self._api_url("editMessageCaption"),
                    data={
                        "chat_id": str(cid),
                        "message_id": str(mid),
                        "caption": caption,
                        "parse_mode": "HTML",
                        "reply_markup": reply_markup,
                    },
                )
            except Exception as exc:
                logger.debug("Не удалось обновить подпись поста (%s/%s): %s", cid, mid, exc)

        if clicked_chat_id and clicked_msg_id is not None:
            click_pair = (str(clicked_chat_id), int(clicked_msg_id))
            if click_pair not in edited_pairs:
                try:
                    client.post(
                        self._api_url("editMessageReplyMarkup"),
                        data={
                            "chat_id": str(clicked_chat_id),
                            "message_id": str(clicked_msg_id),
                            "reply_markup": reply_markup,
                        },
                    )
                except Exception as exc:
                    logger.debug("Не удалось обновить кнопки повтора (%s/%s): %s", clicked_chat_id, clicked_msg_id, exc)

    def send_alert_photo(
        self,
        event: AlertEvent,
        jpeg_bytes: bytes,
        alert_id: str | None = None,
    ) -> dict[str, int]:
        """Мгновенно отправляет фото только в чаты с enabled=True и возвращает {chat_id: message_id}."""
        if not self.enabled:
            return {}
        recipients = self._target_chats(for_fall=is_fall_event(event))
        eff_alert_id = str(alert_id or "").strip() or f"alert_{int(time.time() * 1000)}"
        compact_key = _make_compact_alert_key(eff_alert_id)

        icon = "🚨" if event.severity == AlertSeverity.CRITICAL else "⚠️"
        now_str = datetime.now().strftime("%H:%M:%S")
        base_caption_html = (
            f"{icon} <b>{html.escape(event.title)}</b>\n"
            f"🕒 Время: <code>{now_str}</code>\n"
            f"{html.escape(event.description)}"
        )
        initial_caption = f"{base_caption_html}\n\n<i>⏳ Видеоклип загружается…</i>"
        reply_markup = _build_alert_inline_keyboard(compact_key)
        msg_ids: dict[str, int] = {}

        if recipients:
            with self._make_client(12.0) as client:
                for cid in recipients:
                    try:
                        resp = client.post(
                            self._api_url("sendPhoto"),
                            data={
                                "chat_id": cid,
                                "caption": initial_caption,
                                "parse_mode": "HTML",
                                "reply_markup": reply_markup,
                            },
                            files={"photo": ("alert.jpg", jpeg_bytes, "image/jpeg")},
                        )
                        payload = resp.json()
                        if payload.get("ok"):
                            msg_ids[cid] = int(payload["result"]["message_id"])
                        else:
                            logger.error("Ошибка sendPhoto (%s): %s", cid, payload)
                    except Exception as exc:
                        logger.error("Сетевая ошибка sendPhoto (%s): %s", cid, exc)
        else:
            logger.warning(
                "Тревога «%s», но нет активных чатов с включёнными уведомлениями в @%s",
                event.title,
                self.bot_username,
            )

        with self._posts_lock:
            post = _AlertPostState(
                alert_id=eff_alert_id,
                compact_key=compact_key,
                base_caption_html=base_caption_html,
                messages=dict(msg_ids),
            )
            self._register_alert_post_locked(post)

        if event.severity == AlertSeverity.CRITICAL:
            self._start_escalation_loop(event.title, alert_id=eff_alert_id)
        return msg_ids

    def replace_alert_photo_with_video(
        self,
        mp4_bytes: bytes,
        reply_msg_ids: dict[str, int],
        title: str,
        alert_id: str | None = None,
    ) -> bool:
        """
        Заменяет ранее отправленное фото тревоги на готовый MP4-клип в том же сообщении
        через editMessageMedia (с кэшированием file_id для 2+ подписчиков и сохранением
        уже нажатых статусов/кнопок). При сбое редактирования делает фоллбэк на sendVideo.
        """
        if not self.enabled or not mp4_bytes:
            return False

        first_pair = next(iter(reply_msg_ids.items()), None) if reply_msg_ids else None
        with self._posts_lock:
            post = self._find_post_locked(
                key_or_alert_id=alert_id,
                chat_id=str(first_pair[0]) if first_pair else None,
                message_id=int(first_pair[1]) if first_pair else None,
            )
            if post is not None:
                post.has_video = True
                eff_key = post.compact_key
            else:
                eff_alert_id = str(alert_id or "").strip() or f"alert_{int(time.time() * 1000)}"
                eff_key = _make_compact_alert_key(eff_alert_id)

        def _snapshot_caption_and_markup() -> tuple[str, str]:
            with self._posts_lock:
                if post is not None:
                    cap = f"{post.base_caption_html}{post.status_suffix_html}"
                    mk = _build_alert_inline_keyboard(
                        post.compact_key,
                        acked_by=post.acked_by,
                        wrong_marked=post.wrong_marked,
                    )
                    return cap, mk
                return (
                    f"🎬 <b>{html.escape(title)}</b>",
                    _build_alert_inline_keyboard(eff_key),
                )

        def _extract_video_file_id(api_payload: dict[str, Any]) -> str | None:
            res = api_payload.get("result")
            if isinstance(res, dict):
                vid_obj = res.get("video")
                if isinstance(vid_obj, dict):
                    fid = str(vid_obj.get("file_id") or "").strip()
                    if fid:
                        return fid
            return None

        def _send_video_fallback(
            client: httpx.Client,
            cid_str: str,
            cached_fid: str | None,
        ) -> tuple[bool, str | None, int | None]:
            cap, mk = _snapshot_caption_and_markup()
            data_fb: dict[str, str] = {
                "chat_id": cid_str,
                "caption": cap,
                "parse_mode": "HTML",
                "supports_streaming": "true",
                "reply_markup": mk,
            }
            try:
                if cached_fid:
                    data_fb["video"] = cached_fid
                    r_fb = client.post(self._api_url("sendVideo"), data=data_fb)
                else:
                    r_fb = client.post(
                        self._api_url("sendVideo"),
                        data=data_fb,
                        files={"video": ("fallguard_clip.mp4", mp4_bytes, "video/mp4")},
                    )
                fb_payload = r_fb.json()
                if fb_payload.get("ok"):
                    new_fid = cached_fid or _extract_video_file_id(fb_payload)
                    res_obj = fb_payload.get("result")
                    new_mid = (
                        int(res_obj["message_id"])
                        if isinstance(res_obj, dict) and "message_id" in res_obj
                        else None
                    )
                    return True, new_fid, new_mid
                logger.error("Ошибка sendVideo (%s): %s", cid_str, str(fb_payload)[:250])
            except Exception as exc:
                logger.error("Сетевая ошибка sendVideo (%s): %s", cid_str, exc)
            return False, cached_fid, None

        uploaded_file_id: str | None = None
        delivered = 0

        with self._make_client(25.0) as client:
            if reply_msg_ids:
                for raw_cid, msg_id in reply_msg_ids.items():
                    cid = str(raw_cid)
                    caption, reply_markup = _snapshot_caption_and_markup()
                    media_spec = {
                        "type": "video",
                        "media": uploaded_file_id if uploaded_file_id else "attach://video",
                        "caption": caption,
                        "parse_mode": "HTML",
                        "supports_streaming": True,
                    }
                    edit_data = {
                        "chat_id": cid,
                        "message_id": str(msg_id),
                        "media": json.dumps(media_spec, ensure_ascii=False),
                        "reply_markup": reply_markup,
                    }
                    edited_ok = False
                    try:
                        if uploaded_file_id:
                            resp = client.post(self._api_url("editMessageMedia"), data=edit_data)
                        else:
                            resp = client.post(
                                self._api_url("editMessageMedia"),
                                data=edit_data,
                                files={"video": ("fallguard_clip.mp4", mp4_bytes, "video/mp4")},
                            )
                        payload = resp.json()
                        if payload.get("ok"):
                            edited_ok = True
                            delivered += 1
                            if not uploaded_file_id:
                                uploaded_file_id = _extract_video_file_id(payload)
                        else:
                            logger.warning(
                                "editMessageMedia вернул отказ (%s/%s), откат на sendVideo: %s",
                                cid,
                                msg_id,
                                str(payload)[:220],
                            )
                    except Exception as exc:
                        logger.warning(
                            "Сбой editMessageMedia (%s/%s), откат на sendVideo: %s",
                            cid,
                            msg_id,
                            exc,
                        )

                    if not edited_ok:
                        fb_ok, uploaded_file_id, new_mid = _send_video_fallback(
                            client,
                            cid,
                            uploaded_file_id,
                        )
                        if fb_ok:
                            delivered += 1
                            if new_mid is not None and post is not None:
                                with self._posts_lock:
                                    post.messages[cid] = new_mid
                                    self._key_by_msg[(cid, new_mid)] = post.compact_key
            else:
                recipients = self._target_chats()
                for cid in recipients:
                    fb_ok, uploaded_file_id, new_mid = _send_video_fallback(
                        client,
                        cid,
                        uploaded_file_id,
                    )
                    if fb_ok:
                        delivered += 1
                        if new_mid is not None and post is not None:
                            with self._posts_lock:
                                post.messages[cid] = new_mid
                                self._key_by_msg[(cid, new_mid)] = post.compact_key

        return delivered > 0

    def send_followup_video(
        self,
        mp4_bytes: bytes,
        reply_msg_ids: dict[str, int],
        title: str,
        alert_id: str | None = None,
    ) -> None:
        """Заменяет пост тревожного фото на готовый MP4-клип (с фоллбэком на sendVideo)."""
        self.replace_alert_photo_with_video(
            mp4_bytes=mp4_bytes,
            reply_msg_ids=reply_msg_ids,
            title=title,
            alert_id=alert_id,
        )

    def send_text(
        self,
        text: str,
        with_buttons: bool = False,
        alert_id: str | None = None,
    ) -> tuple[bool, str]:
        if not self.enabled:
            return False, "Не задан TG_BOT_TOKEN в .env"
        recipients = self._target_chats()
        if not recipients:
            return (
                False,
                f"Нет активных чатов с включёнными уведомлениями — откройте @{self.bot_username}",
            )

        btn_markup: str | None = None
        if with_buttons:
            with self._posts_lock:
                post = self._find_post_locked(key_or_alert_id=alert_id)
                if post is not None:
                    btn_markup = _build_alert_inline_keyboard(
                        post.compact_key,
                        acked_by=post.acked_by,
                        wrong_marked=post.wrong_marked,
                    )
                else:
                    btn_markup = _build_alert_inline_keyboard(alert_id or "alert_escalation")

        last_err = ""
        sent = 0
        with self._make_client(10.0) as client:
            for cid in recipients:
                data: dict[str, str] = {"chat_id": cid, "text": text}
                if btn_markup is not None:
                    data["reply_markup"] = btn_markup
                try:
                    r = client.post(self._api_url("sendMessage"), data=data)
                    body = r.json()
                    if body.get("ok"):
                        sent += 1
                    else:
                        last_err = str(body.get("description", r.text))
                except Exception as exc:
                    last_err = str(exc)
        return (sent > 0), (f"Отправлено в {sent} чат(ов)" if sent > 0 else last_err)

    def send_test_snapshot(self, jpeg_bytes: bytes | None) -> tuple[bool, str]:
        if not self.enabled:
            return False, "Не задан TG_BOT_TOKEN в .env"
        recipients = self._target_chats()
        if not recipients:
            return (
                False,
                f"Нет активных подписчиков — включите уведомления в @{self.bot_username}",
            )
        if not jpeg_bytes:
            return self.send_text(
                "✅ Проверка связи FallGuard: бот подключён (кадр с камеры ещё загружается).",
                with_buttons=False,
            )
        sent = 0
        last_err = ""
        with self._make_client(12.0) as client:
            for cid in recipients:
                try:
                    r = client.post(
                        self._api_url("sendPhoto"),
                        data={
                            "chat_id": cid,
                            "caption": "✅ Проверка FallGuard: камера и кнопки управления активны.",
                            "reply_markup": self._user_keyboard(cid),
                        },
                        files={"photo": ("test.jpg", jpeg_bytes, "image/jpeg")},
                    )
                    body = r.json()
                    if body.get("ok"):
                        sent += 1
                    else:
                        last_err = str(body.get("description", r.text))
                except Exception as exc:
                    last_err = str(exc)
        return (sent > 0), (f"Проверочный снимок доставлен ({sent})" if sent > 0 else last_err)

    def _start_escalation_loop(self, title: str, alert_id: str | None = None) -> None:
        with self._repeat_lock:
            if self._repeat_active:
                return
            self._repeat_active = True

        def _worker() -> None:
            try:
                for attempt in range(1, self._max_repeats + 1):
                    if self._stop_event.wait(self._repeat_interval):
                        return
                    if not self._is_fall_unack():
                        return
                    self.send_text(
                        f"🚨 Повтор #{attempt}: {title}\n"
                        f"Тревога не подтверждена! Нажмите «✅ Я рядом», чтобы остановить сигнал.",
                        with_buttons=True,
                        alert_id=alert_id,
                    )
            finally:
                with self._repeat_lock:
                    self._repeat_active = False

        threading.Thread(target=_worker, name="tg-fall-escalation", daemon=True).start()

    def _send_live_card(
        self,
        client: httpx.Client,
        chat_id: str,
        caption: str,
        fallback_text: str | None = None,
        reply_markup: str | None = None,
    ) -> None:
        snap = self._get_snapshot()
        markup = (
            reply_markup
            if reply_markup is not None
            else self._user_keyboard(chat_id)
        )
        if snap:
            client.post(
                self._api_url("sendPhoto"),
                data={
                    "chat_id": chat_id,
                    "caption": caption,
                    "reply_markup": markup,
                },
                files={"photo": ("live.jpg", snap, "image/jpeg")},
            )
        else:
            client.post(
                self._api_url("sendMessage"),
                data={
                    "chat_id": chat_id,
                    "text": fallback_text or f"{caption}\n(Ожидание видеопотока с камеры…)",
                    "reply_markup": markup,
                },
            )

    def _status_summary_line(self) -> str:
        if self._get_status_text is not None:
            try:
                return self._get_status_text()
            except Exception:
                pass
        return "Система охраны активна"

    def _build_report_text(self) -> str:
        if self._get_report_text is not None:
            try:
                txt = self._get_report_text()
                if txt and txt.strip():
                    return txt.strip()
            except Exception as exc:
                logger.warning("Ошибка формирования сводки ухода: %s", exc)
        return f"📊 Сводка ухода за 24 ч\n{self._status_summary_line()}"

    def _send_report_card(
        self,
        client: httpx.Client,
        chat_id: str,
        suffix: str = "",
        reply_markup: str | None = None,
    ) -> None:
        report_txt = self._build_report_text()
        markup = (
            reply_markup
            if reply_markup is not None
            else self._user_keyboard(chat_id)
        )
        client.post(
            self._api_url("sendMessage"),
            data={
                "chat_id": chat_id,
                "text": f"{report_txt}{suffix}",
                "reply_markup": markup,
            },
        )

    def _send_latest_clip_card(
        self,
        client: httpx.Client,
        chat_id: str,
        suffix: str = "",
        reply_markup: str | None = None,
    ) -> bool:
        clip_pair: tuple[bytes, str] | None = None
        if self._get_latest_clip is not None:
            try:
                clip_pair = self._get_latest_clip()
            except Exception as exc:
                logger.warning("Ошибка чтения последнего клипа: %s", exc)

        markup = (
            reply_markup
            if reply_markup is not None
            else self._user_keyboard(chat_id)
        )
        if clip_pair and clip_pair[0]:
            mp4_bytes, label = clip_pair
            clean_label = str(label or "").strip()
            caption = (
                f"🎬 Последний клип: {clean_label}{suffix}"
                if clean_label
                else f"🎬 Последний записанный видеоклип{suffix}"
            )
            client.post(
                self._api_url("sendVideo"),
                data={
                    "chat_id": chat_id,
                    "caption": caption,
                    "supports_streaming": "true",
                    "reply_markup": markup,
                },
                files={"video": ("latest_clip.mp4", mp4_bytes, "video/mp4")},
            )
            return True

        client.post(
            self._api_url("sendMessage"),
            data={
                "chat_id": chat_id,
                "text": f"🎬 В архиве пока нет записанных видеоклипов тревог.{suffix}",
                "reply_markup": markup,
            },
        )
        return False

    def _poll_updates_loop(self) -> None:
        offset = 0
        self.sync_bot_profile_and_commands()

        while not self._stop_event.is_set():
            try:
                with self._make_client(25.0) as client:
                    resp = client.get(
                        self._api_url("getUpdates"),
                        params={
                            "offset": offset,
                            "timeout": 18,
                            "allowed_updates": json.dumps(["callback_query", "message"]),
                        },
                    )
                    data = resp.json()
                    if not data.get("ok"):
                        self._stop_event.wait(4.0)
                        continue
                    for upd in data.get("result", []):
                        offset = max(offset, int(upd["update_id"]) + 1)
                        try:
                            self._handle_update(client, upd)
                        except Exception as exc:
                            logger.error("Ошибка обработки апдейта Telegram: %s", exc)
            except Exception:
                self._stop_event.wait(3.0)

    def _handle_update(self, client: httpx.Client, upd: dict) -> None:
        cb = upd.get("callback_query")
        if cb:
            msg = cb.get("message") or {}
            chat = msg.get("chat") or {}
            sender = cb.get("from") or {}
            chat_id, chat_name, _ = _extract_chat_meta(chat, sender)
            cb_id = cb.get("id")
            if not chat_id:
                return

            if not self._is_known_subscriber(chat_id):
                if cb_id:
                    client.post(
                        self._api_url("answerCallbackQuery"),
                        data={
                            "callback_query_id": cb_id,
                            "text": "Чат отключён. Отправьте /start для повторной привязки.",
                        },
                    )
                return

            with self._sub_lock:
                sub_entry = self._subscribers.get(chat_id)
                fallback_name = str(sub_entry.get("name") or chat_name) if sub_entry else chat_name
            sender_display = _extract_sender_display(sender, fallback_name)
            raw_mid = msg.get("message_id")
            clicked_msg_id = int(raw_mid) if isinstance(raw_mid, int) else None

            action = str(cb.get("data", ""))
            answer_txt = "Выполнено"
            user_kb = self._user_keyboard(chat_id)

            if action == "noop":
                answer_txt = "Уже отмечено"
            elif action.startswith("near:"):
                key = action.split(":", 1)[1].strip()
                self._on_pause(900.0)
                with self._posts_lock:
                    post = self._find_post_locked(
                        key_or_alert_id=key,
                        chat_id=chat_id,
                        message_id=clicked_msg_id,
                    )
                    if post is not None:
                        post.acked_by = sender_display
                        self._rebuild_post_suffix_locked(post)
                if post is not None:
                    self._edit_post_captions(
                        client,
                        post,
                        clicked_chat_id=chat_id,
                        clicked_msg_id=clicked_msg_id,
                    )
                elif clicked_msg_id is not None:
                    old_cap = html.escape(str(msg.get("caption") or "Тревога FallGuard").strip())
                    new_cap = (
                        f"{old_cap}\n\n"
                        f"✅ <b>Принято ({html.escape(sender_display)})</b> · пауза 15 мин"
                    )
                    client.post(
                        self._api_url("editMessageCaption"),
                        data={
                            "chat_id": chat_id,
                            "message_id": str(clicked_msg_id),
                            "caption": new_cap,
                            "parse_mode": "HTML",
                            "reply_markup": _build_alert_inline_keyboard(
                                key or "alert",
                                acked_by=sender_display,
                            ),
                        },
                    )
                answer_txt = "Принято — охрана на паузе 15 мин"
            elif action.startswith("wrong:"):
                key = action.split(":", 1)[1].strip()
                with self._posts_lock:
                    post = self._find_post_locked(
                        key_or_alert_id=key,
                        chat_id=chat_id,
                        message_id=clicked_msg_id,
                    )
                    target_alert_id = post.alert_id if post is not None else (key or "unknown")
                face_saved = False
                if self._on_false_alert is not None:
                    try:
                        fb_res = self._on_false_alert(target_alert_id, chat_id, sender_display)
                        if isinstance(fb_res, dict) and bool(fb_res.get("visitor_face_saved")):
                            face_saved = True
                    except Exception as exc:
                        logger.error("Ошибка обработки ложной тревоги (%s): %s", target_alert_id, exc)
                if post is not None:
                    with self._posts_lock:
                        post.wrong_marked = True
                        post.wrong_by = sender_display
                        if face_saved:
                            post.visitor_face_saved = True
                        self._rebuild_post_suffix_locked(post)
                    self._edit_post_captions(
                        client,
                        post,
                        clicked_chat_id=chat_id,
                        clicked_msg_id=clicked_msg_id,
                    )
                elif clicked_msg_id is not None:
                    old_cap = html.escape(str(msg.get("caption") or "Тревога FallGuard").strip())
                    face_note = " · лицо «Своего» запомнено" if face_saved else ""
                    new_cap = (
                        f"{old_cap}\n\n"
                        f"🛠 <b>Ошибка записана ({html.escape(sender_display)})</b> · сохранено для разбора{face_note}"
                    )
                    client.post(
                        self._api_url("editMessageCaption"),
                        data={
                            "chat_id": chat_id,
                            "message_id": str(clicked_msg_id),
                            "caption": new_cap,
                            "parse_mode": "HTML",
                            "reply_markup": _build_alert_inline_keyboard(
                                key or "alert",
                                wrong_marked=True,
                            ),
                        },
                    )
                answer_txt = (
                    "Ошибка сохранена, лицо «Своего» запомнено"
                    if face_saved
                    else "Ошибка сохранена для разбора, тревога снята"
                )
            elif action.startswith("ack:"):
                secs = float(action.split(":", 1)[1])
                self._on_pause(secs)
                mins = int(secs // 60)
                with self._posts_lock:
                    legacy_post = self._find_post_locked(
                        chat_id=chat_id,
                        message_id=clicked_msg_id,
                    )
                    if legacy_post is not None:
                        legacy_post.acked_by = sender_display
                        self._rebuild_post_suffix_locked(legacy_post)
                if legacy_post is not None:
                    self._edit_post_captions(
                        client,
                        legacy_post,
                        clicked_chat_id=chat_id,
                        clicked_msg_id=clicked_msg_id,
                    )
                elif clicked_msg_id is not None and (
                    msg.get("caption") or msg.get("photo") or msg.get("video")
                ):
                    old_cap = html.escape(str(msg.get("caption") or "Тревога FallGuard").strip())
                    client.post(
                        self._api_url("editMessageCaption"),
                        data={
                            "chat_id": chat_id,
                            "message_id": str(clicked_msg_id),
                            "caption": (
                                f"{old_cap}\n\n"
                                f"✅ <b>Принято ({html.escape(sender_display)})</b> · пауза {mins} мин"
                            ),
                            "parse_mode": "HTML",
                            "reply_markup": _build_alert_inline_keyboard(
                                "legacy_ack",
                                acked_by=sender_display,
                            ),
                        },
                    )
                else:
                    client.post(
                        self._api_url("sendMessage"),
                        data={
                            "chat_id": chat_id,
                            "text": f"✅ Принято. Охрана поставлена на паузу ({mins} мин).",
                            "reply_markup": user_kb,
                        },
                    )
                answer_txt = f"Охрана на паузе на {mins} мин"
            elif action == "resume":
                self._on_resume()
                answer_txt = "Охрана снова активна"
                client.post(
                    self._api_url("sendMessage"),
                    data={
                        "chat_id": chat_id,
                        "text": "▶️ Охрана снова включена.",
                        "reply_markup": user_kb,
                    },
                )
            elif action == "snap":
                status_line = self._status_summary_line()
                self._send_live_card(
                    client,
                    chat_id,
                    caption=f"📸 Свежий кадр FallGuard\n{status_line}",
                    reply_markup=user_kb,
                )
                answer_txt = "Снимок отправлен"
            elif action == "report":
                self._send_report_card(client, chat_id, reply_markup=user_kb)
                answer_txt = "Сводка за 24 ч отправлена"
            elif action == "clip":
                sent_clip = self._send_latest_clip_card(
                    client,
                    chat_id,
                    reply_markup=user_kb,
                )
                answer_txt = "Видеоклип отправлен" if sent_clip else "Архив клипов пока пуст"

            if cb_id:
                client.post(
                    self._api_url("answerCallbackQuery"),
                    data={"callback_query_id": cb_id, "text": answer_txt},
                )
            return

        msg = upd.get("message")
        if not msg:
            return

        chat = msg.get("chat") or {}
        sender = msg.get("from") or {}
        chat_id, name, username = _extract_chat_meta(chat, sender)
        if not chat_id:
            return

        raw_txt = str(msg.get("text") or msg.get("caption") or "").strip()
        cmd = raw_txt.split()[0].lower().split("@")[0] if raw_txt else ""

        is_enable_cmd = raw_txt == BTN_UNMUTE_TEXT or cmd in ("/unmute", "/on", "/start")
        is_disable_cmd = raw_txt == BTN_MUTE_TEXT or cmd in ("/mute", "/off")

        # Регистрируем нового участника (существующему не перезатираем его персональный enabled)
        is_new = self.register_subscriber(
            chat_id,
            name=name,
            username=username,
            force_enable=is_enable_cmd,
        )

        if is_disable_cmd:
            self.set_subscriber_enabled(chat_id, False)
            client.post(
                self._api_url("sendMessage"),
                data={
                    "chat_id": chat_id,
                    "text": (
                        "🔕 Уведомления о тревогах для вашего чата отключены.\n"
                        "Другие подключённые родственники продолжают получать оповещения.\n"
                        "Нажмите «🔔 Включить уведомления», чтобы снова принимать тревоги."
                    ),
                    "reply_markup": self._user_keyboard(chat_id),
                },
            )
            return

        if raw_txt == BTN_UNMUTE_TEXT or cmd in ("/unmute", "/on"):
            self.set_subscriber_enabled(chat_id, True)
            status_line = self._status_summary_line()
            client.post(
                self._api_url("sendMessage"),
                data={
                    "chat_id": chat_id,
                    "text": (
                        "🔔 Уведомления о тревогах включены!\n"
                        f"{status_line}"
                    ),
                    "reply_markup": self._user_keyboard(chat_id),
                },
            )
            return

        if raw_txt in (BTN_FALLS_ONLY_TEXT, BTN_ALL_EVENTS_TEXT) or cmd in ("/fallsonly", "/allevents"):
            want_falls = raw_txt == BTN_FALLS_ONLY_TEXT or cmd == "/fallsonly"
            self.set_subscriber_mode(chat_id, ALERT_MODE_FALLS if want_falls else ALERT_MODE_ALL)
            client.post(
                self._api_url("sendMessage"),
                data={
                    "chat_id": chat_id,
                    "text": (
                        "🚨 Режим «Только падения»: вам будут приходить только падение и сползание.\n"
                        "Посадка, выход из кровати и жест рукой — без уведомлений (видны в хронике пульта)."
                        if want_falls
                        else "📣 Режим «Все события»: посадка, выход из кровати, жест рукой и падения."
                    ),
                    "reply_markup": self._user_keyboard(chat_id),
                },
            )
            return

        user_kb = self._user_keyboard(chat_id)
        joined_note = "\n🔔 Чат автоматически подключён к тревогам FallGuard." if is_new else ""

        if cmd == "/pause15":
            self._on_pause(900.0)
            client.post(
                self._api_url("sendMessage"),
                data={
                    "chat_id": chat_id,
                    "text": f"✅ Принято — я подошёл. Охрана на паузе на 15 минут.{joined_note}",
                    "reply_markup": user_kb,
                },
            )
            return

        if cmd == "/pause60":
            self._on_pause(3600.0)
            client.post(
                self._api_url("sendMessage"),
                data={
                    "chat_id": chat_id,
                    "text": f"⏸ Охрана поставлена на паузу на 1 час.{joined_note}",
                    "reply_markup": user_kb,
                },
            )
            return

        if cmd == "/resume":
            self._on_resume()
            client.post(
                self._api_url("sendMessage"),
                data={
                    "chat_id": chat_id,
                    "text": f"▶️ Охрана снова включена.{joined_note}",
                    "reply_markup": user_kb,
                },
            )
            return

        if raw_txt == BTN_STATUS_TEXT or cmd in ("/status", "/report"):
            self._send_report_card(client, chat_id, suffix=joined_note, reply_markup=user_kb)
            return

        if cmd == "/clip":
            self._send_latest_clip_card(client, chat_id, suffix=joined_note, reply_markup=user_kb)
            return

        if cmd == "/app":
            if self._webapp_url.startswith("https://"):
                webapp_kb = json.dumps(
                    {
                        "inline_keyboard": [
                            [
                                {
                                    "text": "📱 Открыть пульт FallGuard",
                                    "web_app": {"url": self._webapp_url},
                                }
                            ]
                        ]
                    },
                    ensure_ascii=False,
                )
                client.post(
                    self._api_url("sendMessage"),
                    data={
                        "chat_id": chat_id,
                        "text": (
                            f"📱 Нажмите кнопку ниже или синюю кнопку «Пульт» рядом с полем ввода, "
                            f"чтобы открыть защищённый видеопульт FallGuard.{joined_note}"
                        ),
                        "reply_markup": webapp_kb,
                    },
                )
            else:
                client.post(
                    self._api_url("sendMessage"),
                    data={
                        "chat_id": chat_id,
                        "text": f"📱 Внешний HTTPS-адрес видеопульта Mini App не настроен.{joined_note}",
                        "reply_markup": user_kb,
                    },
                )
            return

        status_line = self._status_summary_line()
        if raw_txt == BTN_PHOTO_TEXT or cmd == "/photo":
            self._send_live_card(
                client,
                chat_id,
                caption=f"📸 Свежий кадр FallGuard\n{status_line}{joined_note}",
                reply_markup=user_kb,
            )
            return

        if is_new or cmd == "/start":
            self.set_subscriber_enabled(chat_id, True)
            welcome_caption = (
                f"🛡 Чат подключён к системе охраны FallGuard!\n"
                f"Сюда будут мгновенно приходить снимки и видео при подъёме или падении бабушки.\n\n"
                f"{status_line}\n\n"
                f"Используйте синюю кнопку «Пульт» (или /app) для открытия живого видеопульта "
                f"и нижние кнопки для управления своими уведомлениями, снимка и статуса."
            )
            self._send_live_card(
                client,
                chat_id,
                caption=welcome_caption,
                reply_markup=self._user_keyboard(chat_id),
            )
        else:
            self._send_live_card(
                client,
                chat_id,
                caption=f"📸 Текущий статус FallGuard\n{status_line}",
                reply_markup=user_kb,
            )
