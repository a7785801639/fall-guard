package app.fallguard

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.VibrationEffect
import android.os.Vibrator
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import org.json.JSONObject

/** Данные активной тревоги для полноэкранного окна. */
data class AlarmInfo(
    val id: String,
    val title: String,
    val description: String,
    val ts: Long,
    val imageUrl: String,
    val base: String,
    val kind: Signals.Kind = Signals.Kind.FALL,
    /** Тревогу держит сервер (падение, критическое) — он же её и снимет. */
    val serverTracked: Boolean = true,
)

/**
 * Фоновый сервис: держит long-poll к серверу FallGuard и поднимает тревогу.
 * Падение / сползание — сирена на канале будильника (звучит и в беззвучном режиме)
 * и окно поверх экрана блокировки; прочие события — обычное уведомление.
 */
class AlertService : Service() {
    private lateinit var prefs: Prefs
    private lateinit var nm: NotificationManager
    private val ui = Handler(Looper.getMainLooper())
    @Volatile private var running = false
    private var worker: Thread? = null
    private var player: MediaPlayer? = null
    private var savedAlarmVolume = -1
    private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var ringing = false
    @Volatile private var silencedUntilQuiet = false
    @Volatile private var linkText = "Подключение к серверу…"
    @Volatile private var linkTitle = "FallGuard на страже"

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        prefs = Prefs(this)
        nm = getSystemService(NotificationManager::class.java)
        createChannels()
        watchNetwork()
    }

    // ---------------------------------------------------------------- смена сети
    @Volatile private var activeConn: java.net.HttpURLConnection? = null
    private val wake = Object()
    @Volatile private var netChanged = false
    private var netCallback: android.net.ConnectivityManager.NetworkCallback? = null

    /**
     * Ушли из дома или вернулись: запрос по старой сети висел бы до таймаута (до 40 с) —
     * рвём его сразу и переподключаемся, заново выбирая адрес (дома — домашний).
     */
    private fun watchNetwork() {
        val cm = getSystemService(android.net.ConnectivityManager::class.java) ?: return
        val cb = object : android.net.ConnectivityManager.NetworkCallback() {
            private var current: android.net.Network? = null
            private var seen = false // первый вызов приходит сразу при регистрации — это не смена сети
            override fun onAvailable(network: android.net.Network) {
                val changed = seen && network != current
                seen = true
                current = network
                if (changed) reconnectNow()
            }
            override fun onLost(network: android.net.Network) {
                if (network == current) { current = null; reconnectNow() }
            }
        }
        try { cm.registerDefaultNetworkCallback(cb); netCallback = cb } catch (_: Exception) {}
    }

    private fun reconnectNow() {
        netChanged = true
        val c = activeConn
        if (c != null) Thread { try { c.disconnect() } catch (_: Exception) {} }.start()
        // Будим из паузы между попытками. Не interrupt(): флаг прерывания сорвал бы следующий запрос
        synchronized(wake) { wake.notifyAll() }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        goForeground()
        if (!prefs.paired) {
            stopSelf()
            return START_NOT_STICKY
        }
        when (intent?.action) {
            ACTION_ACK -> {
                // Проверочная тревога (без сервера) только замолкает — охрану на паузу не ставим
                val test = current?.base?.isEmpty() == true
                silencedUntilQuiet = true; stopAlarm()
                if (!test) post("/api/pause", JSONObject().put("seconds", 900))
            }
            ACTION_FALSE -> {
                val id = intent.getStringExtra(EXTRA_ID) ?: ""
                val test = current?.base?.isEmpty() == true
                silencedUntilQuiet = true
                stopAlarm()
                when {
                    test -> {}
                    id.isNotEmpty() -> post("/api/feedback/$id/false_alarm", JSONObject().put("reporter", Build.MODEL))
                    else -> post("/api/pause", JSONObject().put("seconds", 900))
                }
            }
            ACTION_SILENCE -> { silencedUntilQuiet = true; stopAlarm() }
            ACTION_NEARBY -> {
                // «Я рядом» из уведомления о событии или с иконки приложения: пауза на 15 минут
                val id = intent.getStringExtra(EXTRA_ID) ?: ""
                post("/api/pause", JSONObject().put("seconds", 900)) { ok ->
                    if (ok && id.isNotEmpty()) nm.cancel(eventNotifId(id))
                    val msg = if (ok) "Пауза 15 минут — вы рядом" else "Нет связи с сервером — пауза не включена"
                    ui.post { android.widget.Toast.makeText(this, msg, android.widget.Toast.LENGTH_SHORT).show() }
                }
            }
            ACTION_TEST_SIGNAL -> testSignal(Signals.Kind.entries.firstOrNull { it.name == intent.getStringExtra(EXTRA_ID) } ?: Signals.Kind.FALL)
            ACTION_TEST -> startAlarm(
                AlarmInfo("", "Проверка тревоги", "Так прозвучит сигнал о падении. Это только проверка.", ServerClock.now(), "", ""),
            )
        }
        ensureWorker()
        return START_STICKY
    }

    private fun ensureWorker() {
        if (!running) {
            running = true
            worker = Thread({ pollLoop() }, "fallguard-poll").also { it.start() }
        }
    }

    override fun onDestroy() {
        running = false
        worker?.interrupt()
        stopAlarm()
        netCallback?.let { try { getSystemService(android.net.ConnectivityManager::class.java)?.unregisterNetworkCallback(it) } catch (_: Exception) {} }
        super.onDestroy()
    }

    // ---------------------------------------------------------------- опрос сервера
    private fun pollLoop() {
        var base: String? = null
        var lastOk = System.currentTimeMillis()
        var lostWarned = false
        var backoff = 2000L
        while (running) {
            if (netChanged) { netChanged = false; base = null; backoff = 2000L }
            if (base == null) base = Api.reachableBase(prefs)
            val res = if (base == null) HttpResult(-1, "") else Api.request(
                base,
                "/api/alerts/poll?after=${prefs.cursor}&wait=${if (ringing) 3 else 25}",
                token = prefs.token,
                connectMs = 5000,
                readMs = 40000,
                onOpen = { activeConn = it },
            )
            activeConn = null
            when {
                res.ok -> {
                    backoff = 2000L
                    lastOk = System.currentTimeMillis()
                    if (lostWarned) { lostWarned = false; nm.cancel(NOTIF_LOST) }
                    handlePoll(res.json(), base!!)
                    maybeCheckUpdate()
                }
                res.code == 401 || res.code == 403 -> {
                    notifyEvent(NOTIF_LOST, "Телефон отвязан от FallGuard", "Откройте приложение и подключите телефон заново.")
                    prefs.token = ""
                    running = false
                    ui.post { stopSelf() }
                }
                else -> {
                    base = null
                    linkTitle = "Нет связи с сервером"
                    setLink("Переподключаюсь…")
                    Glance.update(this, null, false, "", false)
                    if (!lostWarned && System.currentTimeMillis() - lastOk > LOST_AFTER_MS) {
                        lostWarned = true
                        notifyEvent(NOTIF_LOST, "Нет связи с FallGuard", "Тревоги сейчас не дойдут до этого телефона. Проверьте интернет и сервер.", kind = Signals.Kind.LINK)
                        if (Sounds.style(this, Signals.Kind.LINK) != Signals.Style.SILENT) Sounds.playNotify(this, Signals.Kind.LINK)
                    }
                    // Сеть сменилась — пробуем сразу, без нарастающей паузы
                    if (netChanged) continue
                    try { synchronized(wake) { if (!netChanged) wake.wait(backoff) } } catch (_: InterruptedException) { return }
                    backoff = (backoff * 2).coerceAtMost(15000L)
                }
            }
        }
    }

    /** Проверка одного типа сигнала ровно так, как он придёт: тревогой, уведомлением со звуком или без. */
    private fun testSignal(kind: Signals.Kind) {
        val title = "Проверка: ${kind.title.lowercase()}"
        when (Signals.effectiveStyle(kind, "", Sounds.style(this, kind))) {
            Signals.Style.ALARM -> startAlarm(AlarmInfo("", title, "Так придёт этот сигнал. Это только проверка.", ServerClock.now(), "", "", kind, serverTracked = false))
            Signals.Style.NOTIFY -> {
                val am = getSystemService(AudioManager::class.java)
                val quiet = am != null && am.ringerMode != AudioManager.RINGER_MODE_NORMAL
                notifyEvent(NOTIF_EVENT, title, if (quiet) "Телефон в беззвучном режиме — уведомление придёт без звука." else "Так придёт уведомление со звуком.", kind = kind)
                Sounds.playNotify(this, kind)
                if (quiet) ui.post { android.widget.Toast.makeText(this, "Телефон в беззвучном — уведомление придёт без звука", android.widget.Toast.LENGTH_LONG).show() }
            }
            Signals.Style.SILENT -> notifyEvent(NOTIF_EVENT, title, "Так придёт уведомление без звука.", kind = kind)
        }
    }

    // ---------------------------------------------------------------- обновления
    @Volatile private var updateThread: Thread? = null

    /** Раз в 12 часов спрашиваем GitHub о новой версии; во время тревоги — никогда. */
    private fun maybeCheckUpdate() {
        if (!Updater.enabled || ringing || updateThread?.isAlive == true) return
        updateThread = Thread {
            val rel = Updater.checkIfDue(this) ?: return@Thread
            if (ringing || Updater.notifiedFor(this) >= rel.code) return@Thread
            Updater.markNotified(this, rel.code)
            val open = PendingIntent.getActivity(
                this, 9, Intent(this, SettingsActivity::class.java).putExtra(SettingsActivity.EXTRA_UPDATE, true)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
            val n = NotificationCompat.Builder(this, CH_EVENTS)
                .setSmallIcon(R.drawable.ic_stat)
                .setContentTitle("Доступно обновление FallGuard ${rel.name}")
                .setContentText("Нажмите, чтобы установить")
                .setAutoCancel(true)
                .setContentIntent(open)
                .build()
            try { nm.notify(NOTIF_UPDATE, n) } catch (_: Exception) {}
        }.also { it.start() }
    }

    private fun handlePoll(j: JSONObject, base: String) {
        val raw = j.optJSONArray("alerts")
        val list = mutableListOf<Signals.Alert>()
        val byId = HashMap<String, JSONObject>()
        if (raw != null) for (i in 0 until raw.length()) {
            val a = raw.getJSONObject(i)
            list += Signals.Alert(a.optString("id"), a.optString("state"), a.optString("severity"), a.optBoolean("is_fall"))
            byId[a.optString("id")] = a
        }
        ServerClock.sync(j.optDouble("server_ts", 0.0))
        val cursor = j.optLong("cursor", prefs.cursor)
        if (cursor > prefs.cursor) prefs.cursor = cursor

        val alarmActive = j.optBoolean("alarm_active")
        val actions = Signals.decide(list, alarmActive, ringing, silencedUntilQuiet, current?.serverTracked ?: true, j.optString("state")) { Sounds.style(this, it) }
        for (act in actions) when (act) {
            is Signals.Action.Notify -> {
                val a = byId[act.alert.id] ?: continue
                // У каждого события своё уведомление: поздний кадр не перезапишет чужое
                val nid = eventNotifId(act.alert.id)
                // «Я рядом» — только когда речь о самой бабушке, не о камере или связи
                val nearbyId = if (act.kind in NEARBY_KINDS) act.alert.id else ""
                notifyEvent(nid, a.optString("title"), a.optString("description"), alertId = nearbyId, kind = act.kind)
                if (act.sound) Sounds.playNotify(this, act.kind)
                attachSnapshot(nid, a.optString("title"), a.optString("description"), base, a.optString("image_url"), nearbyId, act.kind)
            }
            is Signals.Action.Ring -> {
                val a = byId[act.alert.id] ?: continue
                silencedUntilQuiet = false
                val info = AlarmInfo(
                    a.optString("id"), a.optString("title"), a.optString("description"),
                    (a.optDouble("ts", 0.0) * 1000).toLong(), a.optString("image_url"), base, act.kind,
                    Signals.serverTracked(act.alert),
                )
                ui.post { startAlarm(info) }
            }
            Signals.Action.Stop -> {
                silencedUntilQuiet = false
                if (current?.base?.isNotEmpty() == true) {
                    // Тревогу сняли на сервере: подошёл свой или кто-то нажал «Принято»
                    val note = if (j.optString("state") == "caregiver") "Рядом свой — тревога снята" else "Тревога снята"
                    ui.post { stopAlarm() }
                    notifyEvent(NOTIF_EVENT, note, j.optString("status_line"))
                }
            }
            Signals.Action.RingUnconfirmed -> {
                // Тревога на сервере ещё не подтверждена (например, телефон только что включили)
                val info = AlarmInfo("", "Тревога не подтверждена", j.optString("status_line"), ServerClock.now(), "", base)
                ui.post { startAlarm(info) }
            }
        }
        if (!alarmActive && actions.none { it is Signals.Action.Ring }) silencedUntilQuiet = false
        val mode = when (j.optString("mode")) {
            "falls" -> "только падения"
            "standup" -> "только вставание"
            else -> "все события"
        }
        val enabled = j.optBoolean("enabled", true)
        val state = j.optString("state")
        val (title, _) = Glance.describe(state, alarmActive)
        linkTitle = title
        setLink(j.optString("status_line", "На связи") + (if (enabled) " · $mode" else " · тревоги выключены"))
        Glance.update(this, state, alarmActive, if (enabled) "" else "тревоги выключены", true)
    }

    private fun post(path: String, body: JSONObject, done: ((Boolean) -> Unit)? = null) {
        Thread {
            val base = Api.reachableBase(prefs)
            val ok = base != null && Api.request(base, path, "POST", prefs.token, body).ok
            done?.invoke(ok)
        }.start()
    }

    // ---------------------------------------------------------------- тревога
    private fun startAlarm(info: AlarmInfo) {
        current = info
        ringing = true
        val full = PendingIntent.getActivity(
            this, 1, Intent(this, AlarmActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val n = NotificationCompat.Builder(this, CH_ALARM)
            .setSmallIcon(R.drawable.ic_stat)
            .setContentTitle(info.title)
            .setContentText(info.description)
            .setStyle(NotificationCompat.BigTextStyle().bigText(info.description))
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setOngoing(true)
            .setColor(ContextCompat.getColor(this, R.color.alarm))
            .setContentIntent(full)
            .setFullScreenIntent(full, true)
            .addAction(0, "Я рядом", servicePending(ACTION_ACK, info.id, 2))
            .addAction(0, "Ложная", servicePending(ACTION_FALSE, info.id, 3))
            .build()
        nm.notify(NOTIF_ALARM, n)
        if (!silencedUntilQuiet) {
            // Пока звучит «встала», пришло падение — переключаемся на звук падения
            if (player != null && sirenKind != info.kind) stopSiren()
            startSiren(info.kind)
        }
        // Если экран уже включён и приложение на виду — открываем окно сразу
        try { startActivity(Intent(this, AlarmActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) } catch (_: Exception) {}
    }

    private var sirenKind: Signals.Kind? = null

    private fun startSiren(kind: Signals.Kind = Signals.Kind.FALL) {
        if (player != null) return
        sirenKind = kind
        try {
            val pm = getSystemService(PowerManager::class.java)
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "fallguard:alarm").apply { acquire(15 * 60 * 1000L) }
            val am = getSystemService(AudioManager::class.java)
            savedAlarmVolume = am.getStreamVolume(AudioManager.STREAM_ALARM)
            am.setStreamVolume(AudioManager.STREAM_ALARM, am.getStreamMaxVolume(AudioManager.STREAM_ALARM), 0)
        } catch (_: Exception) {}
        // Выбранный звук на канале будильника (звучит и в беззвучном режиме); не открылся — встроенная сирена,
        // не открылась и она — системный будильник
        val siren = android.net.Uri.parse("android.resource://$packageName/${R.raw.siren}")
        val gain = Sounds.alarmGain(this, kind)
        player = Sounds.play(this, Sounds.uriOf(this, kind), AudioAttributes.USAGE_ALARM, loop = true, fallback = siren, gain = gain)
            ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)?.let { Sounds.play(this, it, AudioAttributes.USAGE_ALARM, loop = true, gain = gain) }
        try {
            val vib = getSystemService(Vibrator::class.java)
            val attrs = AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).build()
            vib?.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 700, 400), 0), attrs)
        } catch (_: Exception) {}
        // Предохранитель: сирена не звучит дольше 15 минут без подтверждения
        ui.removeCallbacks(sirenTimeout)
        ui.postDelayed(sirenTimeout, 15 * 60 * 1000L)
    }

    private val sirenTimeout = Runnable { stopSiren() }

    private fun stopSiren() {
        ui.removeCallbacks(sirenTimeout)
        try { player?.stop() } catch (_: Exception) {}
        try { player?.release() } catch (_: Exception) {}
        player = null
        try { getSystemService(Vibrator::class.java)?.cancel() } catch (_: Exception) {}
        if (savedAlarmVolume >= 0) {
            try { getSystemService(AudioManager::class.java).setStreamVolume(AudioManager.STREAM_ALARM, savedAlarmVolume, 0) } catch (_: Exception) {}
            savedAlarmVolume = -1
        }
        try { if (wakeLock?.isHeld == true) wakeLock?.release() } catch (_: Exception) {}
        wakeLock = null
    }

    private fun stopAlarm() {
        stopSiren()
        ringing = false
        current = null
        nm.cancel(NOTIF_ALARM)
        onAlarmStopped?.invoke()
    }

    // ---------------------------------------------------------------- уведомления
    private fun servicePending(action: String, id: String, req: Int): PendingIntent =
        PendingIntent.getService(
            this, req, Intent(this, AlertService::class.java).setAction(action).putExtra(EXTRA_ID, id),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

    private fun openAppPending(): PendingIntent =
        PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)

    private fun createChannels() {
        val alarm = NotificationChannel(CH_ALARM, getString(R.string.ch_alarm), NotificationManager.IMPORTANCE_HIGH).apply {
            description = getString(R.string.ch_alarm_desc)
            setSound(null, null) // звук даёт сирена на канале будильника
            enableVibration(false)
            setBypassDnd(true)
            lockscreenVisibility = Notification.VISIBILITY_PUBLIC
        }
        val events = NotificationChannel(CH_EVENTS, getString(R.string.ch_events), NotificationManager.IMPORTANCE_HIGH).apply {
            description = getString(R.string.ch_events_desc)
            setSound(null, null) // звук выбирается в настройках приложения и играется им самим
        }
        try { nm.deleteNotificationChannel("events_v1") } catch (_: Exception) {}
        val link = NotificationChannel(CH_LINK, getString(R.string.ch_link), NotificationManager.IMPORTANCE_LOW).apply {
            description = getString(R.string.ch_link_desc)
            setShowBadge(false)
        }
        nm.createNotificationChannels(listOf(alarm, events, link))
    }

    private fun linkNotification(): Notification =
        NotificationCompat.Builder(this, CH_LINK)
            .setSmallIcon(R.drawable.ic_stat)
            .setContentTitle(linkTitle)
            .setContentText(linkText)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openAppPending())
            .build()

    private fun goForeground() {
        val type = if (Build.VERSION.SDK_INT >= 34) ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE else 0
        try {
            ServiceCompat.startForeground(this, NOTIF_LINK, linkNotification(), type)
        } catch (_: Exception) {}
    }

    private var shownTitle = ""
    private fun setLink(text: String) {
        if (text == linkText && linkTitle == shownTitle) return
        linkText = text
        shownTitle = linkTitle
        try { nm.notify(NOTIF_LINK, linkNotification()) } catch (_: Exception) {}
    }

    /**
     * Снимок события в уведомлении: видно, что происходит, прямо из шторки.
     * Сначала уведомление уходит сразу, без картинки; кадр подставляется, когда сервер его сохранит.
     */
    private fun eventNotifId(alertId: String): Int = 1000 + Math.floorMod(alertId.hashCode(), 5000)

    private fun attachSnapshot(id: Int, title: String, text: String, base: String, path: String, alertId: String, kind: Signals.Kind) {
        if (base.isEmpty() || path.isEmpty()) return
        val token = prefs.token
        Thread {
            for (attempt in 0 until 3) {
                val data = Api.bytes(base, path, token)
                val bmp = data?.let { android.graphics.BitmapFactory.decodeByteArray(it, 0, it.size) }
                if (bmp != null) {
                    // Смахнули уведомление — не возвращаем его ради картинки
                    val still = try { nm.activeNotifications.any { it.id == id } } catch (_: Exception) { false }
                    if (still) notifyEvent(id, title, text, bmp, silentUpdate = true, alertId = alertId, kind = kind)
                    return@Thread
                }
                try { Thread.sleep(1500) } catch (_: InterruptedException) { return@Thread }
            }
        }.start()
    }

    private fun notifyEvent(
        id: Int, title: String, text: String, picture: android.graphics.Bitmap? = null,
        silentUpdate: Boolean = false, alertId: String = "", kind: Signals.Kind? = null,
    ) {
        // Цвет и значок — по событию: в шторке и строке состояния сразу видно, что случилось
        val (icon, color) = when (kind) {
            Signals.Kind.FALL -> R.drawable.ic_k_fall to R.color.alarm
            Signals.Kind.STANDUP -> R.drawable.ic_k_walk to R.color.warn
            Signals.Kind.SITTING -> R.drawable.ic_k_sit to R.color.warn
            Signals.Kind.HELP -> R.drawable.ic_k_hand to R.color.warn
            Signals.Kind.OTHER -> R.drawable.ic_k_alert to R.color.alarm
            Signals.Kind.LINK -> R.drawable.ic_k_link to R.color.muted
            null -> R.drawable.ic_stat to R.color.calm
        }
        val style: NotificationCompat.Style = if (picture != null)
            NotificationCompat.BigPictureStyle().bigPicture(picture).setSummaryText(text).bigLargeIcon(null as android.graphics.Bitmap?)
        else NotificationCompat.BigTextStyle().bigText(text)
        val n = NotificationCompat.Builder(this, CH_EVENTS)
            .setSmallIcon(icon)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(style)
            .setLargeIcon(picture)
            .setOnlyAlertOnce(silentUpdate)
            .setColor(ContextCompat.getColor(this, color))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .setContentIntent(openAppPending())
            .apply { if (alertId.isNotEmpty()) addAction(0, "Я рядом · 15 мин", servicePending(ACTION_NEARBY, alertId, id)) }
            .build()
        try { nm.notify(id, n) } catch (_: Exception) {}
    }

    companion object {
        const val ACTION_ACK = "app.fallguard.ACK"
        const val ACTION_FALSE = "app.fallguard.FALSE"
        const val ACTION_SILENCE = "app.fallguard.SILENCE"
        const val ACTION_NEARBY = "app.fallguard.NEARBY"
        private val NEARBY_KINDS = setOf(Signals.Kind.STANDUP, Signals.Kind.SITTING, Signals.Kind.HELP)
        const val ACTION_TEST = "app.fallguard.TEST"
        const val ACTION_TEST_SIGNAL = "app.fallguard.TEST_SIGNAL"
        const val EXTRA_ID = "alert_id"
        private const val CH_ALARM = "alarm_v1"
        private const val CH_EVENTS = "events_v2"
        private const val CH_LINK = "link_v1"
        private const val NOTIF_LINK = 1
        private const val NOTIF_ALARM = 2
        private const val NOTIF_EVENT = 3
        private const val NOTIF_LOST = 4
        private const val NOTIF_UPDATE = 5
        private const val LOST_AFTER_MS = 120_000L

        @Volatile var current: AlarmInfo? = null
        @Volatile var onAlarmStopped: (() -> Unit)? = null

        fun start(ctx: Context, action: String? = null, alertId: String = "") {
            val i = Intent(ctx, AlertService::class.java)
            if (action != null) i.setAction(action).putExtra(EXTRA_ID, alertId)
            try { ContextCompat.startForegroundService(ctx, i) } catch (_: Exception) {}
        }
    }
}
