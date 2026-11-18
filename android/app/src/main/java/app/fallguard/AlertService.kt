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
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        goForeground()
        if (!prefs.paired) {
            stopSelf()
            return START_NOT_STICKY
        }
        when (intent?.action) {
            ACTION_ACK -> { silencedUntilQuiet = true; stopAlarm(); post("/api/pause", JSONObject().put("seconds", 900)) }
            ACTION_FALSE -> {
                val id = intent.getStringExtra(EXTRA_ID) ?: ""
                silencedUntilQuiet = true
                stopAlarm()
                if (id.isNotEmpty()) post("/api/feedback/$id/false_alarm", JSONObject().put("reporter", Build.MODEL))
                else post("/api/pause", JSONObject().put("seconds", 900))
            }
            ACTION_SILENCE -> { silencedUntilQuiet = true; stopAlarm() }
            ACTION_TEST_SIGNAL -> testSignal(Signals.Kind.entries.firstOrNull { it.name == intent.getStringExtra(EXTRA_ID) } ?: Signals.Kind.FALL)
            ACTION_TEST -> startAlarm(
                AlarmInfo("", "Проверка тревоги", "Так прозвучит сигнал о падении. Это только проверка.", System.currentTimeMillis(), "", ""),
            )
        }
        if (!running) {
            running = true
            worker = Thread({ pollLoop() }, "fallguard-poll").also { it.start() }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        running = false
        worker?.interrupt()
        stopAlarm()
        super.onDestroy()
    }

    // ---------------------------------------------------------------- опрос сервера
    private fun pollLoop() {
        var base: String? = null
        var lastOk = System.currentTimeMillis()
        var lostWarned = false
        var backoff = 2000L
        while (running) {
            if (base == null) base = Api.reachableBase(prefs)
            val res = if (base == null) HttpResult(-1, "") else Api.request(
                base,
                "/api/alerts/poll?after=${prefs.cursor}&wait=${if (ringing) 3 else 25}",
                token = prefs.token,
                connectMs = 5000,
                readMs = 40000,
            )
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
                        notifyEvent(NOTIF_LOST, "Нет связи с FallGuard", "Тревоги сейчас не дойдут до этого телефона. Проверьте интернет и сервер.")
                        if (Sounds.style(this, Signals.Kind.LINK) != Signals.Style.SILENT) Sounds.playNotify(this, Signals.Kind.LINK)
                    }
                    try { Thread.sleep(backoff) } catch (_: InterruptedException) { return }
                    backoff = (backoff * 2).coerceAtMost(15000L)
                }
            }
        }
    }

    /** Проверка одного типа сигнала ровно так, как он придёт: тревогой, уведомлением со звуком или без. */
    private fun testSignal(kind: Signals.Kind) {
        val title = "Проверка: ${kind.title.lowercase()}"
        when (Signals.effectiveStyle(kind, "", Sounds.style(this, kind))) {
            Signals.Style.ALARM -> startAlarm(AlarmInfo("", title, "Так придёт этот сигнал. Это только проверка.", System.currentTimeMillis(), "", "", kind, serverTracked = false))
            Signals.Style.NOTIFY -> { notifyEvent(NOTIF_EVENT, title, "Так придёт уведомление со звуком."); Sounds.playNotify(this, kind) }
            Signals.Style.SILENT -> notifyEvent(NOTIF_EVENT, title, "Так придёт уведомление без звука.")
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
        val cursor = j.optLong("cursor", prefs.cursor)
        if (cursor > prefs.cursor) prefs.cursor = cursor

        val alarmActive = j.optBoolean("alarm_active")
        val actions = Signals.decide(list, alarmActive, ringing, silencedUntilQuiet, current?.serverTracked ?: true, j.optString("state")) { Sounds.style(this, it) }
        for (act in actions) when (act) {
            is Signals.Action.Notify -> {
                val a = byId[act.alert.id] ?: continue
                notifyEvent(NOTIF_EVENT, a.optString("title"), a.optString("description"))
                if (act.sound) Sounds.playNotify(this, act.kind)
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
                val info = AlarmInfo("", "Тревога не подтверждена", j.optString("status_line"), System.currentTimeMillis(), "", base)
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

    private fun post(path: String, body: JSONObject) {
        Thread {
            val base = Api.reachableBase(prefs) ?: return@Thread
            Api.request(base, path, "POST", prefs.token, body)
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
        player = Sounds.play(this, Sounds.uriOf(this, kind), AudioAttributes.USAGE_ALARM, loop = true, fallback = siren)
            ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)?.let { Sounds.play(this, it, AudioAttributes.USAGE_ALARM, loop = true) }
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

    private fun notifyEvent(id: Int, title: String, text: String) {
        val n = NotificationCompat.Builder(this, CH_EVENTS)
            .setSmallIcon(R.drawable.ic_stat)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .setContentIntent(openAppPending())
            .build()
        try { nm.notify(id, n) } catch (_: Exception) {}
    }

    companion object {
        const val ACTION_ACK = "app.fallguard.ACK"
        const val ACTION_FALSE = "app.fallguard.FALSE"
        const val ACTION_SILENCE = "app.fallguard.SILENCE"
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
