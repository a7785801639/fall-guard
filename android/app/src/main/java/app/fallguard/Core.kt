package app.fallguard

import android.content.Context
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/** Локальные настройки: адреса сервера, токен устройства, курсор ленты тревог. */
class Prefs(ctx: Context) {
    private val sp = ctx.applicationContext.getSharedPreferences("fallguard", Context.MODE_PRIVATE)

    var token: String
        get() = sp.getString("token", "") ?: ""
        set(v) = sp.edit().putString("token", v).apply()
    var deviceId: String
        get() = sp.getString("device_id", "") ?: ""
        set(v) = sp.edit().putString("device_id", v).apply()
    var lanUrl: String
        get() = sp.getString("lan_url", "") ?: ""
        set(v) = sp.edit().putString("lan_url", v).apply()
    var publicUrl: String
        get() = sp.getString("public_url", "") ?: ""
        set(v) = sp.edit().putString("public_url", v).apply()
    var cursor: Long
        get() = sp.getLong("cursor", 0L)
        set(v) = sp.edit().putLong("cursor", v).apply()

    var keepScreenOn: Boolean
        get() = sp.getBoolean("keep_screen_on", false)
        set(v) = sp.edit().putBoolean("keep_screen_on", v).apply()
    /** Видео в маленьком окне поверх других приложений, когда пульт сворачивают. */
    var pipEnabled: Boolean
        get() = sp.getBoolean("pip_enabled", true)
        set(v) = sp.edit().putBoolean("pip_enabled", v).apply()
    var onboardingShown: Boolean
        get() = sp.getBoolean("onboarding_shown", false)
        set(v) = sp.edit().putBoolean("onboarding_shown", v).apply()

    val paired: Boolean get() = token.isNotEmpty() && urls.isNotEmpty()

    /** Домашний адрес пробуем первым: дома он быстрее и работает без интернета. */
    val urls: List<String> get() = listOf(lanUrl, publicUrl).filter { it.isNotEmpty() }.distinct()

    fun clear() = sp.edit().clear().apply()
}

class HttpResult(val code: Int, val body: String) {
    val ok: Boolean get() = code in 200..299
    fun json(): JSONObject = try { JSONObject(body) } catch (_: Exception) { JSONObject() }
}

/** Готов ли телефон к громкой тревоге: уведомления, фон, окно поверх блокировки. */
object Readiness {
    /** overlay — по желанию: мини-тревога сверху экрана; в «готово» не входит. */
    data class State(val notifications: Boolean, val battery: Boolean, val fullScreen: Boolean, val overlay: Boolean = false) {
        val allOk: Boolean get() = notifications && battery && fullScreen
    }

    fun alarmChannelMuted(ctx: Context): Boolean = try {
        val nm = ctx.getSystemService(android.app.NotificationManager::class.java)
        nm.areNotificationsEnabled() && (nm.getNotificationChannel("alarm_v1")?.importance ?: android.app.NotificationManager.IMPORTANCE_HIGH) < android.app.NotificationManager.IMPORTANCE_HIGH
    } catch (_: Exception) { false }

    /** Ведёт прямо туда, где проблема: в настройки канала «Тревоги» или всех уведомлений приложения. */
    fun openNotificationSettings(ctx: Context) {
        val i = if (alarmChannelMuted(ctx)) android.content.Intent(android.provider.Settings.ACTION_CHANNEL_NOTIFICATION_SETTINGS)
            .putExtra(android.provider.Settings.EXTRA_APP_PACKAGE, ctx.packageName).putExtra(android.provider.Settings.EXTRA_CHANNEL_ID, "alarm_v1")
        else android.content.Intent(android.provider.Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(android.provider.Settings.EXTRA_APP_PACKAGE, ctx.packageName)
        try { ctx.startActivity(i.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)) } catch (_: Exception) {}
    }

    fun check(ctx: Context): State {
        val nm = ctx.getSystemService(android.app.NotificationManager::class.java)
        val pm = ctx.getSystemService(android.os.PowerManager::class.java)
        // Канал «Тревоги» можно вручную приглушить в настройках Android — тогда тревога придёт тихим уведомлением
        val alarmChannel = try { nm.getNotificationChannel("alarm_v1") } catch (_: Exception) { null }
        val channelOk = alarmChannel == null || alarmChannel.importance >= android.app.NotificationManager.IMPORTANCE_HIGH
        return State(
            notifications = nm.areNotificationsEnabled() && channelOk,
            battery = pm.isIgnoringBatteryOptimizations(ctx.packageName),
            fullScreen = android.os.Build.VERSION.SDK_INT < 34 || nm.canUseFullScreenIntent(),
            overlay = AlarmOverlay.allowed(ctx),
        )
    }
}

object Device {
    /** Убирает телефон из списка получателей на сервере, останавливает службу и стирает привязку. */
    fun unpair(ctx: Context, prefs: Prefs) {
        val token = prefs.token
        val id = prefs.deviceId
        val urls = prefs.urls
        Thread {
            if (id.isNotEmpty()) {
                for (base in urls) if (Api.request(base, "/api/devices/$id", "DELETE", token).ok) break
            }
        }.start()
        ctx.stopService(android.content.Intent(ctx, AlertService::class.java))
        prefs.clear()
    }
}

object Api {
    fun normalizeUrl(raw: String): String {
        var s = raw.trim().trimEnd('/')
        if (s.isEmpty()) return ""
        if (!s.startsWith("http://") && !s.startsWith("https://")) {
            val host = s.substringBefore('/').substringBefore(':')
            val local = host.matches(Regex("""\d+\.\d+\.\d+\.\d+""")) || !host.contains('.') ||
                host.endsWith(".local") || host.endsWith(".lan")
            s = (if (local) "http://" else "https://") + s
        }
        return s
    }

    fun request(
        base: String,
        path: String,
        method: String = "GET",
        token: String = "",
        body: JSONObject? = null,
        connectMs: Int = 4000,
        readMs: Int = 10000,
        /** Отдаёт соединение наружу: его можно оборвать из другого потока (смена сети). */
        onOpen: ((HttpURLConnection) -> Unit)? = null,
    ): HttpResult {
        var conn: HttpURLConnection? = null
        return try {
            conn = URL(base + path).openConnection() as HttpURLConnection
            onOpen?.invoke(conn)
            conn.requestMethod = method
            conn.connectTimeout = connectMs
            conn.readTimeout = readMs
            conn.setRequestProperty("Accept", "application/json")
            if (token.isNotEmpty()) conn.setRequestProperty("Authorization", "Bearer $token")
            if (body != null) {
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json")
                conn.outputStream.use { it.write(body.toString().toByteArray()) }
            }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            HttpResult(code, stream?.bufferedReader()?.use { it.readText() } ?: "")
        } catch (e: Exception) {
            HttpResult(-1, e.message ?: "network error")
        } finally {
            conn?.disconnect()
        }
    }

    fun bytes(base: String, path: String, token: String): ByteArray? {
        var conn: HttpURLConnection? = null
        return try {
            conn = URL(base + path).openConnection() as HttpURLConnection
            conn.connectTimeout = 4000
            conn.readTimeout = 8000
            if (token.isNotEmpty()) conn.setRequestProperty("Authorization", "Bearer $token")
            if (conn.responseCode == 200) conn.inputStream.use { it.readBytes() } else null
        } catch (_: Exception) {
            null
        } finally {
            conn?.disconnect()
        }
    }

    /** Первый отвечающий адрес сервера (домашний, затем внешний) или null. */
    fun reachableBase(prefs: Prefs): String? =
        prefs.urls.firstOrNull { request(it, "/api/health", connectMs = 2500, readMs = 3000).ok }
}
