package app.fallguard

import android.app.PendingIntent
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.graphics.drawable.Icon
import android.os.Build
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Что сейчас с бабушкой — коротко, для «взгляда»: постоянное уведомление, виджет, плитка в шторке.
 * Последнее известное состояние хранится локально, чтобы виджет рисовался и без сети.
 */
object Glance {
    enum class Tone(val color: Int) { CALM(0xFF5BE0A1.toInt()), CARE(0xFF7EA8FF.toInt()), WARN(0xFFFFB547.toInt()), ALARM(0xFFE5383F.toInt()), OFF(0xFF959CA9.toInt()) }

    private val TITLES = mapOf(
        "idle" to ("Бабушку пока не видно" to Tone.CALM),
        "in_bed" to ("Бабушка в кровати" to Tone.CALM),
        "caregiver" to ("Бабушка под присмотром" to Tone.CARE),
        "paused" to ("Охрана на паузе" to Tone.OFF),
        "help_call" to ("Бабушка зовёт на помощь" to Tone.WARN),
        "sitting_up" to ("Бабушка садится" to Tone.WARN),
        "left_bed" to ("Бабушка встала одна" to Tone.WARN),
        "missing" to ("Бабушка ушла из кадра" to Tone.WARN),
        "fallen" to ("Падение!" to Tone.ALARM),
    )

    data class Snap(val title: String, val tone: Tone, val sub: String, val ts: Long, val online: Boolean)

    fun describe(state: String, alarm: Boolean): Pair<String, Tone> {
        val (t, tone) = TITLES[state] ?: ("Охрана на связи" to Tone.CALM)
        return if (alarm && state != "fallen") "Тревога не подтверждена" to Tone.ALARM else t to tone
    }

    private fun sp(ctx: Context) = ctx.applicationContext.getSharedPreferences("fallguard_glance", Context.MODE_PRIVATE)

    fun load(ctx: Context): Snap {
        val p = sp(ctx)
        val online = p.getBoolean("online", false)
        val ts = p.getLong("ts", 0L)
        if (!Prefs(ctx).paired) return Snap("Телефон не подключён", Tone.OFF, "Откройте FallGuard", 0L, false)
        if (ts == 0L) return Snap("FallGuard", Tone.OFF, "Ждём связи с сервером", 0L, false)
        val tone = try { Tone.valueOf(p.getString("tone", "CALM")!!) } catch (_: Exception) { Tone.CALM }
        return Snap(p.getString("title", "")!!, if (online) tone else Tone.OFF, p.getString("sub", "")!!, ts, online)
    }

    /** Вызывается сервисом после каждого ответа сервера или при потере связи. */
    fun update(ctx: Context, state: String?, alarm: Boolean, sub: String, online: Boolean) {
        val p = sp(ctx)
        val prev = load(ctx)
        // Батарея: на диск пишем, только если что-то поменялось или «обновлено в …» устарело на 5 минут
        val nowMs = System.currentTimeMillis()
        val desc = state?.let { describe(it, alarm) }
        val changed = online != prev.online || (desc != null && (desc.first != prev.title || desc.second != prev.tone || sub != prev.sub))
        if (!changed && (desc == null || nowMs - prev.ts < 5 * 60_000L)) return
        val e = p.edit().putBoolean("online", online)
        if (desc != null) e.putString("title", desc.first).putString("tone", desc.second.name).putString("sub", sub).putLong("ts", nowMs)
        e.apply()
        val now = load(ctx)
        // Плитка в шторке перечитывает состояние, только если что-то видимое поменялось
        if (now.title != prev.title || now.tone != prev.tone || now.online != prev.online || now.sub != prev.sub || now.ts - prev.ts > 60_000) {
            if (Build.VERSION.SDK_INT >= 24) {
                try { TileService.requestListeningState(ctx, ComponentName(ctx, StatusTile::class.java)) } catch (_: Exception) {}
            }
        }
    }

    fun clock(ts: Long): String = if (ts <= 0) "" else SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(ts))

    fun openApp(ctx: Context): PendingIntent = PendingIntent.getActivity(
        ctx, 7, Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )
}

/** Плитка в шторке быстрых настроек: состояние и вход в пульт одним касанием. */
class StatusTile : TileService() {
    override fun onStartListening() {
        super.onStartListening()
        val s = Glance.load(this)
        val t = qsTile ?: return
        t.label = if (s.tone == Glance.Tone.ALARM) "Тревога!" else "FallGuard"
        if (Build.VERSION.SDK_INT >= 29) t.subtitle = if (!s.online && s.ts > 0) "нет связи" else s.title.removePrefix("Бабушка ")
        t.state = when {
            s.ts == 0L -> Tile.STATE_UNAVAILABLE
            !s.online -> Tile.STATE_INACTIVE
            else -> Tile.STATE_ACTIVE
        }
        // Значок по состоянию: щит — спокойно, человек — встаёт или ушла, знак — тревога
        t.icon = Icon.createWithResource(this, when {
            !s.online -> R.drawable.ic_k_link
            s.tone == Glance.Tone.ALARM -> R.drawable.ic_k_alert
            s.tone == Glance.Tone.WARN -> R.drawable.ic_k_walk
            else -> R.drawable.ic_stat
        })
        t.contentDescription = s.title
        t.updateTile()
    }

    @Suppress("DEPRECATION")
    @android.annotation.SuppressLint("StartActivityAndCollapseDeprecated")
    override fun onClick() {
        super.onClick()
        val i = Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        if (Build.VERSION.SDK_INT >= 34) startActivityAndCollapse(Glance.openApp(this))
        else startActivityAndCollapse(i)
    }
}
