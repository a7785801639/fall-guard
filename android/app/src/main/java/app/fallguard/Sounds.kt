package app.fallguard

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.net.Uri
import java.io.File

/** Звуки и способ сигнала для каждого типа события: встроенные мелодии, системные и свой файл. */
object Sounds {
    data class Sound(val id: String, val title: String)

    val BUILT_IN = listOf(
        Sound("siren", "Сирена"),
        Sound("pulse", "Пульс"),
        Sound("bell", "Колокол"),
        Sound("chime", "Перезвон"),
        Sound("soft", "Мягкий"),
        Sound("system_alarm", "Будильник телефона"),
        Sound("system_notification", "Уведомление телефона"),
    )
    const val CUSTOM = "custom"

    private fun sp(ctx: Context) = ctx.applicationContext.getSharedPreferences("fallguard_signals", Context.MODE_PRIVATE)

    fun defaultSound(kind: Signals.Kind): String = when (kind) {
        Signals.Kind.FALL, Signals.Kind.OTHER -> "siren"
        Signals.Kind.STANDUP -> "pulse"
        Signals.Kind.LINK -> "system_notification"
        else -> "chime"
    }

    fun style(ctx: Context, kind: Signals.Kind): Signals.Style? =
        sp(ctx).getString("style_${kind.name}", null)?.let { s -> Signals.Style.entries.firstOrNull { it.name == s } }

    fun setStyle(ctx: Context, kind: Signals.Kind, style: Signals.Style) = sp(ctx).edit().putString("style_${kind.name}", style.name).apply()

    fun sound(ctx: Context, kind: Signals.Kind): String {
        val s = sp(ctx).getString("sound_${kind.name}", null) ?: defaultSound(kind)
        return if (s == CUSTOM && !customFile(ctx, kind).exists()) defaultSound(kind) else s
    }

    fun setSound(ctx: Context, kind: Signals.Kind, id: String) = sp(ctx).edit().putString("sound_${kind.name}", id).apply()

    fun customFile(ctx: Context, kind: Signals.Kind) = File(File(ctx.filesDir, "sounds").apply { mkdirs() }, "${kind.name.lowercase()}.audio")

    fun customName(ctx: Context, kind: Signals.Kind): String = sp(ctx).getString("custom_name_${kind.name}", "Свой файл") ?: "Свой файл"

    fun title(ctx: Context, kind: Signals.Kind): String {
        val id = sound(ctx, kind)
        return if (id == CUSTOM) customName(ctx, kind) else BUILT_IN.firstOrNull { it.id == id }?.title ?: "Сирена"
    }

    /** Копирует выбранный файл в память приложения и проверяет, что его можно проиграть. */
    fun importCustom(ctx: Context, kind: Signals.Kind, uri: Uri, displayName: String): Boolean {
        val tmp = File(customFile(ctx, kind).path + ".tmp")
        return try {
            ctx.contentResolver.openInputStream(uri)?.use { input ->
                if (input.available() > 15 * 1024 * 1024) return false
                tmp.outputStream().use { input.copyTo(it) }
            } ?: return false
            val ok = try { MediaPlayer().apply { setDataSource(tmp.path); prepare(); release() }; true } catch (_: Exception) { false }
            if (!ok) { tmp.delete(); return false }
            tmp.renameTo(customFile(ctx, kind))
            sp(ctx).edit().putString("custom_name_${kind.name}", displayName.take(40)).putString("sound_${kind.name}", CUSTOM).apply()
            true
        } catch (_: Exception) {
            tmp.delete(); false
        }
    }

    fun uriOf(ctx: Context, kind: Signals.Kind, id: String = sound(ctx, kind)): Uri = when (id) {
        "siren" -> res(ctx, R.raw.siren)
        "pulse" -> res(ctx, R.raw.pulse)
        "bell" -> res(ctx, R.raw.bell)
        "chime" -> res(ctx, R.raw.chime)
        "soft" -> res(ctx, R.raw.soft)
        "system_alarm" -> RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM) ?: res(ctx, R.raw.siren)
        "system_notification" -> RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION) ?: res(ctx, R.raw.chime)
        CUSTOM -> Uri.fromFile(customFile(ctx, kind))
        else -> res(ctx, R.raw.siren)
    }

    private fun res(ctx: Context, id: Int) = Uri.parse("android.resource://${ctx.packageName}/$id")

    /**
     * Проигрыватель с подстраховкой: если выбранный звук не открылся или сломался во время игры,
     * сразу играет встроенная сирена (для тревоги) — тревога не должна оказаться беззвучной.
     */
    fun play(ctx: Context, uri: Uri, usage: Int, loop: Boolean, fallback: Uri? = null): MediaPlayer? {
        fun make(u: Uri): MediaPlayer = MediaPlayer().apply {
            setAudioAttributes(AudioAttributes.Builder().setUsage(usage).setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build())
            setDataSource(ctx, u)
            isLooping = loop
            prepare()
        }
        val p = try { make(uri) } catch (_: Exception) { fallback?.let { try { make(it) } catch (_: Exception) { null } } } ?: return null
        if (fallback != null) {
            p.setOnErrorListener { mp, _, _ ->
                try { mp.reset(); mp.setDataSource(ctx, fallback); mp.isLooping = loop; mp.prepare(); mp.start() } catch (_: Exception) {}
                true
            }
        }
        if (!loop) p.setOnCompletionListener { try { it.release() } catch (_: Exception) {} }
        p.start()
        return p
    }

    /** Звук уведомления — только если телефон не в беззвучном режиме. */
    fun playNotify(ctx: Context, kind: Signals.Kind) {
        val am = ctx.getSystemService(AudioManager::class.java)
        if (am != null && am.ringerMode != AudioManager.RINGER_MODE_NORMAL) return
        play(ctx, uriOf(ctx, kind), AudioAttributes.USAGE_NOTIFICATION_EVENT, loop = false)
    }
}
