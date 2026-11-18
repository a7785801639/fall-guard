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
    /** Системный звук хранится как «sys:<uri>», его название — отдельно (список на телефоне может измениться). */
    const val SYS = "sys:"

    data class SystemSound(val id: String, val title: String, val group: String)

    @Volatile private var systemCache: List<SystemSound>? = null

    /** Мелодии телефона: будильники, уведомления, звонки. Читает системную базу — вызывать не из главного потока. */
    fun systemSounds(ctx: Context): List<SystemSound> {
        systemCache?.let { return it }
        val out = LinkedHashMap<String, SystemSound>()
        for ((type, group) in listOf(
            RingtoneManager.TYPE_ALARM to "Будильники телефона",
            RingtoneManager.TYPE_NOTIFICATION to "Уведомления телефона",
            RingtoneManager.TYPE_RINGTONE to "Мелодии звонка",
        )) {
            try {
                val rm = RingtoneManager(ctx.applicationContext).apply { setType(type) }
                rm.cursor?.use { c ->
                    while (c.moveToNext()) {
                        val uri = rm.getRingtoneUri(c.position)?.toString() ?: continue
                        val title = c.getString(RingtoneManager.TITLE_COLUMN_INDEX)?.trim().orEmpty()
                        if (title.isNotEmpty() && uri !in out) out[uri] = SystemSound(SYS + uri, title, group)
                    }
                }
            } catch (_: Exception) {}
        }
        return out.values.toList().also { systemCache = it }
    }

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

    fun setSound(ctx: Context, kind: Signals.Kind, id: String, title: String = "") =
        sp(ctx).edit().putString("sound_${kind.name}", id).putString("sound_title_${kind.name}", title).apply()

    /** Громкость в процентах, как её выбрал человек; перед игрой ограничивается по способу сигнала. */
    fun volume(ctx: Context, kind: Signals.Kind): Int = sp(ctx).getInt("volume_${kind.name}", 100).coerceIn(0, 100)

    fun setVolume(ctx: Context, kind: Signals.Kind, percent: Int) = sp(ctx).edit().putInt("volume_${kind.name}", percent.coerceIn(0, 100)).apply()

    fun gain(ctx: Context, kind: Signals.Kind): Float =
        Signals.gain(Signals.effectiveStyle(kind, "", style(ctx, kind)), volume(ctx, kind))

    /** Сирена звучит всегда как тревога — и когда событие стало критическим при выбранном «Со звуком»: нижняя граница тревоги. */
    fun alarmGain(ctx: Context, kind: Signals.Kind): Float = Signals.gain(Signals.Style.ALARM, volume(ctx, kind))

    fun customFile(ctx: Context, kind: Signals.Kind) = File(File(ctx.filesDir, "sounds").apply { mkdirs() }, "${kind.name.lowercase()}.audio")

    fun customName(ctx: Context, kind: Signals.Kind): String = sp(ctx).getString("custom_name_${kind.name}", "Свой файл") ?: "Свой файл"

    fun title(ctx: Context, kind: Signals.Kind): String {
        val id = sound(ctx, kind)
        if (id == CUSTOM) return customName(ctx, kind)
        if (id.startsWith(SYS)) return sp(ctx).getString("sound_title_${kind.name}", null)?.takeIf { it.isNotEmpty() } ?: "Звук телефона"
        return BUILT_IN.firstOrNull { it.id == id }?.title ?: "Сирена"
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

    /** Файл, выбранный в системном окне: имя берётся из него же. Возвращает имя или null, если файл не проигрывается. */
    fun importPicked(ctx: Context, kind: Signals.Kind, uri: Uri): String? {
        var name = "Свой файл"
        try {
            ctx.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) name = c.getString(0).substringBeforeLast('.')
            }
        } catch (_: Exception) {}
        return if (importCustom(ctx, kind, uri, name)) name else null
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
        else -> if (id.startsWith(SYS)) Uri.parse(id.removePrefix(SYS)) else res(ctx, R.raw.siren)
    }

    private fun res(ctx: Context, id: Int) = Uri.parse("android.resource://${ctx.packageName}/$id")

    /**
     * Проигрыватель с подстраховкой: если выбранный звук не открылся или сломался во время игры,
     * сразу играет встроенная сирена (для тревоги) — тревога не должна оказаться беззвучной.
     */
    fun play(ctx: Context, uri: Uri, usage: Int, loop: Boolean, fallback: Uri? = null, gain: Float = 1f): MediaPlayer? {
        fun make(u: Uri): MediaPlayer = MediaPlayer().apply {
            setAudioAttributes(AudioAttributes.Builder().setUsage(usage).setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build())
            setDataSource(ctx, u)
            isLooping = loop
            setVolume(gain, gain)
            prepare()
        }
        val p = try { make(uri) } catch (_: Exception) { fallback?.let { try { make(it) } catch (_: Exception) { null } } } ?: return null
        if (fallback != null) {
            p.setOnErrorListener { mp, _, _ ->
                try { mp.reset(); mp.setDataSource(ctx, fallback); mp.isLooping = loop; mp.setVolume(gain, gain); mp.prepare(); mp.start() } catch (_: Exception) {}
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
        // Выбранный звук пропал (удалён из телефона) — играет встроенный, а не тишина
        play(ctx, uriOf(ctx, kind), AudioAttributes.USAGE_NOTIFICATION_EVENT, loop = false,
            fallback = uriOf(ctx, kind, defaultSound(kind)), gain = gain(ctx, kind))
    }
}
