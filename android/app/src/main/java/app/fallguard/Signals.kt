package app.fallguard

/**
 * Как телефон сигналит о каждом типе события. Чистая логика без Android — покрыта юнит-тестами.
 */
object Signals {
    /** Типы событий, для которых можно выбрать способ сигнала и звук. */
    enum class Kind(val title: String, val sub: String) {
        FALL("Падение", "всегда громкая тревога"),
        STANDUP("Встала с кровати", "ушла или встала одна"),
        SITTING("Садится", "свесила ноги"),
        HELP("Зовёт рукой", "рука над головой"),
        OTHER("Другие тревоги", "долго нет в кровати и т. п."),
        LINK("Нет связи с сервером", "телефон не получает тревоги"),
    }

    /**
     * Способ: громкая тревога на весь экран; мини-тревога (та же сирена, но карточкой сверху экрана,
     * если телефон в руках; на заблокированном — на весь экран); уведомление со звуком; без звука.
     */
    enum class Style(val title: String) {
        ALARM("Тревога на весь экран"), MINI("Мини-тревога сверху"), NOTIFY("Уведомление со звуком"), SILENT("Без звука");

        /** Звенит сиреной и ждёт «Принято». */
        val rings: Boolean get() = this == ALARM || this == MINI
    }

    /** Какие способы можно выбрать: падение — только тревога (полная или мини), обрыв связи — только уведомление. */
    fun allowedStyles(kind: Kind): List<Style> = when (kind) {
        Kind.FALL -> listOf(Style.ALARM, Style.MINI)
        Kind.LINK -> listOf(Style.NOTIFY, Style.SILENT)
        else -> Style.entries.toList()
    }

    fun kindOf(state: String, severity: String, isFall: Boolean): Kind = when {
        isFall || state == "fallen" -> Kind.FALL
        state == "left_bed" || state == "missing" -> Kind.STANDUP
        state == "sitting_up" -> Kind.SITTING
        state == "help_call" -> Kind.HELP
        else -> Kind.OTHER
    }.let { if (it != Kind.FALL && severity == "critical") Kind.OTHER else it }

    fun defaultStyle(kind: Kind, severity: String = ""): Style = when {
        kind == Kind.FALL -> Style.ALARM
        severity == "critical" -> Style.ALARM
        kind == Kind.OTHER -> Style.ALARM
        else -> Style.NOTIFY
    }

    /** Падение и любые критические события — всегда тревога с сиреной: полная или мини, тише нельзя. */
    fun effectiveStyle(kind: Kind, severity: String, chosen: Style?): Style {
        if (kind == Kind.FALL || severity == "critical") return if (chosen == Style.MINI) Style.MINI else Style.ALARM
        return chosen ?: defaultStyle(kind, severity)
    }

    /** Громкость в процентах: от 5 % у любого сигнала (ноль — это способ «Без звука», а не громкость). */
    const val ALARM_MIN_VOLUME = 5
    const val NOTIFY_MIN_VOLUME = 5

    fun minVolume(style: Style): Int = if (style.rings) ALARM_MIN_VOLUME else NOTIFY_MIN_VOLUME

    fun clampVolume(style: Style, percent: Int): Int = percent.coerceIn(minVolume(style), 100)

    /**
     * Множитель для MediaPlayer. Он линейный по амплитуде, а ухо слышит громкость примерно логарифмически,
     * поэтому проценты идут по квадратичной кривой: 50 % слышно заметно тише, 100 % — полная громкость.
     */
    fun gain(style: Style, percent: Int): Float {
        val f = clampVolume(style, percent) / 100f
        return f * f
    }

    data class Alert(val id: String, val state: String, val severity: String, val isFall: Boolean)

    sealed class Action {
        data class Ring(val alert: Alert, val kind: Kind) : Action()
        data class Notify(val alert: Alert, val kind: Kind, val sound: Boolean) : Action()
        object Stop : Action()
        object RingUnconfirmed : Action()
    }

    /**
     * Что делать после ответа сервера.
     * alarmActive — сервер считает тревогу непринятой; ringing — телефон уже звонит;
     * silenced — звук на этом телефоне выключен до конца текущей тревоги.
     */
    private val RESOLVED = setOf("in_bed", "caregiver", "paused")

    /** Сервер сам держит и снимает только падения и критические тревоги. */
    fun serverTracked(a: Alert): Boolean = a.isFall || a.state == "fallen" || a.severity == "critical"

    fun decide(
        alerts: List<Alert>,
        alarmActive: Boolean,
        ringing: Boolean,
        silenced: Boolean,
        currentServerTracked: Boolean = true,
        serverState: String = "",
        styleOf: (Kind) -> Style?,
    ): List<Action> {
        val out = mutableListOf<Action>()
        var ring: Action.Ring? = null
        for (a in alerts) {
            val kind = kindOf(a.state, a.severity, a.isFall)
            when (effectiveStyle(kind, a.severity, styleOf(kind))) {
                Style.ALARM, Style.MINI -> ring = Action.Ring(a, kind) // последняя тревожная — та, что на экране
                Style.NOTIFY -> out += Action.Notify(a, kind, sound = true)
                Style.SILENT -> out += Action.Notify(a, kind, sound = false)
            }
        }
        when {
            ring != null -> out += ring
            // Тревогу, поднятую телефоном по настройке (например, «встала» → тревога), сервер не знает:
            // её снимает только человек на телефоне (или предохранитель через 15 минут)
            // её снимаем, когда ситуация разрешилась: легла, рядом свой, кто-то нажал «Принято» (пауза)
            !alarmActive -> if (ringing && (currentServerTracked || serverState in RESOLVED)) out += Action.Stop
            !ringing && !silenced -> out += Action.RingUnconfirmed
        }
        return out
    }
}

/**
 * Часы сервера. Время событий приходит с сервера, а часы телефона могут уйти —
 * тогда секундомер тревоги замирал бы на 0:00 или врал. Считаем «сейчас» по серверу.
 */
object ServerClock {
    @Volatile var offsetMs = 0L
        private set

    /** Мелкие колебания (задержка сети) не трогаем, заметное расхождение принимаем сразу. */
    fun nextOffset(current: Long, serverMs: Long, localMs: Long): Long {
        val d = serverMs - localMs
        return if (kotlin.math.abs(d - current) > 2000) d else current
    }

    fun sync(serverSec: Double, localMs: Long = System.currentTimeMillis()) {
        if (serverSec <= 0) return
        offsetMs = nextOffset(offsetMs, (serverSec * 1000).toLong(), localMs)
    }

    fun now(): Long = System.currentTimeMillis() + offsetMs
}

/** Размер кольца секундомера на экране тревоги: сколько места осталось под ним. */
object AlarmLayout {
    const val RING_MAX_DP = 180f
    const val RING_MIN_DP = 84f
    const val RING_MARGIN_DP = 8f

    /** Диаметр кольца в dp или null, если места слишком мало и кольцо лучше убрать. */
    /** Подпись «без ответа» — только в кольце, где ей хватает места. */
    const val LABEL_MIN_DP = 130f

    /** Цифры уменьшаются медленнее кольца, чтобы оставаться читаемыми. */
    fun textScale(ringDp: Float): Float = (ringDp / RING_MAX_DP * 1.1f).coerceAtMost(1f)

    fun ringSize(spareDp: Float): Float? {
        val size = (spareDp - RING_MARGIN_DP).coerceAtMost(RING_MAX_DP)
        return if (size < RING_MIN_DP) null else size
    }
}
