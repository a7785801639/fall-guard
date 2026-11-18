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

    /** Способ: громкая тревога на весь экран, уведомление со звуком, уведомление без звука. */
    enum class Style(val title: String) { ALARM("Тревога на весь экран"), NOTIFY("Уведомление со звуком"), SILENT("Без звука") }

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

    /** Падение и любые критические события нельзя приглушить ниже полноэкранной тревоги. */
    fun effectiveStyle(kind: Kind, severity: String, chosen: Style?): Style {
        if (kind == Kind.FALL || severity == "critical") return Style.ALARM
        return chosen ?: defaultStyle(kind, severity)
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
                Style.ALARM -> ring = Action.Ring(a, kind) // последняя тревожная — та, что на экране
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
