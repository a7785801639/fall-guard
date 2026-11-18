package app.fallguard

import app.fallguard.Signals.Action
import app.fallguard.Signals.Alert
import app.fallguard.Signals.Kind
import app.fallguard.Signals.Style
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SignalsTest {
    private val fall = Alert("f", "fallen", "critical", true)
    private val stand = Alert("s", "left_bed", "warning", false)
    private val sit = Alert("t", "sitting_up", "warning", false)
    private val absence = Alert("a", "left_bed", "critical", false)
    private val defaults: (Kind) -> Style? = { null }

    @Test fun fallAlwaysRings() {
        val acts = Signals.decide(listOf(fall), alarmActive = true, ringing = false, silenced = false) { Style.SILENT }
        assertEquals(listOf<Action>(Action.Ring(fall, Kind.FALL)), acts)
    }

    @Test fun standupIsNotificationByDefaultButCanRing() {
        assertEquals(listOf<Action>(Action.Notify(stand, Kind.STANDUP, true)), Signals.decide(listOf(stand), false, false, false, styleOf = defaults))
        val loud = Signals.decide(listOf(stand), false, false, false) { if (it == Kind.STANDUP) Style.ALARM else null }
        assertEquals(listOf<Action>(Action.Ring(stand, Kind.STANDUP)), loud)
        val quiet = Signals.decide(listOf(stand), false, false, false) { if (it == Kind.STANDUP) Style.SILENT else null }
        assertEquals(listOf<Action>(Action.Notify(stand, Kind.STANDUP, false)), quiet)
    }

    @Test fun criticalNonFallStillRings() {
        val acts = Signals.decide(listOf(absence), true, false, false) { Style.SILENT }
        assertEquals(listOf<Action>(Action.Ring(absence, Kind.OTHER)), acts)
    }

    @Test fun mixedBatchNotifiesAndRingsForTheFall() {
        val acts = Signals.decide(listOf(sit, fall), true, false, false, styleOf = defaults)
        assertEquals(listOf(Action.Notify(sit, Kind.SITTING, true), Action.Ring(fall, Kind.FALL)), acts)
    }

    @Test fun serverClearedAlarmStopsRinging() {
        assertEquals(listOf<Action>(Action.Stop), Signals.decide(emptyList(), alarmActive = false, ringing = true, silenced = false, styleOf = defaults))
        assertTrue(Signals.decide(emptyList(), alarmActive = false, ringing = false, silenced = false, styleOf = defaults).isEmpty())
    }

    @Test fun unacknowledgedOnServerRingsUnlessSilenced() {
        assertEquals(listOf<Action>(Action.RingUnconfirmed), Signals.decide(emptyList(), true, ringing = false, silenced = false, styleOf = defaults))
        assertTrue(Signals.decide(emptyList(), true, ringing = false, silenced = true, styleOf = defaults).isEmpty())
        assertTrue(Signals.decide(emptyList(), true, ringing = true, silenced = false, styleOf = defaults).isEmpty())
    }

    @Test fun newFallAfterSilenceRingsAgain() {
        val acts = Signals.decide(listOf(fall), alarmActive = true, ringing = false, silenced = true, styleOf = defaults)
        assertEquals(listOf<Action>(Action.Ring(fall, Kind.FALL)), acts)
    }

    @Test fun locallyEscalatedAlarmIsNotStoppedByServer() {
        // «Встала» настроена как тревога: на следующем опросе сервер не знает о ней — звонок продолжается
        assertTrue(Signals.decide(emptyList(), alarmActive = false, ringing = true, silenced = false, currentServerTracked = false, styleOf = defaults).isEmpty())
        // а тревогу о падении сервер снимает
        assertEquals(listOf<Action>(Action.Stop), Signals.decide(emptyList(), alarmActive = false, ringing = true, silenced = false, currentServerTracked = true, styleOf = defaults))
        assertTrue(Signals.serverTracked(fall) && Signals.serverTracked(absence) && !Signals.serverTracked(stand))
    }

    @Test fun kindMapping() {
        assertEquals(Kind.HELP, Signals.kindOf("help_call", "warning", false))
        assertEquals(Kind.STANDUP, Signals.kindOf("missing", "warning", false))
        assertEquals(Kind.FALL, Signals.kindOf("whatever", "critical", true))
        assertEquals(Kind.OTHER, Signals.kindOf("sitting_up", "critical", false))
    }
}
