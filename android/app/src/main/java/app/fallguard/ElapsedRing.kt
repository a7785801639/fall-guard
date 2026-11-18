package app.fallguard

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.sin

/**
 * Секундомер тревоги: дуга плавно обегает круг за минуту, на конце — светящаяся точка.
 * Через минуту без ответа становится янтарной, через три — красной.
 */
class ElapsedRing @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    var startTs = 0L
        set(v) { field = v; invalidate() }

    private val dp = resources.displayMetrics.density
    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = 0x26FFFFFF }
    private val arc = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND }
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val halo = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val rect = RectF()

    /** Плавная анимация, только если в системе не отключены анимации. */
    private val smooth: Boolean get() = ValueAnimator.areAnimatorsEnabled()

    override fun onAttachedToWindow() { super.onAttachedToWindow(); invalidate() }

    fun colorFor(sec: Long): Int = when {
        sec >= 180 -> 0xFFFF6B70.toInt()
        sec >= 60 -> 0xFFFFD166.toInt()
        else -> 0xFFFFFFFF.toInt()
    }

    override fun onDraw(c: Canvas) {
        val w = (minOf(width, height) * 0.055f).coerceIn(2.5f * dp, 7f * dp)
        track.strokeWidth = w; arc.strokeWidth = w
        val pad = w * 1.6f
        val size = minOf(width, height) - pad * 2
        val l = (width - size) / 2f; val t = (height - size) / 2f
        rect.set(l, t, l + size, t + size)
        c.drawOval(rect, track)

        val ms = (ServerClock.now() - startTs).coerceAtLeast(0)
        val frac = if (smooth) (ms % 60_000) / 60_000f else (ms / 1000 % 60) / 60f
        val color = colorFor(ms / 1000)
        arc.color = color; dot.color = color
        halo.color = (color and 0x00FFFFFF) or 0x40000000
        c.drawArc(rect, -90f, 360f * frac, false, arc)
        val a = Math.toRadians(-90.0 + 360.0 * frac)
        val x = rect.centerX() + rect.width() / 2 * cos(a).toFloat()
        val y = rect.centerY() + rect.height() / 2 * sin(a).toFloat()
        c.drawCircle(x, y, w * 1.6f, halo)
        c.drawCircle(x, y, w * 0.85f, dot)
        // ~25 кадров в секунду: глазу плавно, а тревога может висеть минутами — батарею бережём
        if (smooth && isShown) postInvalidateDelayed(40)
    }
}
