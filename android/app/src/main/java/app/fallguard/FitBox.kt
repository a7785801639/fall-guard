package app.fallguard

import android.content.Context
import android.util.AttributeSet
import android.view.View
import android.view.ViewGroup

/**
 * Кадр с камеры целиком — без обрезки и без пустых полос внутри, в пропорциях камеры.
 *
 * Один дочерний блок — вписывается в своё место по центру.
 * Два — первый (кадр) прижат к верху и занимает сколько может, оставив место второму;
 * второй стоит по центру того, что осталось под кадром.
 */
class FitBox @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : ViewGroup(ctx, attrs) {
    /** Ширина / высота кадра; по умолчанию 16:9. */
    var aspect = 16f / 9f
        set(v) {
            if (v <= 0f || kotlin.math.abs(v - field) < 0.01f) return
            field = v; requestLayout()
        }

    private val below: View? get() = if (childCount > 1) getChildAt(1).takeIf { it.visibility != View.GONE } else null

    override fun onMeasure(widthSpec: Int, heightSpec: Int) {
        val w = MeasureSpec.getSize(widthSpec)
        val h = MeasureSpec.getSize(heightSpec)
        val iw = w - paddingLeft - paddingRight
        val ih = h - paddingTop - paddingBottom
        var rest = ih
        below?.let {
            it.measure(MeasureSpec.makeMeasureSpec(iw, MeasureSpec.EXACTLY), MeasureSpec.makeMeasureSpec(ih.coerceAtLeast(0), MeasureSpec.AT_MOST))
            rest = (ih - it.measuredHeight).coerceAtLeast(0)
        }
        val (cw, ch) = fit(iw, if (MeasureSpec.getMode(heightSpec) == MeasureSpec.UNSPECIFIED) -1 else rest, aspect)
        if (childCount > 0) getChildAt(0).measure(MeasureSpec.makeMeasureSpec(cw, MeasureSpec.EXACTLY), MeasureSpec.makeMeasureSpec(ch, MeasureSpec.EXACTLY))
        val used = ch + (below?.measuredHeight ?: 0) + paddingTop + paddingBottom
        setMeasuredDimension(w, if (MeasureSpec.getMode(heightSpec) == MeasureSpec.EXACTLY) h else used)
    }

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        if (childCount == 0) return
        val frame = getChildAt(0)
        val iw = width - paddingLeft - paddingRight
        val ih = height - paddingTop - paddingBottom
        val x = paddingLeft + (iw - frame.measuredWidth) / 2
        val rest = below
        if (rest == null) {
            val y = paddingTop + (ih - frame.measuredHeight) / 2
            frame.layout(x, y, x + frame.measuredWidth, y + frame.measuredHeight)
            return
        }
        frame.layout(x, paddingTop, x + frame.measuredWidth, paddingTop + frame.measuredHeight)
        val top = paddingTop + frame.measuredHeight
        val y = top + ((height - paddingBottom - top - rest.measuredHeight) / 2).coerceAtLeast(0)
        rest.layout(paddingLeft, y, paddingLeft + rest.measuredWidth, y + rest.measuredHeight)
    }

    companion object {
        /** Наибольший прямоугольник с пропорциями [aspect], помещающийся в w×h. */
        fun fit(w: Int, h: Int, aspect: Float): Pair<Int, Int> {
            if (w <= 0) return 0 to 0
            if (h < 0) return w to (w / aspect).toInt() // высота не ограничена
            val byWidth = (w / aspect).toInt()
            return if (byWidth <= h) w to byWidth else (h * aspect).toInt() to h
        }
    }
}
