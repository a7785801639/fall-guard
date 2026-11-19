package app.fallguard

import android.content.Context
import android.util.AttributeSet
import android.view.View
import android.view.ViewGroup

/**
 * Вписывает единственный дочерний блок в своё место с заданными пропорциями — целиком,
 * без обрезки и без пустых полос внутри кадра — по центру или прижав к верху. Пропорции берутся из кадра камеры.
 */
class FitBox @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : ViewGroup(ctx, attrs) {
    /** Прижать кадр к верху (под текст), а не ставить по центру свободного места. */
    var alignTop = attrs?.let { ctx.obtainStyledAttributes(it, R.styleable.FitBox).run { getBoolean(R.styleable.FitBox_alignTop, false).also { recycle() } } } ?: false

    /** Ширина / высота кадра; по умолчанию 16:9. */
    var aspect = 16f / 9f
        set(v) {
            if (v <= 0f || kotlin.math.abs(v - field) < 0.01f) return
            field = v; requestLayout()
        }

    override fun onMeasure(widthSpec: Int, heightSpec: Int) {
        val w = MeasureSpec.getSize(widthSpec)
        val h = MeasureSpec.getSize(heightSpec)
        val child = if (childCount > 0) getChildAt(0) else null
        val (cw, ch) = fit(w - paddingLeft - paddingRight, h - paddingTop - paddingBottom, aspect)
        child?.measure(MeasureSpec.makeMeasureSpec(cw, MeasureSpec.EXACTLY), MeasureSpec.makeMeasureSpec(ch, MeasureSpec.EXACTLY))
        setMeasuredDimension(w, if (MeasureSpec.getMode(heightSpec) == MeasureSpec.EXACTLY) h else ch + paddingTop + paddingBottom)
    }

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        val child = if (childCount > 0) getChildAt(0) else return
        if (child.visibility == View.GONE) return
        val x = paddingLeft + (width - paddingLeft - paddingRight - child.measuredWidth) / 2
        val y = paddingTop + if (alignTop) 0 else (height - paddingTop - paddingBottom - child.measuredHeight) / 2
        child.layout(x, y, x + child.measuredWidth, y + child.measuredHeight)
    }

    companion object {
        /** Наибольший прямоугольник с пропорциями [aspect], помещающийся в w×h. */
        fun fit(w: Int, h: Int, aspect: Float): Pair<Int, Int> {
            if (w <= 0) return 0 to 0
            if (h <= 0) return w to (w / aspect).toInt() // высота не ограничена
            val byWidth = (w / aspect).toInt()
            return if (byWidth <= h) w to byWidth else (h * aspect).toInt() to h
        }
    }
}
