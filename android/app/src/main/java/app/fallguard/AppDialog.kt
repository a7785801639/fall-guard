package app.fallguard

import android.app.Activity
import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.Window
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** Окна в стиле приложения (вместо серого стандартного AlertDialog): карточка, шрифт и цвета пульта. */
object AppDialog {
    data class Action(val title: String, val style: Int, val onClick: () -> Unit)

    fun content(ctx: Context, title: String, message: String?, body: View?, actions: List<Action>, dismiss: () -> Unit): View {
        fun dp(v: Int) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), ctx.resources.displayMetrics).toInt()
        val card = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.dialog_bg)
            setPadding(0, dp(22), 0, dp(16))
        }
        card.addView(TextView(ctx).apply {
            text = title; textSize = 21f; setTextColor(ctx.getColor(R.color.text))
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(dp(22), 0, dp(22), 0)
        })
        if (!message.isNullOrBlank()) card.addView(TextView(ctx).apply {
            text = message; textSize = 15f; setTextColor(ctx.getColor(R.color.muted)); setLineSpacing(0f, 1.15f)
            setPadding(dp(22), dp(8), dp(22), 0)
        })
        if (body != null) card.addView(ScrollView(ctx).apply {
            isVerticalScrollBarEnabled = false
            addView(body)
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
        })
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(16), dp(14), dp(16), 0)
        }
        actions.forEachIndexed { i, a ->
            row.addView(Button(ctx, null, 0, a.style).apply {
                text = a.title; isAllCaps = false; gravity = Gravity.CENTER
                setOnClickListener { dismiss(); a.onClick() }
                layoutParams = LinearLayout.LayoutParams(0, dp(52), 1f).apply { if (i > 0) marginStart = dp(10) }
            })
        }
        card.addView(row)
        return card
    }

    fun show(act: Activity, title: String, message: String? = null, body: View? = null, actions: List<Action>, onDismiss: () -> Unit = {}): Dialog {
        val d = Dialog(act)
        d.requestWindowFeature(Window.FEATURE_NO_TITLE)
        d.setContentView(content(act, title, message, body, actions) { d.dismiss() })
        d.window?.apply {
            setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            val w = (act.resources.displayMetrics.widthPixels * 0.92f).toInt().coerceAtMost((440 * act.resources.displayMetrics.density).toInt())
            setLayout(w, if (body != null) (act.resources.displayMetrics.heightPixels * 0.82f).toInt() else ViewGroup.LayoutParams.WRAP_CONTENT)
        }
        d.setOnDismissListener { onDismiss() }
        d.show()
        return d
    }
}
