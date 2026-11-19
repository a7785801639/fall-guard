package app.fallguard

import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.animation.DecelerateInterpolator
import android.view.animation.OvershootInterpolator
import android.widget.ImageView
import android.widget.TextView
import kotlin.math.abs

/**
 * Мини-тревога сверху экрана поверх всех приложений — когда телефон в руках и разблокирован.
 * Карточка выезжает сверху; смахивание вверх сворачивает её в «пилюлю», касание пилюли разворачивает,
 * касание карточки открывает полноэкранную тревогу с живым видео.
 */
class AlarmOverlay(private val ctx: Context) {
    private val ui = Handler(Looper.getMainLooper())
    private val wm = ctx.getSystemService(WindowManager::class.java)
    private val d = ctx.resources.displayMetrics.density
    private var root: View? = null
    private var info: AlarmInfo? = null
    private var collapsed = false
    private var closing = false
    private val anims = mutableListOf<android.animation.Animator>()

    val showing: Boolean get() = root != null

    companion object {
        fun allowed(ctx: Context): Boolean = Build.VERSION.SDK_INT < 23 || Settings.canDrawOverlays(ctx)
    }

    private val tick = object : Runnable {
        override fun run() {
            val i = info ?: return
            val sec = ((ServerClock.now() - i.ts) / 1000).coerceAtLeast(0)
            val t = if (sec >= 3600) "%d:%02d:%02d".format(sec / 3600, sec / 60 % 60, sec % 60) else "%d:%02d".format(sec / 60, sec % 60)
            root?.findViewById<TextView>(R.id.ovEyebrow)?.text = if (i.base.isEmpty()) "ПРОВЕРКА · $t" else "ТРЕВОГА · $t"
            root?.findViewById<TextView>(R.id.ovPillText)?.text = "Тревога · $t"
            ui.postDelayed(this, 1000)
        }
    }

    /** Показать (или обновить, если уже на экране). false — нет разрешения или не получилось. */
    @SuppressLint("ClickableViewAccessibility", "InflateParams")
    fun show(i: AlarmInfo): Boolean {
        if (!allowed(ctx)) return false
        info = i
        closing = false
        if (root == null) {
            val themed = android.view.ContextThemeWrapper(ctx, R.style.Theme_FallGuard)
            val v = LayoutInflater.from(themed).inflate(R.layout.overlay_alarm, null)
            val type = if (Build.VERSION.SDK_INT >= 26) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY else @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE
            val lp = WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.WRAP_CONTENT, type,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT,
            ).apply {
                gravity = Gravity.TOP
                if (Build.VERSION.SDK_INT >= 28) layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
            // Под строкой состояния и вырезом камеры: отступ — по настоящим системным отступам
            val sb = ctx.resources.getIdentifier("status_bar_height", "dimen", "android").let { if (it > 0) ctx.resources.getDimensionPixelSize(it) else (24 * d).toInt() }
            val baseL = v.paddingLeft; val baseR = v.paddingRight
            v.setPadding(baseL, sb + (8 * d).toInt(), baseR, v.paddingBottom)
            v.setOnApplyWindowInsetsListener { view, ins ->
                val c = androidx.core.view.WindowInsetsCompat.toWindowInsetsCompat(ins, view)
                    .getInsets(androidx.core.view.WindowInsetsCompat.Type.statusBars() or androidx.core.view.WindowInsetsCompat.Type.displayCutout())
                view.setPadding(baseL + c.left, maxOf(sb, c.top) + (8 * d).toInt(), baseR + c.right, view.paddingBottom)
                ins
            }
            try { wm.addView(v, lp) } catch (_: Exception) { return false }
            root = v
            wire(v)
            enter(v)
        }
        bind(i)
        ui.removeCallbacks(tick); ui.post(tick)
        return true
    }

    private fun bind(i: AlarmInfo) {
        val v = root ?: return
        v.findViewById<TextView>(R.id.ovTitle).text = i.title
        v.findViewById<ImageView>(R.id.ovKind).setImageResource(
            when (i.kind) {
                Signals.Kind.FALL -> R.drawable.ic_k_fall
                Signals.Kind.STANDUP -> R.drawable.ic_k_walk
                Signals.Kind.SITTING -> R.drawable.ic_k_sit
                Signals.Kind.HELP -> R.drawable.ic_k_hand
                Signals.Kind.LINK -> R.drawable.ic_k_link
                Signals.Kind.OTHER -> R.drawable.ic_k_alert
            },
        )
        v.findViewById<View>(R.id.ovAck).contentDescription = if (i.base.isEmpty()) "Выключить проверку" else "Принято, иду"
        val thumb = v.findViewById<ImageView>(R.id.ovThumb)
        thumb.clipToOutline = true
        if (i.imageUrl.isNotEmpty() && i.base.isNotEmpty()) {
            val token = Prefs(ctx).token
            Thread {
                for (attempt in 0 until 4) {
                    val data = Api.bytes(i.base, i.imageUrl, token)
                    val bmp = data?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
                    if (bmp != null) { ui.post { if (info === i) thumb.setImageBitmap(bmp) }; return@Thread }
                    try { Thread.sleep(1200) } catch (_: InterruptedException) { return@Thread }
                }
            }.start()
        } else thumb.setImageResource(R.drawable.ic_shield)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun wire(v: View) {
        val card = v.findViewById<View>(R.id.ovCard)
        val pill = v.findViewById<View>(R.id.ovPill)
        v.findViewById<View>(R.id.ovAck).setOnClickListener {
            val i = info ?: return@setOnClickListener
            AlertService.start(ctx, if (i.base.isEmpty()) AlertService.ACTION_SILENCE else AlertService.ACTION_ACK, i.id)
            finish(true, "Принято", if (i.base.isEmpty()) "Это была проверка" else "Идите к ней — сирена выключена")
        }
        v.findViewById<View>(R.id.ovOpen).setOnClickListener { openFull() }
        pill.setOnClickListener { expand() }
        // Смахивание вверх — свернуть; касание — открыть на весь экран
        var y0 = 0f; var moved = false
        card.setOnTouchListener { view, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> { y0 = e.rawY; moved = false; true }
                MotionEvent.ACTION_MOVE -> {
                    val dy = e.rawY - y0
                    if (abs(dy) > 8 * d) moved = true
                    view.translationY = if (dy < 0) dy else dy * 0.25f
                    view.alpha = (1f + dy / (view.height * 1.4f)).coerceIn(0.3f, 1f)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    val dy = e.rawY - y0
                    if (dy < -36 * d) collapse()
                    else {
                        view.animate().translationY(0f).alpha(1f).setDuration(220).setInterpolator(OvershootInterpolator(1.2f)).start()
                        if (!moved && e.actionMasked == MotionEvent.ACTION_UP) openFull()
                    }
                    true
                }
                else -> false
            }
        }
    }

    private fun enter(v: View) {
        val card = v.findViewById<View>(R.id.ovCard)
        card.translationY = -260 * d; card.alpha = 0f; card.scaleX = 0.94f; card.scaleY = 0.94f
        card.animate().translationY(0f).alpha(1f).scaleX(1f).scaleY(1f).setDuration(460)
            .setInterpolator(OvershootInterpolator(0.9f)).start()
        startPulse(v)
    }

    private fun startPulse(v: View) {
        anims.forEach { it.cancel() }; anims.clear()
        if (!ValueAnimator.areAnimatorsEnabled()) return
        val ring = v.findViewById<View>(R.id.ovRing)
        anims += android.animation.AnimatorSet().apply {
            playTogether(
                ObjectAnimator.ofFloat(ring, View.SCALE_X, 1f, 1.35f).apply { repeatCount = ValueAnimator.INFINITE },
                ObjectAnimator.ofFloat(ring, View.SCALE_Y, 1f, 1.35f).apply { repeatCount = ValueAnimator.INFINITE },
                ObjectAnimator.ofFloat(ring, View.ALPHA, 0.65f, 0f).apply { repeatCount = ValueAnimator.INFINITE },
            )
            duration = 1300; start()
        }
        // Кнопка «Принято» мягко «дышит», как «ответить» на входящем вызове
        val ack = v.findViewById<View>(R.id.ovAck)
        anims += android.animation.AnimatorSet().apply {
            playTogether(
                ObjectAnimator.ofFloat(ack, View.SCALE_X, 1f, 1.08f).apply { repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE },
                ObjectAnimator.ofFloat(ack, View.SCALE_Y, 1f, 1.08f).apply { repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE },
            )
            duration = 650; start()
        }
        for (id in intArrayOf(R.id.ovPillDot, R.id.ovDot)) {
            anims += ObjectAnimator.ofFloat(v.findViewById<View>(id), View.ALPHA, 1f, 0.25f).apply {
                duration = 650; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE; start()
            }
        }
    }

    fun collapse() {
        val v = root ?: return
        if (collapsed) return
        collapsed = true
        val card = v.findViewById<View>(R.id.ovCard)
        val pill = v.findViewById<View>(R.id.ovPill)
        card.animate().translationY(-card.height.toFloat()).alpha(0f).setDuration(220).setInterpolator(DecelerateInterpolator())
            .withEndAction { card.visibility = View.GONE }.start()
        pill.visibility = View.VISIBLE
        pill.scaleX = 0.6f; pill.scaleY = 0.6f; pill.alpha = 0f
        pill.animate().scaleX(1f).scaleY(1f).alpha(1f).setStartDelay(120).setDuration(300).setInterpolator(OvershootInterpolator(1.6f)).start()
    }

    fun expand() {
        val v = root ?: return
        if (!collapsed) return
        collapsed = false
        val card = v.findViewById<View>(R.id.ovCard)
        val pill = v.findViewById<View>(R.id.ovPill)
        pill.animate().alpha(0f).scaleX(0.7f).scaleY(0.7f).setStartDelay(0).setDuration(150).withEndAction { pill.visibility = View.GONE }.start()
        card.visibility = View.VISIBLE
        card.translationY = -card.height.toFloat().coerceAtLeast(120 * d); card.alpha = 0f
        card.animate().translationY(0f).alpha(1f).setDuration(360).setInterpolator(OvershootInterpolator(0.9f)).start()
    }

    private fun openFull() {
        try { ctx.startActivity(Intent(ctx, AlarmActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) } catch (_: Exception) {}
        collapse()
    }

    /** Нажали «Принято» здесь или тревогу сняли на сервере: короткое подтверждение — и карточка уезжает. */
    fun finish(ok: Boolean, title: String, sub: String) {
        val v = root ?: return
        if (closing) return
        closing = true
        ui.removeCallbacks(tick)
        anims.forEach { it.cancel() }; anims.clear()
        if (collapsed) { collapsed = false; v.findViewById<View>(R.id.ovPill).visibility = View.GONE; v.findViewById<View>(R.id.ovCard).apply { visibility = View.VISIBLE; translationY = 0f; alpha = 1f } }
        val card = v.findViewById<View>(R.id.ovCard)
        if (ok) card.setBackgroundResource(R.drawable.overlay_card_ok)
        v.findViewById<View>(R.id.ovActions).visibility = View.GONE
        v.findViewById<View>(R.id.ovRing).alpha = 0f
        v.findViewById<View>(R.id.ovDot).apply { alpha = 1f; backgroundTintList = android.content.res.ColorStateList.valueOf(ctx.getColor(R.color.calm)) }
        v.findViewById<TextView>(R.id.ovEyebrow).apply { text = if (ok) "ГОТОВО" else "ТРЕВОГА СНЯТА"; setTextColor(ctx.getColor(R.color.calm)) }
        v.findViewById<TextView>(R.id.ovTitle).text = title
        v.findViewById<TextView>(R.id.ovSub).apply { text = sub; visibility = View.VISIBLE }
        v.findViewById<ImageView>(R.id.ovKind).apply { setImageResource(R.drawable.ic_b_check); setBackgroundResource(R.drawable.done_circle_calm); setColorFilter(ctx.getColor(R.color.on_calm)) }
        card.animate().scaleX(1.02f).scaleY(1.02f).setDuration(120).withEndAction {
            card.animate().scaleX(1f).scaleY(1f).setDuration(160).start()
        }.start()
        ui.postDelayed({
            card.animate().translationY(-card.height.toFloat() - 40 * d).alpha(0f).setDuration(260).setInterpolator(DecelerateInterpolator())
                .withEndAction { remove() }.start()
        }, 1400)
    }

    fun remove() {
        ui.removeCallbacksAndMessages(null)
        anims.forEach { it.cancel() }; anims.clear()
        root?.let { try { wm.removeView(it) } catch (_: Exception) {} }
        root = null; info = null; collapsed = false; closing = false
    }

    /** Сейчас идёт показ подтверждения — сервер может снять тревогу параллельно, не перебиваем. */
    val finishing: Boolean get() = closing
}
