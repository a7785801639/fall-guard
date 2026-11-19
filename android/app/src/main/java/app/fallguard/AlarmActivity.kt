package app.fallguard

import android.app.Activity
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.app.KeyguardManager
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Кадр ниже этой высоты (dp) — повод убрать большой значок над заголовком. */
const val HERO_MIN_FRAME_DP = 150

/** Окно тревоги поверх экрана блокировки. */
class AlarmActivity : Activity() {
    private val ui = Handler(Looper.getMainLooper())
    private var alarmTs = 0L
    private val pulses = mutableListOf<AnimatorSet>()
    private val tick = object : Runnable {
        override fun run() {
            updateElapsed()
            findViewById<View>(R.id.alarmRing)?.invalidate() // без системных анимаций кольцо иначе замрёт
            ui.postDelayed(this, 1000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        getSystemService(KeyguardManager::class.java)?.requestDismissKeyguard(this, null)
        // Телефон в руке поворачивается как попало — на телефоне экран тревоги всегда вертикальный;
        // планшет на подставке показывает его как стоит
        if (resources.configuration.smallestScreenWidthDp < 600) requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        setContentView(R.layout.activity_alarm)
        edgeToEdge()
        // Пока идёт подтверждение, сервер может уже снять тревогу — не обрываем анимацию
        AlertService.onAlarmStopped = { runOnUiThread { if (!closing) finish() } }
        // Низкий экран: кадр важнее большого значка — убираем значок, кадр становится крупнее
        findViewById<View>(R.id.alarmFrame).addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> fitHero() }
        bind()
        enter()
    }

    private fun fitHero() {
        if (findViewById<FitBox>(R.id.alarmFrame).childCount < 2) return // горизонтальная раскладка: значок в шапке
        val badge = findViewById<View>(R.id.alarmBadgeWrap)
        val card = findViewById<View>(R.id.alarmFrameCard)
        if (badge.visibility == View.VISIBLE && card.height in 1 until (HERO_MIN_FRAME_DP * resources.displayMetrics.density).toInt())
            card.post { badge.visibility = View.GONE }
    }

    /**
     * Окно под прозрачными панелями. Отступы считаем сами: fitsSystemWindows заменил бы
     * собственные поля экрана отступами панелей, и на Android 15 текст лёг бы вплотную к краям.
     */
    private fun edgeToEdge() {
        androidx.core.view.WindowCompat.setDecorFitsSystemWindows(window, false)
        if (android.os.Build.VERSION.SDK_INT >= 29) { window.isNavigationBarContrastEnforced = false; window.isStatusBarContrastEnforced = false }
        val content = findViewById<View>(R.id.alarmContent)
        val base = intArrayOf(content.paddingLeft, content.paddingTop, content.paddingRight, content.paddingBottom)
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(content) { v, insets ->
            val b = insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars() or androidx.core.view.WindowInsetsCompat.Type.displayCutout())
            v.setPadding(base[0] + b.left, base[1] + b.top, base[2] + b.right, base[3] + b.bottom)
            insets
        }
    }

    /** Экран тревоги появляется не скачком: блоки по очереди поднимаются на место */
    private fun enter() {
        val root = findViewById<android.view.ViewGroup>(R.id.alarmContent) ?: return
        // Вспышка: свечение сначала ярче, потом переходит в «дыхание»
        findViewById<View>(R.id.alarmGlow).apply { alpha = 0f; animate().alpha(1f).setDuration(260).start() }
        val dy = resources.displayMetrics.density * 22
        for (i in 0 until root.childCount) {
            val v = root.getChildAt(i)
            v.alpha = 0f
            v.translationY = dy
            v.animate().alpha(1f).translationY(0f).setStartDelay(i * 45L).setDuration(340)
                .setInterpolator(android.view.animation.DecelerateInterpolator(1.6f)).start()
        }
    }

    override fun onResume() {
        super.onResume()
        ui.removeCallbacks(tick)
        ui.post(tick)
        startPulse()
        startAmbient()
        findViewById<View>(R.id.alarmRing).invalidate()
        if (alarmInfo != null) startLive()
    }

    override fun onPause() {
        liveGen++
        ui.removeCallbacks(tick)
        stopPulse()
        stopAmbient()
        stopUrgency()
        super.onPause()
    }

    /** Кольца расходятся от знака тревоги, второе — со сдвигом на полпериода */
    private fun startPulse() {
        stopPulse()
        val badge = findViewById<View>(R.id.alarmBadge)
        listOf(R.id.alarmRing1 to 0L, R.id.alarmRing2 to 700L).forEach { (id, delay) ->
            val ring = findViewById<View>(id)
            val set = AnimatorSet().apply {
                playTogether(
                    ObjectAnimator.ofFloat(ring, View.SCALE_X, 1f, 1.9f).apply { repeatCount = ValueAnimator.INFINITE },
                    ObjectAnimator.ofFloat(ring, View.SCALE_Y, 1f, 1.9f).apply { repeatCount = ValueAnimator.INFINITE },
                    ObjectAnimator.ofFloat(ring, View.ALPHA, 0.7f, 0f).apply { repeatCount = ValueAnimator.INFINITE },
                )
                duration = 1400
                startDelay = delay
            }
            set.start()
            pulses += set
        }
        val beat = AnimatorSet().apply {
            playTogether(
                ObjectAnimator.ofFloat(badge, View.SCALE_X, 1f, 1.08f).apply { repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE },
                ObjectAnimator.ofFloat(badge, View.SCALE_Y, 1f, 1.08f).apply { repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE },
            )
            duration = 700
        }
        beat.start()
        pulses += beat
    }

    private fun stopPulse() {
        pulses.forEach { it.cancel() }
        pulses.clear()
    }

    private fun updateElapsed() {
        val sec = ((ServerClock.now() - alarmTs) / 1000).coerceAtLeast(0)
        val text = if (sec >= 3600) "%d:%02d:%02d".format(sec / 3600, sec / 60 % 60, sec % 60)
        else "%d:%02d".format(sec / 60, sec % 60)
        val elapsed = findViewById<TextView>(R.id.alarmElapsed)
        elapsed.text = text
        // Чем дольше без ответа, тем тревожнее: белый → янтарный через минуту → мигает через три
        elapsed.setTextColor(findViewById<ElapsedRing>(R.id.alarmRing).colorFor(sec))
        if (sec >= 180) {
            if (urgency == null) urgency = ObjectAnimator.ofFloat(elapsed, View.ALPHA, 1f, 0.45f).apply {
                duration = 700; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE; start()
            }
        } else stopUrgency()
    }

    private var urgency: ObjectAnimator? = null

    private fun stopUrgency() {
        urgency?.cancel(); urgency = null
        findViewById<View>(R.id.alarmElapsed)?.alpha = 1f
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        bind()
    }

    private fun bind() {
        val info = AlertService.current
        if (info == null) {
            finish()
            return
        }
        alarmTs = info.ts
        findViewById<ElapsedRing>(R.id.alarmRing).startTs = info.ts
        findViewById<ImageView>(R.id.alarmIcon).setImageResource(
            when (info.kind) {
                Signals.Kind.FALL -> R.drawable.ic_k_fall
                Signals.Kind.STANDUP -> R.drawable.ic_k_walk
                Signals.Kind.SITTING -> R.drawable.ic_k_sit
                Signals.Kind.HELP -> R.drawable.ic_k_hand
                Signals.Kind.LINK -> R.drawable.ic_k_link
                Signals.Kind.OTHER -> R.drawable.ic_k_alert
            },
        )
        updateElapsed()
        findViewById<TextView>(R.id.alarmTime).text =
            (if (info.base.isEmpty()) "ПРОВЕРКА · " else "ТРЕВОГА · ") + SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(info.ts))
        findViewById<TextView>(R.id.alarmTitle).text = info.title

        iconLabel(findViewById(R.id.btnAck), R.drawable.ic_b_check, "Принято, иду")
        iconLabel(findViewById(R.id.btnFalse), R.drawable.ic_b_x, "Ложная тревога")
        iconLabel(findViewById(R.id.btnOpen), R.drawable.ic_b_eye, "Смотреть камеру")
        iconLabel(findViewById(R.id.btnSilence), R.drawable.ic_b_mute, "Выключить звук на этом телефоне")
        findViewById<Button>(R.id.btnAck).setOnClickListener { v ->
            if (closing) return@setOnClickListener
            // Проверочная тревога (без сервера) только замолкает — охрану на паузу не ставим
            AlertService.start(this, if (info.base.isEmpty()) AlertService.ACTION_SILENCE else AlertService.ACTION_ACK, info.id)
            confirm(v, true, "Принято", if (info.base.isEmpty()) "Это была проверка — так и сработает" else "Идите к ней — сирена выключена")
        }
        findViewById<Button>(R.id.btnFalse).setOnClickListener { v ->
            if (closing) return@setOnClickListener
            AlertService.start(this, if (info.base.isEmpty()) AlertService.ACTION_SILENCE else AlertService.ACTION_FALSE, info.id)
            confirm(v, false, "Ложная тревога", "Отметили — так система станет точнее")
        }
        findViewById<Button>(R.id.btnSilence).setOnClickListener {
            AlertService.start(this, AlertService.ACTION_SILENCE, info.id)
            finish()
        }
        findViewById<Button>(R.id.btnOpen).setOnClickListener {
            AlertService.start(this, AlertService.ACTION_SILENCE, info.id)
            startActivity(Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            finish()
        }

        val image = findViewById<ImageView>(R.id.alarmImage)
        findViewById<View>(R.id.alarmFrameCard).clipToOutline = true // скругление кадра (XML-атрибут работает только с API 31)
        image.setOnClickListener { liveWanted = !liveWanted; renderChip(); if (liveWanted) startLive() else showEventShot() }
        alarmInfo = info
        eventShot = null
        renderChip()
        if (info.imageUrl.isNotEmpty() && info.base.isNotEmpty()) {
            val token = Prefs(this).token
            Thread {
                // Снимок сохраняется на сервере чуть позже самой тревоги — пробуем несколько раз
                for (attempt in 0 until 4) {
                    val data = Api.bytes(info.base, info.imageUrl, token)
                    if (data != null) {
                        val bmp = BitmapFactory.decodeByteArray(data, 0, data.size)
                        eventShot = bmp
                        runOnUiThread { if (!isFinishing && !liveShowing) setFrame(bmp) }
                        return@Thread
                    }
                    try { Thread.sleep(1200) } catch (_: InterruptedException) { return@Thread }
                }
            }.start()
        }
        startLive()
    }

    // ---------------------------------------------------------------- атмосфера: дыхание фона и блик на кнопке
    private val ambient = mutableListOf<android.animation.Animator>()

    private fun startAmbient() {
        stopAmbient()
        if (!ValueAnimator.areAnimatorsEnabled()) return
        val glow = findViewById<View>(R.id.alarmGlow)
        ambient += ObjectAnimator.ofFloat(glow, View.ALPHA, 1f, 0.45f).apply {
            duration = 1300; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE
            interpolator = android.view.animation.AccelerateDecelerateInterpolator()
            startDelay = 300
            start()
        }
        val wrap = findViewById<View>(R.id.ackWrap)
        val shine = findViewById<View>(R.id.ackShine)
        val r = resources.displayMetrics.density * 16
        wrap.outlineProvider = object : android.view.ViewOutlineProvider() {
            override fun getOutline(view: View, outline: android.graphics.Outline) { outline.setRoundRect(0, 0, view.width, view.height, r) }
        }
        wrap.clipToOutline = true
        wrap.post {
            if (isFinishing) return@post
            ambient += ObjectAnimator.ofFloat(shine, View.TRANSLATION_X, -shine.width.toFloat(), wrap.width.toFloat()).apply {
                duration = 1100; startDelay = 900
                interpolator = android.view.animation.AccelerateDecelerateInterpolator()
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.RESTART
                // Пауза между бликами — через задержку повтора
                addListener(object : android.animation.AnimatorListenerAdapter() {
                    override fun onAnimationRepeat(a: android.animation.Animator) {
                        (a as ObjectAnimator).pause(); ui.postDelayed({ if (a.isPaused) a.resume() }, 1800)
                    }
                })
                start()
            }
        }
    }

    private fun stopAmbient() {
        ambient.forEach { it.cancel() }
        ambient.clear()
        findViewById<View>(R.id.alarmGlow)?.alpha = 1f
    }

    // ---------------------------------------------------------------- подтверждение нажатия
    @Volatile private var closing = false

    /** Экран раскрывается кругом от нажатой кнопки: галочка, короткая подпись — и окно закрывается. */
    private fun confirm(from: View, ok: Boolean, title: String, sub: String) {
        closing = true
        from.performHapticFeedback(if (android.os.Build.VERSION.SDK_INT >= 30) android.view.HapticFeedbackConstants.CONFIRM else android.view.HapticFeedbackConstants.LONG_PRESS)
        stopAmbient(); stopPulse(); stopUrgency(); liveGen++
        val done = findViewById<View>(R.id.alarmDone)
        done.setBackgroundColor(if (ok) 0xFF0F2A1F.toInt() else 0xFF15181E.toInt())
        findViewById<View>(R.id.doneCircle).setBackgroundResource(if (ok) R.drawable.done_circle_calm else R.drawable.done_circle_neutral)
        findViewById<ImageView>(R.id.doneIcon).setImageResource(if (ok) R.drawable.ic_done_check else R.drawable.ic_done_flag)
        findViewById<TextView>(R.id.doneTitle).text = title
        findViewById<TextView>(R.id.doneSub).text = sub
        val circle = findViewById<View>(R.id.doneCircle)
        circle.scaleX = 0.4f; circle.scaleY = 0.4f; circle.alpha = 0f
        val anim = ValueAnimator.areAnimatorsEnabled()
        if (anim) done.visibility = View.INVISIBLE else { done.visibility = View.VISIBLE; circle.scaleX = 1f; circle.scaleY = 1f; circle.alpha = 1f }
        // Размеры известны только после разметки — запускаем раскрытие следующим кадром
        if (anim) done.post {
            val loc = IntArray(2); from.getLocationInWindow(loc)
            val rootLoc = IntArray(2); done.getLocationInWindow(rootLoc)
            val cx = loc[0] - rootLoc[0] + from.width / 2
            val cy = loc[1] - rootLoc[1] + from.height / 2
            val radius = Math.hypot(done.width.toDouble(), done.height.toDouble()).toFloat()
            done.visibility = View.VISIBLE
            try {
                android.view.ViewAnimationUtils.createCircularReveal(done, cx, cy, 0f, radius).setDuration(420).start()
            } catch (_: Exception) {}
            circle.animate().scaleX(1f).scaleY(1f).alpha(1f).setStartDelay(160).setDuration(380)
                .setInterpolator(android.view.animation.OvershootInterpolator(2.2f)).start()
        }
        ui.postDelayed({
            finish()
            @Suppress("DEPRECATION") overridePendingTransition(0, android.R.anim.fade_out)
        }, if (anim) 1300 else 700)
    }

    // ---------------------------------------------------------------- живое видео на экране тревоги
    private var alarmInfo: AlarmInfo? = null
    @Volatile private var eventShot: android.graphics.Bitmap? = null
    @Volatile private var liveWanted = true
    @Volatile private var liveShowing = false
    @Volatile private var liveGen = 0

    private fun renderChip() {
        val chip = findViewById<TextView>(R.id.alarmLive)
        val t = alarmInfo?.ts ?: 0L
        chip.text = if (liveWanted && liveShowing) "● Эфир" else "Снимок " + SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(t))
        chip.setTextColor(if (liveWanted && liveShowing) 0xFFFF6B70.toInt() else 0xFFFFFFFF.toInt())
    }

    /** Кадр показываем целиком: рамка принимает пропорции камеры. */
    private fun setFrame(bmp: android.graphics.Bitmap) {
        findViewById<ImageView>(R.id.alarmImage).setImageBitmap(bmp)
        if (bmp.height > 0) findViewById<FitBox>(R.id.alarmFrame)?.aspect = bmp.width.toFloat() / bmp.height
    }

    private fun showEventShot() {
        liveShowing = false
        eventShot?.let { setFrame(it) }
        renderChip()
    }

    /** Кадры с камеры ~3 раза в секунду с разметкой людей: видно, двигается ли она. */
    private fun startLive() {
        val info = alarmInfo ?: return
        if (info.base.isEmpty() || !liveWanted) return
        val gen = ++liveGen
        val token = Prefs(this).token
        Thread {
            var fails = 0
            while (gen == liveGen && liveWanted && !isFinishing) {
                val data = Api.bytes(info.base, "/api/snapshot.jpg?overlay=1", token)
                val bmp = data?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
                if (bmp != null) {
                    fails = 0
                    runOnUiThread {
                        if (gen == liveGen && liveWanted && !isFinishing) {
                            setFrame(bmp)
                            if (!liveShowing) { liveShowing = true; renderChip() }
                        }
                    }
                } else if (++fails >= 3) {
                    runOnUiThread { if (gen == liveGen) showEventShot() }
                }
                try { Thread.sleep(if (fails > 0) 1500 else 330) } catch (_: InterruptedException) { return@Thread }
            }
        }.start()
    }

    override fun onDestroy() {
        liveGen++
        ui.removeCallbacksAndMessages(null)
        stopPulse()
        stopAmbient()
        AlertService.onAlarmStopped = null
        super.onDestroy()
    }
}

/** Значок перед текстом кнопки — по центру вместе с подписью. */
fun iconLabel(b: Button, icon: Int, text: String) {
    val d = b.context.getDrawable(icon)?.mutate() ?: run { b.text = text; return }
    val size = (b.textSize * 1.1f).toInt()
    d.setBounds(0, 0, size, size)
    d.setTint(b.currentTextColor)
    val align = if (android.os.Build.VERSION.SDK_INT >= 29) android.text.style.ImageSpan.ALIGN_CENTER else android.text.style.ImageSpan.ALIGN_BASELINE
    b.text = android.text.SpannableString("\u00A0 $text").apply {
        setSpan(android.text.style.ImageSpan(d, align), 0, 1, android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    }
}
