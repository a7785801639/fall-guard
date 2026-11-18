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

/** Окно тревоги поверх экрана блокировки. */
class AlarmActivity : Activity() {
    private val ui = Handler(Looper.getMainLooper())
    private var alarmTs = 0L
    private val pulses = mutableListOf<AnimatorSet>()
    private val tick = object : Runnable {
        override fun run() {
            updateElapsed()
            ui.postDelayed(this, 1000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        getSystemService(KeyguardManager::class.java)?.requestDismissKeyguard(this, null)
        setContentView(R.layout.activity_alarm)
        AlertService.onAlarmStopped = { runOnUiThread { finish() } }
        // Таймер виден, только если для него есть место: кнопки важнее
        val box = findViewById<View>(R.id.alarmTimerBox)
        box.addOnLayoutChangeListener { v, _, top, _, bottom, _, _, _, _ ->
            val need = findViewById<View>(R.id.alarmElapsed).measuredHeight +
                findViewById<View>(R.id.alarmElapsedLabel).measuredHeight
            val show = (bottom - top) >= need + v.resources.displayMetrics.density * 24
            for (i in 0 until (v as android.view.ViewGroup).childCount) {
                v.getChildAt(i).visibility = if (show) View.VISIBLE else View.INVISIBLE
            }
        }
        bind()
        enter()
    }

    /** Экран тревоги появляется не скачком: блоки по очереди поднимаются на место */
    private fun enter() {
        val root = findViewById<android.view.ViewGroup>(android.R.id.content).getChildAt(0) as? android.view.ViewGroup ?: return
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
        if (alarmInfo != null) startLive()
    }

    override fun onPause() {
        liveGen++
        ui.removeCallbacks(tick)
        stopPulse()
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
        val sec = ((System.currentTimeMillis() - alarmTs) / 1000).coerceAtLeast(0)
        val text = if (sec >= 3600) "%d:%02d:%02d".format(sec / 3600, sec / 60 % 60, sec % 60)
        else "%d:%02d".format(sec / 60, sec % 60)
        findViewById<TextView>(R.id.alarmElapsed).text = text
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
        updateElapsed()
        findViewById<TextView>(R.id.alarmTime).text =
            "ТРЕВОГА · " + SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(info.ts))
        findViewById<TextView>(R.id.alarmTitle).text = info.title
        findViewById<TextView>(R.id.alarmDesc).text = info.description

        findViewById<Button>(R.id.btnAck).setOnClickListener {
            AlertService.start(this, AlertService.ACTION_ACK, info.id)
            finish()
        }
        findViewById<Button>(R.id.btnFalse).setOnClickListener {
            AlertService.start(this, AlertService.ACTION_FALSE, info.id)
            finish()
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
        image.clipToOutline = true // скругление кадра (XML-атрибут работает только с API 31)
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
                        runOnUiThread { if (!isFinishing && !liveShowing) image.setImageBitmap(bmp) }
                        return@Thread
                    }
                    try { Thread.sleep(1200) } catch (_: InterruptedException) { return@Thread }
                }
            }.start()
        }
        startLive()
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

    private fun showEventShot() {
        liveShowing = false
        eventShot?.let { findViewById<ImageView>(R.id.alarmImage).setImageBitmap(it) }
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
                            findViewById<ImageView>(R.id.alarmImage).setImageBitmap(bmp)
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
        ui.removeCallbacks(tick)
        stopPulse()
        AlertService.onAlarmStopped = null
        super.onDestroy()
    }
}
