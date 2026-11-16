package app.fallguard

import android.app.Activity
import android.app.KeyguardManager
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Окно тревоги поверх экрана блокировки. */
class AlarmActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        getSystemService(KeyguardManager::class.java)?.requestDismissKeyguard(this, null)
        setContentView(R.layout.activity_alarm)
        AlertService.onAlarmStopped = { runOnUiThread { finish() } }
        bind()
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
        if (info.imageUrl.isNotEmpty() && info.base.isNotEmpty()) {
            val token = Prefs(this).token
            Thread {
                // Снимок сохраняется на сервере чуть позже самой тревоги — пробуем несколько раз
                for (attempt in 0 until 4) {
                    val data = Api.bytes(info.base, info.imageUrl, token)
                    if (data != null) {
                        val bmp = BitmapFactory.decodeByteArray(data, 0, data.size)
                        runOnUiThread { if (!isFinishing) image.setImageBitmap(bmp) }
                        return@Thread
                    }
                    try { Thread.sleep(1200) } catch (_: InterruptedException) { return@Thread }
                }
            }.start()
        }
    }

    override fun onDestroy() {
        AlertService.onAlarmStopped = null
        super.onDestroy()
    }
}
