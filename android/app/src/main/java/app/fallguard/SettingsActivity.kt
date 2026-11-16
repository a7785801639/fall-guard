package app.fallguard

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast

/** Проверка готовности телефона к тревоге (после привязки — как пошаговая настройка) и адреса сервера. */
class SettingsActivity : Activity() {
    private lateinit var prefs: Prefs
    private var onboarding = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = Prefs(this)
        onboarding = intent.getBooleanExtra(EXTRA_ONBOARDING, false)
        setContentView(R.layout.activity_settings)

        findViewById<EditText>(R.id.setLan).setText(prefs.lanUrl)
        findViewById<EditText>(R.id.setPublic).setText(prefs.publicUrl)
        findViewById<TextView>(R.id.setVersion).text = "FallGuard ${packageManager.getPackageInfo(packageName, 0).versionName}"

        findViewById<Button>(R.id.btnTest).setOnClickListener { AlertService.start(this, AlertService.ACTION_TEST) }
        findViewById<Button>(R.id.btnDone).setOnClickListener { finish() }
        findViewById<Button>(R.id.btnSaveUrls).setOnClickListener {
            val lan = Api.normalizeUrl(findViewById<EditText>(R.id.setLan).text.toString())
            val pub = Api.normalizeUrl(findViewById<EditText>(R.id.setPublic).text.toString())
            if (lan.isEmpty() && pub.isEmpty()) {
                Toast.makeText(this, "Нужен хотя бы один адрес", Toast.LENGTH_SHORT).show()
            } else {
                prefs.lanUrl = lan
                prefs.publicUrl = pub
                Toast.makeText(this, "Адреса сохранены", Toast.LENGTH_SHORT).show()
                restartApp()
            }
        }
        findViewById<Button>(R.id.btnUnpair).setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Отвязать телефон?")
                .setMessage("Тревоги перестанут приходить на этот телефон.")
                .setPositiveButton("Отвязать") { _, _ -> unpair() }
                .setNegativeButton("Отмена", null)
                .show()
        }
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    private fun dp(v: Int): Int = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()

    private fun refresh() {
        val st = Readiness.check(this)
        findViewById<TextView>(R.id.setTitle).text = when {
            onboarding && !st.allOk -> "Ещё пара шагов"
            onboarding -> "Всё готово"
            else -> "Настройки приложения"
        }
        findViewById<TextView>(R.id.setInfo).text = when {
            !st.allOk -> "Разрешите приложению всё отмеченное ниже — иначе Android может заглушить тревогу ночью или на заблокированном телефоне."
            onboarding -> "Телефон подключён. При падении он громко зазвонит, даже в беззвучном режиме. Нажмите «Проверить тревогу», чтобы услышать, как это звучит."
            else -> "Телефон подключён к FallGuard и готов к тревоге. Режим «только падения» переключается в пульте."
        }
        findViewById<View>(R.id.btnDone).visibility = if (onboarding) View.VISIBLE else View.GONE

        val list = findViewById<LinearLayout>(R.id.checklist)
        list.removeAllViews()
        addRow(list, st.notifications, "Уведомления", "Без них тревога не появится") {
            startActivity(Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(Settings.EXTRA_APP_PACKAGE, packageName))
        }
        addRow(list, st.battery, "Работа в фоне", "Чтобы Android не усыплял связь с сервером") {
            try {
                startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName")))
            } catch (_: Exception) {
                startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
            }
        }
        addRow(list, st.fullScreen, "Окно поверх блокировки", "Тревога откроется на весь экран") {
            if (Build.VERSION.SDK_INT >= 34) {
                try {
                    startActivity(Intent(Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT, Uri.parse("package:$packageName")))
                } catch (_: Exception) {}
            }
        }
    }

    private fun addRow(parent: LinearLayout, ok: Boolean, title: String, sub: String, onFix: () -> Unit) {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundResource(R.drawable.card)
            setPadding(dp(16), dp(14), dp(12), dp(14))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(8) }
        }
        val mark = TextView(this).apply {
            text = if (ok) "✓" else "!"
            textSize = 18f
            gravity = Gravity.CENTER
            setTextColor(getColor(if (ok) R.color.calm else R.color.warn))
            layoutParams = LinearLayout.LayoutParams(dp(28), LinearLayout.LayoutParams.WRAP_CONTENT)
        }
        val texts = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(8) }
            addView(TextView(this@SettingsActivity).apply { text = title; textSize = 16f; setTextColor(getColor(R.color.text)) })
            addView(TextView(this@SettingsActivity).apply { text = if (ok) "Разрешено" else sub; textSize = 13f; setTextColor(getColor(R.color.muted)) })
        }
        row.addView(mark)
        row.addView(texts)
        if (!ok) {
            row.addView(Button(this, null, 0, R.style.Btn_Primary).apply {
                text = "Разрешить"
                textSize = 14f
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, dp(42)).apply { marginStart = dp(8) }
                setPadding(dp(14), 0, dp(14), 0)
                setOnClickListener { onFix() }
            })
        }
        parent.addView(row)
    }

    private fun unpair() {
        val token = prefs.token
        val id = prefs.deviceId
        val urls = prefs.urls
        Thread {
            if (id.isNotEmpty()) {
                for (base in urls) if (Api.request(base, "/api/devices/$id", "DELETE", token).ok) break
            }
        }.start()
        stopService(Intent(this, AlertService::class.java))
        prefs.clear()
        restartApp()
    }

    private fun restartApp() {
        startActivity(Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK))
        finish()
    }

    companion object {
        const val EXTRA_ONBOARDING = "onboarding"
    }
}
