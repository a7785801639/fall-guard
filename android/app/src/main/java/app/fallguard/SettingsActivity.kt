package app.fallguard

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
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
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.provider.OpenableColumns
import android.media.AudioAttributes
import android.media.MediaPlayer
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
        setupUpdates()
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
        renderSignals()
    }

    private fun refresh() {
        val st = Readiness.check(this)
        findViewById<TextView>(R.id.setTitle).text = when {
            onboarding && !st.allOk -> "Ещё пара шагов"
            onboarding -> "Всё готово"
            else -> "Настройки приложения"
        }
        findViewById<TextView>(R.id.setInfo).text = when {
            !st.allOk -> "Без этого Android может заглушить тревогу."
            onboarding -> "При падении телефон зазвонит даже в беззвучном режиме."
            else -> "Телефон готов к тревоге."
        }
        findViewById<View>(R.id.btnDone).visibility = if (onboarding) View.VISIBLE else View.GONE

        val list = findViewById<LinearLayout>(R.id.checklist)
        list.removeAllViews()
        addRow(list, st.notifications, "Уведомления", "обязательно") {
            startActivity(Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(Settings.EXTRA_APP_PACKAGE, packageName))
        }
        addRow(list, st.battery, "Работа в фоне", "чтобы связь не засыпала") {
            try {
                startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName")))
            } catch (_: Exception) {
                startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
            }
        }
        addRow(list, st.fullScreen, "Окно поверх блокировки", "тревога на весь экран") {
            if (Build.VERSION.SDK_INT >= 34) {
                try {
                    startActivity(Intent(Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT, Uri.parse("package:$packageName")))
                } catch (_: Exception) {}
            }
        }
    }

    // ---------------------------------------------------------------- обновления
    private fun setupUpdates() {
        if (!Updater.enabled) return
        val check = findViewById<Button>(R.id.btnCheckUpdate)
        check.visibility = View.VISIBLE
        check.setOnClickListener {
            check.isEnabled = false; check.text = "Проверяю…"
            Thread {
                val rel = Updater.checkIfDue(this, force = true)
                runOnUiThread {
                    check.isEnabled = true; check.text = "Проверить обновления"
                    if (rel == null) Toast.makeText(this, "Установлена последняя версия", Toast.LENGTH_SHORT).show()
                    renderUpdate()
                }
            }.start()
        }
        findViewById<Button>(R.id.btnUpdate).setOnClickListener { installUpdate() }
        renderUpdate()
        if (intent.getBooleanExtra(EXTRA_UPDATE, false)) installUpdate()
    }

    private fun renderUpdate() {
        val rel = if (Updater.enabled) Updater.pending(this) else null
        findViewById<View>(R.id.updateBox).visibility = if (rel != null) View.VISIBLE else View.GONE
        if (rel != null) findViewById<TextView>(R.id.updateTitle).text = "Доступна версия ${rel.name}"
    }

    private fun installUpdate() {
        val rel = Updater.pending(this) ?: return
        val btn = findViewById<Button>(R.id.btnUpdate)
        btn.isEnabled = false; btn.text = "Скачиваю…"
        Updater.downloadAndInstall(this, rel) { err ->
            btn.isEnabled = true; btn.text = "Установить"
            if (err != null) Toast.makeText(this, err, Toast.LENGTH_LONG).show()
        }
    }

    // ---------------------------------------------------------------- сигналы и звуки
    private var preview: MediaPlayer? = null

    private fun stopPreview() {
        try { preview?.stop() } catch (_: Exception) {}
        try { preview?.release() } catch (_: Exception) {}
        preview = null
    }

    override fun onPause() {
        stopPreview()
        super.onPause()
    }

    private fun renderSignals() {
        val list = findViewById<LinearLayout>(R.id.signalList)
        list.removeAllViews()
        for (kind in Signals.Kind.entries) {
            val style = Signals.effectiveStyle(kind, "", Sounds.style(this, kind))
            val sub = if (style == Signals.Style.SILENT) style.title else "${style.title.substringBefore(" ")} · ${Sounds.title(this, kind)}"
            list.addView(SignalRow.build(this, kind.title, sub, style) { openSignal(kind) })
        }
    }

    /** Окно настройки одного типа: способ сигнала, звук (с прослушиванием) и свой файл. */
    private fun openSignal(kind: Signals.Kind) {
        fun dp(v: Int) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(22), dp(8), dp(22), dp(8)) }
        fun header(t: String) = TextView(this).apply {
            text = t; textSize = 13f; setTextColor(getColor(R.color.muted)); isAllCaps = true; letterSpacing = 0.06f
            setPadding(0, dp(14), 0, dp(4))
        }
        val styles = when (kind) {
            Signals.Kind.FALL -> listOf(Signals.Style.ALARM)
            Signals.Kind.LINK -> listOf(Signals.Style.NOTIFY, Signals.Style.SILENT)
            else -> Signals.Style.entries.toList()
        }
        box.addView(header("Как сигналить"))
        val styleGroup = RadioGroup(this)
        val cur = Signals.effectiveStyle(kind, "", Sounds.style(this, kind))
        styles.forEachIndexed { i, st ->
            styleGroup.addView(RadioButton(this).apply { id = 100 + i; text = st.title; textSize = 16f; isChecked = st == cur; minHeight = dp(44) })
        }
        if (kind == Signals.Kind.FALL) box.addView(TextView(this).apply { text = "Падение всегда поднимает громкую тревогу"; setTextColor(getColor(R.color.muted)); textSize = 13f })
        box.addView(styleGroup)

        box.addView(header("Звук"))
        val soundGroup = RadioGroup(this)
        val options = Sounds.BUILT_IN.map { it.id to it.title }.toMutableList()
        if (Sounds.customFile(this, kind).exists()) options += Sounds.CUSTOM to Sounds.customName(this, kind)
        val curSound = Sounds.sound(this, kind)
        options.forEachIndexed { i, (id, title) ->
            soundGroup.addView(RadioButton(this).apply { this.id = 200 + i; text = title; textSize = 16f; isChecked = id == curSound; minHeight = dp(44) })
        }
        box.addView(soundGroup)
        box.addView(Button(this, null, 0, R.style.Btn).apply {
            text = "Свой файл…"
            setOnClickListener {
                stopPreview()
                val pick = Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("audio/*")
                try { startActivityForResult(pick, REQ_SOUND + kind.ordinal) } catch (_: Exception) {
                    Toast.makeText(this@SettingsActivity, "Нет приложения для выбора файла", Toast.LENGTH_SHORT).show()
                }
            }
        })

        fun usage(st: Signals.Style) = if (st == Signals.Style.ALARM) AudioAttributes.USAGE_ALARM else AudioAttributes.USAGE_NOTIFICATION_EVENT
        soundGroup.setOnCheckedChangeListener { _, checked ->
            val id = options.getOrNull(checked - 200)?.first ?: return@setOnCheckedChangeListener
            Sounds.setSound(this, kind, id)
            stopPreview()
            preview = Sounds.play(this, Sounds.uriOf(this, kind, id), usage(Signals.effectiveStyle(kind, "", Sounds.style(this, kind))), loop = false)
            renderSignals()
        }
        styleGroup.setOnCheckedChangeListener { _, checked ->
            styles.getOrNull(checked - 100)?.let { Sounds.setStyle(this, kind, it) }
            renderSignals()
        }
        AlertDialog.Builder(this)
            .setTitle(kind.title)
            .setView(ScrollView(this).apply { addView(box) })
            .setPositiveButton("Готово", null)
            .setOnDismissListener { stopPreview() }
            .show()
    }

    @Deprecated("Activity API")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val kind = Signals.Kind.entries.getOrNull(requestCode - REQ_SOUND) ?: return
        val uri = data?.data ?: return
        if (resultCode != RESULT_OK) return
        var name = "Свой файл"
        try {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) name = c.getString(0).substringBeforeLast('.')
            }
        } catch (_: Exception) {}
        if (Sounds.importCustom(this, kind, uri, name)) {
            Toast.makeText(this, "Звук «$name» выбран", Toast.LENGTH_SHORT).show()
            renderSignals()
            openSignal(kind)
        } else {
            Toast.makeText(this, "Этот файл не проигрывается — выберите другой", Toast.LENGTH_LONG).show()
        }
    }

    private fun addRow(parent: LinearLayout, ok: Boolean, title: String, sub: String, onFix: () -> Unit) {
        parent.addView(ChecklistRow.build(this, ok, title, sub, onFix))
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
        const val EXTRA_UPDATE = "update"
        private const val REQ_SOUND = 300
    }
}

/** Строка сигнала: название · способ и звук · стрелка. */
object SignalRow {
    fun build(ctx: Context, title: String, sub: String, style: Signals.Style, onClick: () -> Unit): View {
        fun dp(v: Int): Int = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), ctx.resources.displayMetrics).toInt()
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundResource(R.drawable.btn_surface)
            setPadding(dp(16), dp(12), dp(12), dp(12))
            minimumHeight = dp(60)
            isClickable = true
            setOnClickListener { onClick() }
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(8) }
        }
        val dot = View(ctx).apply {
            setBackgroundResource(when (style) { Signals.Style.ALARM -> R.drawable.dot_alarm; Signals.Style.NOTIFY -> R.drawable.dot_warn_solid; else -> R.drawable.dot_muted })
            layoutParams = LinearLayout.LayoutParams(dp(10), dp(10))
        }
        row.addView(dot)
        row.addView(LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(14) }
            addView(TextView(ctx).apply { text = title; textSize = 16f; setTextColor(ctx.getColor(R.color.text)) })
            addView(TextView(ctx).apply { text = sub; textSize = 13f; setTextColor(ctx.getColor(R.color.muted)) })
        })
        row.addView(android.widget.ImageView(ctx).apply { setImageResource(R.drawable.ic_chev); layoutParams = LinearLayout.LayoutParams(dp(22), dp(22)) })
        return row
    }
}

/** Строка проверки готовности: ✓/! · название и пояснение · кнопка «Разрешить». */
object ChecklistRow {
    fun build(ctx: Context, ok: Boolean, title: String, sub: String, onFix: () -> Unit): View {
        fun dp(v: Int): Int = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), ctx.resources.displayMetrics).toInt()
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundResource(R.drawable.card)
            setPadding(dp(16), dp(14), dp(12), dp(14))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(8) }
        }
        row.addView(android.widget.ImageView(ctx).apply {
            setImageResource(if (ok) R.drawable.ic_check else R.drawable.ic_alert)
            setBackgroundResource(if (ok) R.drawable.dot_calm else R.drawable.dot_warn)
            setPadding(dp(7), dp(7), dp(7), dp(7))
            contentDescription = null
            layoutParams = LinearLayout.LayoutParams(dp(36), dp(36))
        })
        row.addView(LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(14) }
            addView(TextView(ctx).apply { text = title; textSize = 16f; setTextColor(ctx.getColor(R.color.text)) })
            addView(TextView(ctx).apply { text = if (ok) "Разрешено" else sub; textSize = 13f; setTextColor(ctx.getColor(R.color.muted)) })
        })
        if (!ok) {
            row.addView(Button(ctx, null, 0, R.style.Btn_Primary).apply {
                text = "Разрешить"
                textSize = 14f
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, dp(42)).apply { marginStart = dp(8) }
                minHeight = 0
                minimumHeight = 0
                gravity = Gravity.CENTER
                includeFontPadding = false
                setPadding(dp(14), 0, dp(14), 0)
                setOnClickListener { onFix() }
            })
        }
        return row
    }
}
