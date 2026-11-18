package app.fallguard

import android.app.Activity
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
        findViewById<View>(R.id.btnBack).setOnClickListener { finish() }
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
            AppDialog.show(this, "Отвязать телефон?", "Тревоги перестанут приходить на этот телефон.", actions = listOf(
                AppDialog.Action("Отмена", R.style.Btn) {},
                AppDialog.Action("Отвязать", R.style.Btn_Danger_Small) { unpair() },
            ))
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

    /** Окно настройки одного типа: способ сигнала, звук (с прослушиванием), свой файл и проверка. */
    private fun openSignal(kind: Signals.Kind) {
        fun usage(st: Signals.Style) = if (st == Signals.Style.ALARM) AudioAttributes.USAGE_ALARM else AudioAttributes.USAGE_NOTIFICATION_EVENT
        val body = SignalDialog.build(this, kind, object : SignalDialog.Callbacks {
            override fun onStyle(style: Signals.Style) { Sounds.setStyle(this@SettingsActivity, kind, style); renderSignals() }
            override fun onSound(id: String) {
                Sounds.setSound(this@SettingsActivity, kind, id)
                stopPreview()
                preview = Sounds.play(this@SettingsActivity, Sounds.uriOf(this@SettingsActivity, kind, id), usage(Signals.effectiveStyle(kind, "", Sounds.style(this@SettingsActivity, kind))), loop = false)
                renderSignals()
            }
            override fun onPickFile() {
                stopPreview()
                val pick = Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("audio/*")
                try { startActivityForResult(pick, REQ_SOUND + kind.ordinal) } catch (_: Exception) {
                    Toast.makeText(this@SettingsActivity, "Нет приложения для выбора файла", Toast.LENGTH_SHORT).show()
                }
            }
            override fun onTest() {
                stopPreview()
                AlertService.start(this@SettingsActivity, AlertService.ACTION_TEST_SIGNAL, kind.name)
            }
        })
        AppDialog.show(this, kind.title, body = body, actions = listOf(AppDialog.Action("Готово", R.style.Btn) {}), onDismiss = { stopPreview() })
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

/** Содержимое окна «Как сигналить» — отдельно от Activity, чтобы его можно было отрисовать в тестах. */
object SignalDialog {
    interface Callbacks {
        fun onStyle(style: Signals.Style)
        fun onSound(id: String)
        fun onPickFile()
        fun onTest()
    }

    fun build(ctx: Context, kind: Signals.Kind, cb: Callbacks): View {
        fun dp(v: Int) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), ctx.resources.displayMetrics).toInt()
        val box = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(22), dp(4), dp(22), dp(12)) }
        fun header(t: String) = TextView(ctx).apply {
            text = t; textSize = 12f; setTextColor(ctx.getColor(R.color.muted)); isAllCaps = true; letterSpacing = 0.08f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, dp(16), 0, dp(6))
        }
        fun radio(id: Int, title: String, checked: Boolean) = RadioButton(ctx).apply {
            this.id = id; text = title; textSize = 16f; isChecked = checked; minHeight = dp(46)
            setTextColor(ctx.getColor(R.color.text))
            buttonTintList = android.content.res.ColorStateList(
                arrayOf(intArrayOf(android.R.attr.state_checked), intArrayOf()),
                intArrayOf(ctx.getColor(R.color.calm), ctx.getColor(R.color.muted)),
            )
            setPadding(dp(8), 0, 0, 0)
        }
        val styles = when (kind) {
            Signals.Kind.FALL -> listOf(Signals.Style.ALARM)
            Signals.Kind.LINK -> listOf(Signals.Style.NOTIFY, Signals.Style.SILENT)
            else -> Signals.Style.entries.toList()
        }
        box.addView(header("Как сигналить"))
        val cur = Signals.effectiveStyle(kind, "", Sounds.style(ctx, kind))
        val styleGroup = RadioGroup(ctx)
        styles.forEachIndexed { i, st -> styleGroup.addView(radio(100 + i, st.title, st == cur)) }
        box.addView(styleGroup)
        if (kind == Signals.Kind.FALL) box.addView(TextView(ctx).apply { text = "Падение всегда поднимает громкую тревогу"; setTextColor(ctx.getColor(R.color.muted)); textSize = 13f })

        box.addView(header("Звук"))
        val options = Sounds.BUILT_IN.map { it.id to it.title }.toMutableList()
        if (Sounds.customFile(ctx, kind).exists()) options += Sounds.CUSTOM to Sounds.customName(ctx, kind)
        val curSound = Sounds.sound(ctx, kind)
        val soundGroup = RadioGroup(ctx)
        options.forEachIndexed { i, (id, title) -> soundGroup.addView(radio(200 + i, title, id == curSound)) }
        box.addView(soundGroup)
        // Стиль не задаёт размеры у кнопок, созданных в коде — задаём явно
        fun btn(style: Int, t: String, top: Int, click: () -> Unit) = Button(ctx, null, 0, style).apply {
            text = t; isAllCaps = false; gravity = Gravity.CENTER; setOnClickListener { click() }
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(52)).apply { topMargin = dp(top) }
        }
        box.addView(btn(R.style.Btn, "Свой файл…", 10) { cb.onPickFile() })
        box.addView(btn(R.style.Btn_Primary, "Проверить", 10) { cb.onTest() })

        styleGroup.setOnCheckedChangeListener { _, checked -> styles.getOrNull(checked - 100)?.let { cb.onStyle(it) } }
        soundGroup.setOnCheckedChangeListener { _, checked -> options.getOrNull(checked - 200)?.first?.let { cb.onSound(it) } }
        return box
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
