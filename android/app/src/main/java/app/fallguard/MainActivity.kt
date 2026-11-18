package app.fallguard

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.app.DownloadManager
import android.app.PendingIntent
import android.app.PictureInPictureParams
import android.app.RemoteAction
import android.content.BroadcastReceiver
import android.content.Context
import android.content.IntentFilter
import android.content.res.Configuration
import android.graphics.Rect
import android.graphics.drawable.Icon
import android.util.Rational
import android.content.ContentValues
import android.graphics.Color
import android.os.Environment
import android.os.VibrationEffect
import android.os.Vibrator
import android.provider.MediaStore
import android.util.Base64
import android.widget.Toast
import java.io.File
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.webkit.CookieManager
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.TextView
import org.json.JSONObject

/** Главный экран: привязка телефона, затем пульт FallGuard во встроенном браузере. */
class MainActivity : Activity() {
    private lateinit var prefs: Prefs
    private lateinit var web: WebView
    private lateinit var setup: View
    private lateinit var offline: View
    private lateinit var fullscreen: FrameLayout
    private var customView: View? = null
    private var customCallback: WebChromeClient.CustomViewCallback? = null
    private var fileCallback: ValueCallback<Array<Uri>>? = null
    private var loadedBase = ""

    // ---- Окно поверх (Picture-in-Picture) ----
    @Volatile private var pipAlarm = false
    @Volatile private var pipZoomed = false
    @Volatile private var pipOverlay = true
    @Volatile private var pipW = 16
    @Volatile private var pipH = 9
    private var pipReceiver: BroadcastReceiver? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = Prefs(this)
        setContentView(R.layout.activity_main)
        web = findViewById(R.id.web)
        setup = findViewById(R.id.setup)
        offline = findViewById(R.id.offline)
        fullscreen = findViewById(R.id.fullscreen)
        setupWebView()

        findViewById<Button>(R.id.btnPair).setOnClickListener {
            pair(
                listOf(Api.normalizeUrl(findViewById<EditText>(R.id.inServer).text.toString())),
                findViewById<EditText>(R.id.inCode).text.toString(),
            )
        }
        findViewById<Button>(R.id.btnRetry).setOnClickListener { connect() }
        findViewById<View>(R.id.btnScan).setOnClickListener { scanQr() }
        findViewById<Button>(R.id.btnOfflineSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        if (!handlePairLink(intent)) route()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handlePairLink(intent)
    }

    override fun onResume() {
        super.onResume()
        applyKeepScreenOn()
        if (!prefs.paired) show(setup) else if (loadedBase.isEmpty()) connect()
        web.onResume()
    }

    override fun onPause() {
        // В маленьком окне видео должно идти дальше — WebView не останавливаем
        if (!isInPictureInPictureMode) web.onPause()
        super.onPause()
    }

    override fun onStart() {
        super.onStart()
        web.evaluateJavascript("window.fallguardBackground && window.fallguardBackground(false)", null)
    }

    override fun onStop() {
        // Свернули (или закрыли маленькое окно): видео, сокеты и опрос пульта останавливаются —
        // тревоги в фоне ловит AlertService, трафик и батарея не тратятся
        web.evaluateJavascript("window.fallguardBackground && window.fallguardBackground(true)", null)
        if (isInPictureInPictureMode) web.onPause()
        super.onStop()
    }

    override fun onDestroy() {
        pipReceiver?.let { try { unregisterReceiver(it) } catch (_: Exception) {} }
        super.onDestroy()
    }

    /** Свернули приложение кнопкой «Домой» — уходим в маленькое окно, если смотрим видео. */
    override fun onUserLeaveHint() {
        super.onUserLeaveHint()
        if (Build.VERSION.SDK_INT < 31) enterPip()
    }

    private fun pipAllowed(): Boolean =
        prefs.pipEnabled && web.visibility == View.VISIBLE && loadedBase.isNotEmpty() && customView == null &&
            packageManager.hasSystemFeature(PackageManager.FEATURE_PICTURE_IN_PICTURE)

    private fun pipAction(icon: Int, title: String, cmd: String, req: Int): RemoteAction {
        val pi = PendingIntent.getBroadcast(
            this, req, Intent(PIP_ACTION).setPackage(packageName).putExtra("cmd", cmd),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return RemoteAction(Icon.createWithResource(this, icon), title, title, pi)
    }

    private fun pipParams(): PictureInPictureParams {
        // Android принимает соотношение сторон от 1:2.39 до 2.39:1
        var w = pipW.coerceAtLeast(1); var h = pipH.coerceAtLeast(1)
        if (w.toFloat() / h > 2.39f) w = (h * 2.39f).toInt()
        if (h.toFloat() / w > 2.39f) h = (w * 2.39f).toInt()
        val actions = mutableListOf<RemoteAction>()
        if (pipAlarm) actions += pipAction(R.drawable.ic_pip_check, "Принято, иду", "ack", 31)
        actions += pipAction(R.drawable.ic_pip_zoom, if (pipZoomed) "Весь кадр" else "Кровать крупно", "zoom", 32)
        if (!pipAlarm) actions += pipAction(R.drawable.ic_pip_eye, if (pipOverlay) "Скрыть разметку" else "Показать разметку", "overlay", 33)
        val b = PictureInPictureParams.Builder()
            .setAspectRatio(Rational(w, h))
            .setActions(actions.take(maxNumPictureInPictureActions.coerceAtLeast(1)))
        val r = Rect(); web.getGlobalVisibleRect(r)
        if (!r.isEmpty) b.setSourceRectHint(Rect(r.left, r.top, r.right, r.top + r.width() * h / w))
        if (Build.VERSION.SDK_INT >= 31) {
            b.setAutoEnterEnabled(pipAllowed())
            b.setSeamlessResizeEnabled(false)
        }
        if (Build.VERSION.SDK_INT >= 33) b.setTitle(if (pipAlarm) "Тревога" else "FallGuard")
        return b.build()
    }

    private fun updatePip() {
        if (!packageManager.hasSystemFeature(PackageManager.FEATURE_PICTURE_IN_PICTURE)) return
        try { setPictureInPictureParams(pipParams()) } catch (_: Exception) {}
    }

    private fun enterPip(): Boolean {
        if (!pipAllowed() || isInPictureInPictureMode) return false
        return try { enterPictureInPictureMode(pipParams()) } catch (_: Exception) { false }
    }

    override fun onPictureInPictureModeChanged(isInPip: Boolean, newConfig: Configuration) {
        super.onPictureInPictureModeChanged(isInPip, newConfig)
        if (isInPip) {
            if (pipReceiver == null) {
                pipReceiver = object : BroadcastReceiver() {
                    override fun onReceive(c: Context, i: Intent) {
                        val js = when (i.getStringExtra("cmd")) {
                            "ack" -> "ackAlarm()"
                            "zoom" -> "toggleBedZoom()"
                            "overlay" -> "toggleOverlay()"
                            else -> return
                        }
                        web.evaluateJavascript("try{$js}catch(e){}", null)
                    }
                }
                androidx.core.content.ContextCompat.registerReceiver(
                    this, pipReceiver, IntentFilter(PIP_ACTION), androidx.core.content.ContextCompat.RECEIVER_NOT_EXPORTED,
                )
            }
        }
        // Пульт перестраивается: в окне остаются только видео и статус
        web.evaluateJavascript("window.fallguardPip && window.fallguardPip(${if (isInPip) "true" else "false"})", null)
        if (!isInPip) web.onResume()
    }

    private fun applyKeepScreenOn() {
        if (prefs.keepScreenOn) window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun setSystemBarsHidden(on: Boolean) {
        val ctrl = androidx.core.view.WindowCompat.getInsetsController(window, window.decorView)
        if (on) {
            ctrl.systemBarsBehavior = androidx.core.view.WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            ctrl.hide(androidx.core.view.WindowInsetsCompat.Type.systemBars())
        } else {
            ctrl.show(androidx.core.view.WindowInsetsCompat.Type.systemBars())
        }
    }

    private fun show(target: View) {
        for (v in listOf(web, setup, offline)) v.visibility = if (v === target) View.VISIBLE else View.GONE
        if (target !== web) {
            // Красный фон тревоги от пульта не должен оставаться на экранах привязки и «нет связи»
            val bg = getColor(R.color.bg)
            window.statusBarColor = bg
            findViewById<View>(R.id.root)?.setBackgroundColor(bg)
        }
    }

    private fun route() {
        if (prefs.paired) {
            AlertService.start(this)
            askNotificationPermission()
            connect()
            // Сразу после привязки — проверка, что тревога действительно дойдёт
            if (!prefs.onboardingShown) {
                prefs.onboardingShown = true
                startActivity(Intent(this, SettingsActivity::class.java).putExtra(SettingsActivity.EXTRA_ONBOARDING, true))
            }
        } else {
            show(setup)
        }
    }

    // ---------------------------------------------------------------- привязка
    /** Ссылка из QR-кода пульта: fallguard://pair?code=…&server=…&lan=… */
    private fun handlePairLink(intent: Intent?): Boolean {
        val uri = intent?.data ?: return false
        if (uri.scheme != "fallguard" || uri.host != "pair") return false
        val code = uri.getQueryParameter("code") ?: ""
        val servers = listOf(uri.getQueryParameter("lan"), uri.getQueryParameter("server"))
            .map { Api.normalizeUrl(it ?: "") }.filter { it.isNotEmpty() }.distinct()
        intent.data = null
        if (code.isEmpty() || servers.isEmpty()) return false
        show(setup)
        findViewById<EditText>(R.id.inServer).setText(servers.last())
        findViewById<EditText>(R.id.inCode).setText(code)
        pair(servers, code)
        return true
    }

    /** Сканер QR-кода из пульта: «fallguard://pair?…» сразу привязывает телефон. */
    private fun scanQr() {
        val msg = findViewById<TextView>(R.id.setupMsg)
        val opts = com.google.mlkit.vision.codescanner.GmsBarcodeScannerOptions.Builder()
            .setBarcodeFormats(com.google.mlkit.vision.barcode.common.Barcode.FORMAT_QR_CODE)
            .enableAutoZoom()
            .build()
        com.google.mlkit.vision.codescanner.GmsBarcodeScanning.getClient(this, opts).startScan()
            .addOnSuccessListener { code ->
                val raw = code.rawValue ?: ""
                val uri = try { Uri.parse(raw) } catch (_: Exception) { null }
                if (uri == null || !handlePairLink(Intent(Intent.ACTION_VIEW, uri))) {
                    msg.setTextColor(getColor(R.color.alarm))
                    msg.text = "Это не QR-код FallGuard. Откройте в пульте «Подключить телефон»."
                }
            }
            .addOnFailureListener {
                msg.setTextColor(getColor(R.color.alarm))
                msg.text = "Сканер недоступен на этом телефоне — введите адрес и код ниже."
            }
    }

    private fun pair(servers: List<String>, code: String) {
        val msg = findViewById<TextView>(R.id.setupMsg)
        val btn = findViewById<Button>(R.id.btnPair)
        val list = servers.filter { it.isNotEmpty() }
        if (list.isEmpty()) {
            msg.text = "Введите адрес сервера FallGuard"
            return
        }
        btn.isEnabled = false
        msg.setTextColor(getColor(R.color.muted))
        msg.text = "Подключаюсь…"
        Thread {
            var error = "Сервер не отвечает. Проверьте адрес и что телефон в той же сети или есть интернет."
            var done = false
            val body = JSONObject().put("code", code.trim()).put("name", Build.MODEL ?: "Android").put("platform", "android")
            for (base in list) {
                val res = Api.request(base, "/api/devices/claim", "POST", body = body)
                if (res.ok) {
                    val j = res.json()
                    prefs.token = j.optString("token")
                    prefs.deviceId = j.optJSONObject("device")?.optString("id") ?: ""
                    prefs.cursor = 0
                    // Запоминаем оба адреса: домашний (http) и внешний (https)
                    prefs.lanUrl = list.firstOrNull { it.startsWith("http://") } ?: ""
                    prefs.publicUrl = list.firstOrNull { it.startsWith("https://") } ?: ""
                    done = prefs.paired
                    break
                } else if (res.code > 0) {
                    error = res.json().optString("detail", "Сервер отклонил код привязки")
                    break
                }
            }
            runOnUiThread {
                btn.isEnabled = true
                if (done) {
                    msg.text = ""
                    route()
                } else {
                    msg.setTextColor(getColor(R.color.alarm))
                    msg.text = error
                }
            }
        }.start()
    }

    private fun askNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 10)
        }
    }

    // ---------------------------------------------------------------- пульт
    /** Пока соединяемся — только индикатор; кнопки «Повторить» и «Настройки» — после неудачи. */
    private fun setConnecting(on: Boolean) {
        findViewById<View>(R.id.offlineProgress).visibility = if (on) View.VISIBLE else View.GONE
        findViewById<View>(R.id.offlineIcon).visibility = if (on) View.GONE else View.VISIBLE
        findViewById<View>(R.id.btnRetry).visibility = if (on) View.INVISIBLE else View.VISIBLE
        findViewById<View>(R.id.btnOfflineSettings).visibility = if (on) View.INVISIBLE else View.VISIBLE
    }

    private fun connect() {
        show(offline)
        setConnecting(true)
        findViewById<TextView>(R.id.offlineTitle).text = "Соединяюсь с FallGuard…"
        findViewById<TextView>(R.id.offlineText).text = ""
        Thread {
            val base = Api.reachableBase(prefs)
            runOnUiThread {
                if (isFinishing) return@runOnUiThread
                if (base == null) {
                    setConnecting(false)
                    findViewById<TextView>(R.id.offlineTitle).text = "Нет связи с сервером"
                    findViewById<TextView>(R.id.offlineText).text =
                        "Тревоги придут, когда связь вернётся."
                } else {
                    loadedBase = base
                    val cm = CookieManager.getInstance()
                    cm.setAcceptCookie(true)
                    cm.setCookie(base, "fg_device=${prefs.token}; Path=/; Max-Age=31536000")
                    cm.flush()
                    web.loadUrl("$base/")
                    show(web)
                }
            }
        }.start()
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        web.setBackgroundColor(getColor(R.color.bg))
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            mediaPlaybackRequiresUserGesture = false
            mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
            userAgentString = "$userAgentString FallGuardAndroid/1.0"
        }
        web.addJavascriptInterface(Bridge(), "FallGuardAndroid")
        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val url = request.url.toString()
                if (loadedBase.isNotEmpty() && url.startsWith(loadedBase)) return false
                try { startActivity(Intent(Intent.ACTION_VIEW, request.url)) } catch (_: Exception) {}
                return true
            }

            override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
                if (request.isForMainFrame) {
                    loadedBase = ""
                    connect()
                }
            }
        }
        web.webChromeClient = object : WebChromeClient() {
            override fun onShowCustomView(view: View, callback: CustomViewCallback) {
                customView = view
                customCallback = callback
                fullscreen.addView(view)
                fullscreen.visibility = View.VISIBLE
                setSystemBarsHidden(true)
            }

            override fun onHideCustomView() {
                fullscreen.removeAllViews()
                fullscreen.visibility = View.GONE
                setSystemBarsHidden(false)
                customView = null
                customCallback = null
            }

            override fun onShowFileChooser(
                view: WebView,
                callback: ValueCallback<Array<Uri>>,
                params: FileChooserParams,
            ): Boolean {
                fileCallback?.onReceiveValue(null)
                fileCallback = callback
                return try {
                    @Suppress("DEPRECATION")
                    startActivityForResult(params.createIntent(), REQ_FILE)
                    true
                } catch (_: Exception) {
                    fileCallback = null
                    false
                }
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_FILE) {
            fileCallback?.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(resultCode, data))
            fileCallback = null
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (customView != null) {
            customCallback?.onCustomViewHidden()
            return
        }
        if (web.visibility == View.VISIBLE) {
            web.evaluateJavascript("(window.fallguardBack && window.fallguardBack()) ? 1 : 0") { handled ->
                if (handled != "1") moveTaskToBack(true)
            }
            return
        }
        @Suppress("DEPRECATION")
        super.onBackPressed()
    }

    inner class Bridge {
        @JavascriptInterface
        fun openAppSettings() {
            runOnUiThread { startActivity(Intent(this@MainActivity, SettingsActivity::class.java)) }
        }

        /** Телефон или планшет у кровати как монитор: экран не гаснет, пока открыт пульт. */
        @JavascriptInterface
        fun setKeepScreenOn(on: Boolean) {
            prefs.keepScreenOn = on
            runOnUiThread { applyKeepScreenOn() }
        }

        @JavascriptInterface
        fun isKeepScreenOn(): Boolean = prefs.keepScreenOn

        @JavascriptInterface
        fun isReady(): Boolean = Readiness.check(this@MainActivity).allOk

        /** Пульт сообщает, что показывать в маленьком окне: тревога, увеличение, разметка, размер кадра. */
        @JavascriptInterface
        fun setPipState(alarm: Boolean, zoomed: Boolean, overlay: Boolean, w: Int, h: Int) {
            val changed = alarm != pipAlarm || zoomed != pipZoomed || overlay != pipOverlay || (w > 0 && w != pipW) || (h > 0 && h != pipH)
            pipAlarm = alarm; pipZoomed = zoomed; pipOverlay = overlay
            if (w > 0 && h > 0) { pipW = w; pipH = h }
            if (changed) runOnUiThread { updatePip() }
        }

        /** Кнопка «Маленькое окно» в пульте. */
        @JavascriptInterface
        fun enterPip(): Boolean {
            if (!pipAllowed()) return false
            runOnUiThread { this@MainActivity.enterPip() }
            return true
        }

        @JavascriptInterface
        fun isPipEnabled(): Boolean = prefs.pipEnabled

        @JavascriptInterface
        fun setPipEnabled(on: Boolean) {
            prefs.pipEnabled = on
            runOnUiThread { updatePip() }
        }

        /** Короткий отклик на нажатие: tap, success, warning, error. */
        @JavascriptInterface
        fun haptic(kind: String) {
            val vib = getSystemService(Vibrator::class.java) ?: return
            val effect = when (kind) {
                "success" -> VibrationEffect.createWaveform(longArrayOf(0, 18, 60, 28), -1)
                "warning" -> VibrationEffect.createWaveform(longArrayOf(0, 40, 80, 40), -1)
                "error" -> VibrationEffect.createWaveform(longArrayOf(0, 60, 60, 60, 60, 60), -1)
                else -> VibrationEffect.createOneShot(12, VibrationEffect.DEFAULT_AMPLITUDE)
            }
            try { vib.vibrate(effect) } catch (_: Exception) {}
        }

        /** Строка состояния окрашивается под статус пульта: красная при тревоге. */
        @JavascriptInterface
        fun setBarColor(hex: String) {
            val c = try { Color.parseColor(hex) } catch (_: Exception) { return }
            runOnUiThread {
                window.statusBarColor = c
                // Android 15+ рисует приложение под прозрачной строкой состояния — красим фон под ней
                findViewById<View>(R.id.root)?.setBackgroundColor(c)
            }
        }

        /** Кадр с камеры (JPEG в base64) — в галерею, папка Pictures/FallGuard. */
        @JavascriptInterface
        fun saveImage(b64: String, name: String): Boolean {
            val bytes = try { Base64.decode(b64, Base64.DEFAULT) } catch (_: Exception) { return false }
            if (Build.VERSION.SDK_INT < 29) return false
            return try {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, safeName(name, "jpg"))
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/FallGuard")
                }
                val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return false
                contentResolver.openOutputStream(uri)?.use { it.write(bytes) } ?: return false
                true
            } catch (_: Exception) { false }
        }

        /** Поделиться кадром: Telegram, мессенджеры, почта. */
        @JavascriptInterface
        fun shareImage(b64: String, name: String): Boolean {
            return try {
                val dir = File(cacheDir, "shared").apply { mkdirs() }
                dir.listFiles()?.forEach { it.delete() }
                val f = File(dir, safeName(name, "jpg"))
                f.writeBytes(Base64.decode(b64, Base64.DEFAULT))
                val uri = androidx.core.content.FileProvider.getUriForFile(this@MainActivity, "$packageName.files", f)
                val send = Intent(Intent.ACTION_SEND).setType("image/jpeg").putExtra(Intent.EXTRA_STREAM, uri)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                runOnUiThread { startActivity(Intent.createChooser(send, "Отправить кадр")) }
                true
            } catch (_: Exception) { false }
        }

        /** Видео записи — в «Загрузки» через системный загрузчик, с авторизацией телефона. */
        @JavascriptInterface
        fun downloadVideo(url: String, name: String): Boolean {
            val base = loadedBase
            if (base.isEmpty()) return false
            val full = if (url.startsWith("http")) url else base.trimEnd('/') + "/" + url.trimStart('/')
            if (!full.startsWith(base)) return false
            return try {
                val req = DownloadManager.Request(Uri.parse(full))
                    .addRequestHeader("Cookie", "fg_device=${prefs.token}")
                    .setTitle(name)
                    .setDescription("Запись FallGuard")
                    .setMimeType("video/mp4")
                    .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                    .setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, "FallGuard/" + safeName(name, "mp4"))
                getSystemService(DownloadManager::class.java)?.enqueue(req)
                runOnUiThread { Toast.makeText(this@MainActivity, "Видео скачивается — уведомление покажет, когда готово", Toast.LENGTH_SHORT).show() }
                true
            } catch (_: Exception) { false }
        }
    }

    private fun safeName(raw: String, ext: String): String {
        val n = raw.replace(Regex("[^A-Za-z0-9._-]"), "_").take(80).ifBlank { "fallguard" }
        return if (n.endsWith(".$ext")) n else "$n.$ext"
    }

    companion object {
        private const val REQ_FILE = 20
        private const val PIP_ACTION = "app.fallguard.PIP"
    }
}
