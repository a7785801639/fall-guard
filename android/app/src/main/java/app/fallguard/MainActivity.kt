package app.fallguard

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
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
        if (!prefs.paired) show(setup) else if (loadedBase.isEmpty()) connect()
        web.onResume()
    }

    override fun onPause() {
        web.onPause()
        super.onPause()
    }

    private fun show(target: View) {
        for (v in listOf(web, setup, offline)) v.visibility = if (v === target) View.VISIBLE else View.GONE
    }

    private fun route() {
        if (prefs.paired) {
            AlertService.start(this)
            askNotificationPermission()
            connect()
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
    private fun connect() {
        show(offline)
        findViewById<TextView>(R.id.offlineTitle).text = "Соединяюсь с FallGuard…"
        findViewById<TextView>(R.id.offlineText).text = ""
        Thread {
            val base = Api.reachableBase(prefs)
            runOnUiThread {
                if (isFinishing) return@runOnUiThread
                if (base == null) {
                    findViewById<TextView>(R.id.offlineTitle).text = "Нет связи с сервером"
                    findViewById<TextView>(R.id.offlineText).text =
                        "Тревоги продолжат приходить, как только связь появится.\n" + prefs.urls.joinToString("\n")
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
            }

            override fun onHideCustomView() {
                fullscreen.removeAllViews()
                fullscreen.visibility = View.GONE
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
    }

    companion object {
        private const val REQ_FILE = 20
    }
}
