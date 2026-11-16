package app.fallguard

import android.app.Activity
import android.app.AlertDialog
import android.app.NotificationManager
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast

class SettingsActivity : Activity() {
    private lateinit var prefs: Prefs

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = Prefs(this)
        setContentView(R.layout.activity_settings)

        findViewById<EditText>(R.id.setLan).setText(prefs.lanUrl)
        findViewById<EditText>(R.id.setPublic).setText(prefs.publicUrl)

        findViewById<Button>(R.id.btnTest).setOnClickListener {
            AlertService.start(this, AlertService.ACTION_TEST)
        }
        findViewById<Button>(R.id.btnBattery).setOnClickListener {
            try {
                startActivity(
                    Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName")),
                )
            } catch (_: Exception) {
                startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
            }
        }
        findViewById<Button>(R.id.btnNotif).setOnClickListener {
            startActivity(
                Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(Settings.EXTRA_APP_PACKAGE, packageName),
            )
        }
        findViewById<Button>(R.id.btnFullScreen).setOnClickListener {
            if (Build.VERSION.SDK_INT >= 34) {
                try {
                    startActivity(
                        Intent(Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT, Uri.parse("package:$packageName")),
                    )
                } catch (_: Exception) {}
            }
        }
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

    private fun refresh() {
        val pm = getSystemService(PowerManager::class.java)
        val nm = getSystemService(NotificationManager::class.java)
        val batteryOk = pm.isIgnoringBatteryOptimizations(packageName)
        val notifOk = nm.areNotificationsEnabled()
        val fullOk = Build.VERSION.SDK_INT < 34 || nm.canUseFullScreenIntent()
        val mark = { ok: Boolean -> if (ok) "✓" else "✗" }
        findViewById<TextView>(R.id.setInfo).text = listOf(
            if (prefs.paired) "Телефон подключён к FallGuard." else "Телефон не подключён.",
            "${mark(notifOk)} уведомления разрешены",
            "${mark(batteryOk)} работа в фоне без ограничений",
            "${mark(fullOk)} окно тревоги поверх экрана блокировки",
            "Режим «только падения» переключается в пульте на главном экране.",
        ).joinToString("\n")
        findViewById<View>(R.id.btnBattery).visibility = if (batteryOk) View.GONE else View.VISIBLE
        findViewById<View>(R.id.btnFullScreen).visibility = if (fullOk) View.GONE else View.VISIBLE
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
        startActivity(
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK),
        )
        finish()
    }
}
