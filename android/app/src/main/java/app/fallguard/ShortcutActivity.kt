package app.fallguard

import android.app.Activity
import android.content.Intent
import android.os.Bundle

/**
 * Быстрые действия с иконки приложения. Без своего окна: делает дело и закрывается,
 * открытый пульт и маленькое окно с видео не перезапускаются.
 */
class ShortcutActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val paired = Prefs(this).paired
        when (intent?.getStringExtra(MainActivity.EXTRA_SHORTCUT)) {
            "pause" -> if (paired) AlertService.start(this, AlertService.ACTION_NEARBY)
                else open(Intent(this, MainActivity::class.java))
            "journal" -> open(Intent(this, MainActivity::class.java).putExtra(MainActivity.EXTRA_SHORTCUT, "journal"))
            "settings" -> open(Intent(this, SettingsActivity::class.java))
        }
        finish()
    }

    private fun open(i: Intent) {
        startActivity(i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }
}
