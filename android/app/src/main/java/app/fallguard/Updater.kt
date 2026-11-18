package app.fallguard

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Автообновление из GitHub Releases. Работает только в сборках CI (там один постоянный ключ подписи,
 * и новая версия ставится поверх). Тихо обновиться нельзя: Android просит подтвердить установку.
 */
object Updater {
    data class Release(val code: Int, val name: String, val url: String)

    val enabled: Boolean get() = BuildConfig.UPDATE_REPO.isNotBlank()

    private fun sp(ctx: Context) = ctx.applicationContext.getSharedPreferences("fallguard_update", Context.MODE_PRIVATE)

    /** Номер версии из имени файла «FallGuard-1.N.apk». */
    fun codeFromAsset(name: String): Int? = Regex("""FallGuard-1\.(\d+)\.apk""").find(name)?.groupValues?.get(1)?.toIntOrNull()

    /** Самый свежий релиз с APK новее установленной версии, или null. */
    fun parseLatest(json: String, current: Int): Release? {
        val o = try { JSONObject(json) } catch (_: Exception) { return null }
        val assets = o.optJSONArray("assets") ?: return null
        var best: Release? = null
        for (i in 0 until assets.length()) {
            val a = assets.getJSONObject(i)
            val code = codeFromAsset(a.optString("name")) ?: continue
            // API-адрес файла (с Accept: octet-stream) надёжнее прямой ссылки; прямая — запасная
            val api = a.optString("url")
            if (code > current && (best == null || code > best.code)) best = Release(code, "1.$code", api.ifEmpty { a.optString("browser_download_url") })
        }
        return best
    }

    /** Проверка не чаще раза в 12 часов (лимит GitHub API для неавторизованных — 60 запросов в час). */
    fun checkIfDue(ctx: Context, force: Boolean = false): Release? {
        if (!enabled) return null
        val now = System.currentTimeMillis()
        val p = sp(ctx)
        if (!force && now - p.getLong("checked", 0L) < 12 * 3600_000L) return pending(ctx)
        p.edit().putLong("checked", now).apply()
        var conn: HttpURLConnection? = null
        return try {
            conn = URL("https://api.github.com/repos/${BuildConfig.UPDATE_REPO}/releases/latest").openConnection() as HttpURLConnection
            conn.connectTimeout = 8000; conn.readTimeout = 10000
            conn.setRequestProperty("Accept", "application/vnd.github+json")
            conn.setRequestProperty("User-Agent", "FallGuardAndroid")
            if (conn.responseCode != 200) return pending(ctx)
            val rel = parseLatest(conn.inputStream.bufferedReader().readText(), BuildConfig.VERSION_CODE)
            if (rel != null) p.edit().putInt("code", rel.code).putString("url", rel.url).apply()
            rel
        } catch (_: Exception) {
            pending(ctx)
        } finally {
            conn?.disconnect()
        }
    }

    fun pending(ctx: Context): Release? {
        val p = sp(ctx)
        val code = p.getInt("code", 0)
        val url = p.getString("url", "") ?: ""
        return if (code > BuildConfig.VERSION_CODE && url.isNotEmpty()) Release(code, "1.$code", url) else null
    }

    fun notifiedFor(ctx: Context): Int = sp(ctx).getInt("notified", 0)
    fun markNotified(ctx: Context, code: Int) = sp(ctx).edit().putInt("notified", code).apply()

    /** SHA-256 сертификатов подписи: у установленного приложения и у скачанного файла. */
    private fun certs(info: android.content.pm.PackageInfo?): Set<String> {
        val signers = if (Build.VERSION.SDK_INT >= 28) info?.signingInfo?.apkContentsSigners else @Suppress("DEPRECATION") info?.signatures
        return signers.orEmpty().map { java.security.MessageDigest.getInstance("SHA-256").digest(it.toByteArray()).joinToString("") { b -> "%02x".format(b) } }.toSet()
    }

    /** null — подписи совпадают (или проверить нельзя), иначе — понятное пояснение. */
    fun signatureProblem(ctx: Context, apk: File): String? = try {
        val pm = ctx.packageManager
        val flag = if (Build.VERSION.SDK_INT >= 28) android.content.pm.PackageManager.GET_SIGNING_CERTIFICATES else @Suppress("DEPRECATION") android.content.pm.PackageManager.GET_SIGNATURES
        val mine = certs(pm.getPackageInfo(ctx.packageName, flag))
        val theirs = certs(pm.getPackageArchiveInfo(apk.path, flag))
        if (mine.isEmpty() || theirs.isEmpty() || mine.intersect(theirs).isNotEmpty()) null
        else "Подпись обновления другая — Android не поставит его поверх. Удалите приложение и поставьте новое"
    } catch (_: Exception) { null }

    /** Скачать APK (в фоне) и открыть системную установку. Возвращает текст ошибки или null. */
    fun downloadAndInstall(act: Activity, rel: Release, onDone: (String?) -> Unit) {
        if (Build.VERSION.SDK_INT >= 26 && !act.packageManager.canRequestPackageInstalls()) {
            act.startActivity(Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${act.packageName}")))
            onDone("Разрешите FallGuard устанавливать обновления и нажмите ещё раз")
            return
        }
        Thread {
            val dir = File(act.cacheDir, "updates").apply { mkdirs(); listFiles()?.forEach { it.delete() } }
            val file = File(dir, "FallGuard-${rel.name}.apk")
            var conn: HttpURLConnection? = null
            val err = try {
                var url = URL(rel.url)
                // GitHub отдаёт файл через переадресацию на другой домен — идём по ней вручную
                for (hop in 0 until 5) {
                    conn = url.openConnection() as HttpURLConnection
                    conn.instanceFollowRedirects = false
                    conn.connectTimeout = 10000; conn.readTimeout = 30000
                    conn.setRequestProperty("User-Agent", "FallGuardAndroid")
                    if (url.host == "api.github.com") conn.setRequestProperty("Accept", "application/octet-stream")
                    val code = conn.responseCode
                    if (code in 300..399) { url = URL(url, conn.getHeaderField("Location")); conn.disconnect(); continue }
                    if (code != 200) throw IllegalStateException("HTTP $code")
                    conn.inputStream.use { input -> file.outputStream().use { input.copyTo(it) } }
                    break
                }
                if (file.length() < 100_000) "Файл обновления не скачался" else signatureProblem(act, file)
            } catch (e: Exception) {
                "Не удалось скачать обновление"
            } finally {
                conn?.disconnect()
            }
            act.runOnUiThread {
                if (err == null) {
                    val uri = androidx.core.content.FileProvider.getUriForFile(act, "${act.packageName}.files", file)
                    try {
                        act.startActivity(
                            Intent(Intent.ACTION_VIEW).setDataAndType(uri, "application/vnd.android.package-archive")
                                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    } catch (_: Exception) {}
                }
                onDone(err)
            }
        }.start()
    }
}
