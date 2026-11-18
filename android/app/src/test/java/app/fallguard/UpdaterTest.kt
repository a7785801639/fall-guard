package app.fallguard

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class UpdaterTest {
    private val json = """{"tag_name":"android-v1.7","assets":[
        {"name":"FallGuard-1.7.apk","url":"https://api.github.com/repos/x/y/releases/assets/7","browser_download_url":"https://example/1.7.apk"},
        {"name":"notes.txt","browser_download_url":"https://example/n"}]}"""

    @Test fun parsesNewerRelease() {
        val r = Updater.parseLatest(json, current = 4)!!
        assertEquals(7, r.code)
        assertEquals("1.7", r.name)
        assertEquals("https://api.github.com/repos/x/y/releases/assets/7", r.url)
    }

    @Test fun ignoresSameOrOlder() {
        assertNull(Updater.parseLatest(json, current = 7))
        assertNull(Updater.parseLatest(json, current = 9))
    }

    @Test fun assetNameParsing() {
        assertEquals(12, Updater.codeFromAsset("FallGuard-1.12.apk"))
        assertNull(Updater.codeFromAsset("app-release.apk"))
        assertNull(Updater.parseLatest("not json", 1))
    }
}
