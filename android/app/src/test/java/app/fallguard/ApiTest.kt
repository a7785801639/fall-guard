package app.fallguard

import org.junit.Assert.assertEquals
import org.junit.Test

class ApiTest {
    @Test
    fun normalizeUrl_addsSchemeByAddressKind() {
        assertEquals("http://192.168.1.50:8090", Api.normalizeUrl("192.168.1.50:8090/"))
        assertEquals("http://fallguard.local:8090", Api.normalizeUrl(" fallguard.local:8090 "))
        assertEquals("http://nas:8090", Api.normalizeUrl("nas:8090"))
        assertEquals("https://nofall.example.com", Api.normalizeUrl("nofall.example.com"))
        assertEquals("https://nofall.example.com", Api.normalizeUrl("https://nofall.example.com/"))
        assertEquals("", Api.normalizeUrl("   "))
    }

    @Test fun hangingRequestCanBeAbortedFromAnotherThread() {
        // Сервер принимает соединение и молчит — как запрос по сети, которой уже нет
        val server = java.net.ServerSocket(0)
        val hold = Thread { try { server.accept().use { Thread.sleep(15_000) } } catch (_: Exception) {} }.apply { isDaemon = true; start() }
        var conn: java.net.HttpURLConnection? = null
        val started = System.currentTimeMillis()
        Thread { Thread.sleep(400); conn?.disconnect() }.apply { isDaemon = true; start() }
        val res = Api.request("http://127.0.0.1:${server.localPort}", "/api/alerts/poll", readMs = 10_000, onOpen = { conn = it })
        val took = System.currentTimeMillis() - started
        assertEquals(-1, res.code)
        org.junit.Assert.assertTrue("оборвалось за $took мс", took < 5_000)
        server.close(); hold.interrupt()
    }
}
