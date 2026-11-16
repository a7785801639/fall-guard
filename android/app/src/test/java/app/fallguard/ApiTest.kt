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
}
