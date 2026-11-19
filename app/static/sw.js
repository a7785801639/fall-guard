const CACHE_NAME = "fallguard-shell-v41";
const SHELL_ASSETS = [
  "/",
  "/manifest.webmanifest",
  "/icon.svg"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Живые потоки MJPEG, снимки и видеоклипы с HTTP Range не перехватываем
  if (
    url.pathname.startsWith("/api/stream") ||
    url.pathname.startsWith("/api/snapshot") ||
    url.pathname.startsWith("/api/clips/") ||
    req.headers.has("range")
  ) {
    return;
  }

  // Навигация и статика оболочки — network-first с резервом из кэша
  if (req.mode === "navigate" || SHELL_ASSETS.includes(url.pathname)) {
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res && res.ok) {
            const copy = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, copy)).catch(() => {});
          }
          return res;
        })
        .catch(() => caches.match(req).then((cached) => cached || caches.match("/")))
    );
  }
});
