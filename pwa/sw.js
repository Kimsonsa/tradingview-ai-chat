// TradeAI PWA Service Worker — 최소한의 SW (PWA 설치 조건 충족용)
const CACHE_NAME = 'tradeai-shell-v1';

// PWA 셸 파일만 캐싱 (Streamlit 앱 자체는 캐싱하지 않음)
const SHELL_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Streamlit 앱 요청(8502)은 절대 캐싱하지 않음 — 항상 네트워크
  if (url.port === '8502') return;

  // PWA 셸 파일만 캐시 처리
  if (SHELL_ASSETS.includes(url.pathname)) {
    event.respondWith(
      caches.match(event.request).then(cached => cached || fetch(event.request))
    );
  }
});
