// TradeAI PWA Service Worker v2 — 구 캐시 강제 정리 + PWA 셸 캐싱
const CACHE_NAME = 'tradeai-shell-v2';

// PWA 셸 파일만 캐싱 (Streamlit 앱은 캐싱 안 함)
const SHELL_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png'
];

self.addEventListener('install', (event) => {
  // 즉시 활성화 — 구 SW 대체
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL_ASSETS))
  );
});

self.addEventListener('activate', (event) => {
  // 구 캐시 전부 삭제 (tradeai-v1, tradeai-cache-v1, tradeai-shell-v1 등)
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => {
          console.log('[SW] Deleting old cache:', k);
          return caches.delete(k);
        })
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Streamlit(8502) 요청은 절대 캐싱 안 함
  if (url.port === '8502') return;

  // PWA 셸 파일: 네트워크 우선, 실패 시 캐시
  if (SHELL_ASSETS.includes(url.pathname)) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // 성공하면 캐시도 업데이트
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // 나머지: 네트워크 직접 (캐싱 안 함)
});
