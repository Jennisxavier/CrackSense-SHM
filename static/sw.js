self.addEventListener('install', (event) => {
  console.log('Service Worker installed');
});

self.addEventListener('fetch', (event) => {
  // Simple pass-through fetch handler is required for PWA installability
  event.respondWith(fetch(event.request));
});
