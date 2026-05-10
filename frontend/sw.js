// Service Worker for Jolter push notifications

self.addEventListener("push", function (event) {
  var data = { title: "Jolter", body: "Your Jolt is ready" };
  try {
    data = event.data.json();
  } catch (e) {
    // Use defaults if payload parsing fails
  }

  event.waitUntil(
    self.registration.showNotification(data.title || "Jolter", {
      body: data.body || "Your Jolt is ready",
      icon: "/assets/rewire_icon.png",
      badge: "/assets/rewire_icon.png",
      vibrate: [200, 100, 200],
      tag: "jolt-ready",
      renotify: true,
    })
  );
});

self.addEventListener("notificationclick", function (event) {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then(function (clientList) {
      // If app is already open, focus it
      for (var i = 0; i < clientList.length; i++) {
        var client = clientList[i];
        if (client.url.indexOf(self.location.origin) !== -1 && "focus" in client) {
          return client.focus();
        }
      }
      // Otherwise open the app
      return clients.openWindow("/");
    })
  );
});
