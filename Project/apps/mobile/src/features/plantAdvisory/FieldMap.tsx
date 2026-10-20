import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { WebView, type WebViewMessageEvent } from 'react-native-webview';

import { colors, radii, spacing } from '../../constants/theme';

// Where the map starts before a pin is placed (centre of Sri Lanka).
const START = { latitude: 7.8731, longitude: 80.7718, zoom: 7 };
const PIN_ZOOM = 15;

// A small Leaflet page shown inside a WebView. Leaflet is loaded from the
// unpkg CDN with integrity hashes (same version and hashes as the web
// dashboard) and the tiles come from OpenStreetMap, so no API key is needed.
// The page and the app talk through postMessage:
//   page -> app:  { type: 'ready' } | { type: 'pick', lat, lng } | { type: 'failed' }
//   app  -> page: window.setPin(lat, lng, zoom)
const MAP_HTML = `<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <style>html, body, #map { margin: 0; height: 100%; background: #F6FAF4; }</style>
</head>
<body>
  <div id="map"></div>
  <script>
    function send(message) {
      window.ReactNativeWebView.postMessage(JSON.stringify(message));
    }

    if (typeof L === 'undefined') {
      // Leaflet didn't download (no internet, CDN blocked)
      send({ type: 'failed' });
    } else {
      var map = L.map('map').setView([${START.latitude}, ${START.longitude}], ${START.zoom});
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
      }).addTo(map);

      var marker = null;

      function placeMarker(lat, lng) {
        if (!marker) {
          marker = L.marker([lat, lng], { draggable: true }).addTo(map);
          marker.on('dragend', function () {
            var p = marker.getLatLng().wrap();
            send({ type: 'pick', lat: p.lat, lng: p.lng });
          });
        } else {
          marker.setLatLng([lat, lng]);
        }
      }

      map.on('click', function (e) {
        var p = e.latlng.wrap();
        placeMarker(p.lat, p.lng);
        send({ type: 'pick', lat: p.lat, lng: p.lng });
      });

      // called by the app for GPS or typed coordinates
      window.setPin = function (lat, lng, zoom) {
        placeMarker(lat, lng);
        map.setView([lat, lng], Math.max(map.getZoom(), zoom));
      };

      send({ type: 'ready' });
    }
  </script>
</body>
</html>`;

type FieldMapProps = {
  // the pin the screen currently has (from the map, GPS or typing)
  pin: { latitude: number; longitude: number; source: 'map' | 'gps' | 'typed' } | null;
  onPick: (latitude: number, longitude: number) => void;
  // lets the screen show the coordinate boxes when the map can't load
  onUnavailable: () => void;
};

export function FieldMap({ pin, onPick, onUnavailable }: FieldMapProps) {
  const webViewRef = useRef<WebView>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'failed'>('loading');

  // move the pin when it was set outside the map (GPS or typed). A pin
  // the farmer placed on the map is already there, and re-centring
  // after every tap would make the map jump around.
  useEffect(() => {
    if (status === 'ready' && pin && pin.source !== 'map') {
      webViewRef.current?.injectJavaScript(
        `window.setPin(${pin.latitude}, ${pin.longitude}, ${PIN_ZOOM}); true;`,
      );
    }
  }, [status, pin?.latitude, pin?.longitude, pin?.source]);

  function onMessage(event: WebViewMessageEvent) {
    try {
      const message = JSON.parse(event.nativeEvent.data) as { type: string; lat?: number; lng?: number };
      if (message.type === 'ready') {
        setStatus('ready');
      } else if (message.type === 'failed') {
        setStatus('failed');
        onUnavailable();
      } else if (message.type === 'pick' && typeof message.lat === 'number' && typeof message.lng === 'number') {
        onPick(message.lat, message.lng);
      }
    } catch {
      // ignore anything that isn't one of our messages
    }
  }

  if (status === 'failed') {
    return (
      <View style={[styles.frame, styles.fallback]}>
        <Text style={styles.fallbackText}>
          The map could not be loaded (check the internet connection). Enter the coordinates below instead.
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.frame}>
      <WebView
        ref={webViewRef}
        source={{ html: MAP_HTML, baseUrl: 'https://rubbersense.local/' }}
        originWhitelist={['*']}
        onMessage={onMessage}
        onError={() => {
          setStatus('failed');
          onUnavailable();
        }}
        // lets the map be dragged inside the scrolling screen on Android
        nestedScrollEnabled
        style={styles.webView}
      />
      {status === 'loading' ? (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator color={colors.primary} />
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  frame: {
    height: 300,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
    backgroundColor: colors.background,
  },
  webView: {
    flex: 1,
    backgroundColor: colors.background,
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background,
  },
  fallback: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.md,
  },
  fallbackText: {
    textAlign: 'center',
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 19,
  },
});
