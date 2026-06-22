"""Operator UI for ROI selection and map validation."""

from herald.ui.map_overlay import SceneOverlayServer, build_map_overlay_ui
from herald.ui.rerun import render_scene
from herald.ui.roi_picker import ROIPickerError, _build_roi_picker_ui

__all__ = [
    "ROIPickerError",
    "SceneOverlayServer",
    "build_scene_ui",
    "render_scene",
]


def build_scene_ui(
    *,
    fallback_lat: float | None = None,
    fallback_lon: float | None = None,
) -> str:
    """Single-page Leaflet UI: ROI picker → layout overlay on the same map."""
    fb_lat = "null" if fallback_lat is None else repr(fallback_lat)
    fb_lon = "null" if fallback_lon is None else repr(fallback_lon)
    picker_js = _build_roi_picker_ui()
    overlay_js = build_map_overlay_ui()
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>HERALD — scene map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    #banner {{
      position: absolute; top: 10px; left: 50px; z-index: 1000;
      background: white; padding: 8px 12px; border-radius: 6px;
      font-family: system-ui, sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,.2);
    }}
  </style>
</head>
<body>
  <div id="banner">Allow browser location access, then draw a <b>rectangle</b> over your site.</div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
  <script>
    const fallbackLat = {fb_lat};
    const fallbackLon = {fb_lon};
    const hasFallback = fallbackLat !== null && fallbackLon !== null;
    const map = L.map('map').setView(
      hasFallback ? [fallbackLat, fallbackLon] : [20, 0],
      hasFallback ? 17 : 2
    );
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    function setBanner(html) {{
      document.getElementById('banner').innerHTML = html;
    }}

    {picker_js}
    {overlay_js}

    initPicker();
    initGeolocation();
  </script>
</body>
</html>
"""
