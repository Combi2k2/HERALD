"""Folium map overlay and live Leaflet UI server for layout validation."""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from herald.browser import open_browser as launch_browser
from herald.scene.common.geometry import Frame
from herald.scene.common.graph import SceneGraph
from herald.scene.common.roi import ROI
from herald.ui.roi_picker import ROIPickerError

# Hex colors keyed by common OSM primary tags (matches Rerun palette loosely)
_TAG_COLORS: dict[str, str] = {
    "building": "#60a5fa",
    "landuse": "#4ade80",
    "leisure": "#c084fc",
    "amenity": "#f472b6",
    "natural": "#34d399",
    "water": "#22d3ee",
    "waterway": "#06b6d4",
    "highway": "#fb923c",
    "place": "#facc15",
    "man_made": "#94a3b8",
    "barrier": "#ef4444",
    "untagged": "#e5e7eb",
}

_FOLIUM_STYLE_KEYS = {
    "color": "color",
    "weight": "weight",
    "fill": "fill",
    "fillColor": "fill_color",
    "fillOpacity": "fill_opacity",
    "dashArray": "dash_array",
    "opacity": "opacity",
}


def _popup_html(title: str, props: dict) -> str:
    lines = [f"<b>{html.escape(title)}</b>"]
    for key in ("name", "primary_tag", "primary_value", "id"):
        if key in props and props[key]:
            lines.append(f"{key}: {html.escape(str(props[key]))}")
    body = props.get("tags") if isinstance(props.get("tags"), dict) else props
    if isinstance(body, dict):
        preview = json.dumps(body, indent=2)[:600]
        lines.append(f"<pre>{html.escape(preview)}</pre>")
    return "<br>".join(lines)


def _ring_to_geojson_polygon(ring: list[tuple[float, float]]) -> list[list[float]]:
    coords = [[lon, lat] for lat, lon in ring]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _geojson_polygon_feature(
    ring: list[tuple[float, float]],
    *,
    popup: str,
    properties: dict | None = None,
) -> dict:
    props = dict(properties or {})
    props["popup"] = popup
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {
            "type": "Polygon",
            "coordinates": [_ring_to_geojson_polygon(ring)],
        },
    }


def _write_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    body: bytes,
    *,
    content_type: str,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    _write_response(
        handler,
        status,
        json.dumps(payload).encode(),
        content_type="application/json",
    )


def _overlay_layers_payload(
    *,
    roi: ROI,
    osm_polygons_geojson: dict,
    frame: Frame | None = None,
    graph: SceneGraph | None = None,
    pathways_geojson: dict | None = None,
) -> dict:
    """Layer payload for the live Leaflet UI (`GET /overlay/layers.json`)."""
    lat, lon = roi.latlon_centroid()
    layers: list[dict] = [
        {
            "name": "ROI boundary",
            "show": True,
            "style": {
                "color": "#ffffff",
                "weight": 3,
                "fill": False,
                "dashArray": "8 6",
            },
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    _geojson_polygon_feature(
                        roi.latlon_vertices(),
                        popup=f"ROI area {roi.area() / 1e6:.3f} km²",
                    )
                ],
            },
        },
    ]

    osm_features = []
    for feature in osm_polygons_geojson.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        props = dict(feature.get("properties") or {})
        tag = str(props.get("primary_tag", "other"))
        name = props.get("name") or feature.get("id", "polygon")
        props["popup"] = _popup_html(str(name), {**props, "id": feature.get("id")})
        props["primary_tag"] = tag
        osm_features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": geom,
            }
        )
    if osm_features:
        layers.append(
            {
                "name": "OSM polygons (all)",
                "show": True,
                "style": {
                    "weight": 1,
                    "fill": True,
                    "fillOpacity": 0.35,
                },
                "tag_colors": _TAG_COLORS,
                "geojson": {"type": "FeatureCollection", "features": osm_features},
            }
        )

    if graph is not None:
        outdoor_features = []
        building_features = []
        for node in graph.nodes:
            popup = html.escape(node.desc or node.name or node.id)
            if node.type == "region":
                if node.geom.frame == "ENU":
                    if frame is None:
                        raise ValueError("frame is required to map ENU scene graph geometry")
                    ring = [
                        frame.enu2wgs(float(row[0]), float(row[1]))
                        for row in node.geom.coords
                    ]
                else:
                    ring = [(float(row[0]), float(row[1])) for row in node.geom.coords]
                outdoor_features.append(
                    _geojson_polygon_feature(ring, popup=popup)
                )
            elif node.type == "structure":
                if node.geom.frame == "ENU":
                    if frame is None:
                        raise ValueError("frame is required to map ENU scene graph geometry")
                    ring = [
                        frame.enu2wgs(float(row[0]), float(row[1]))
                        for row in node.geom.coords
                    ]
                else:
                    ring = [(float(row[0]), float(row[1])) for row in node.geom.coords]
                building_features.append(
                    _geojson_polygon_feature(ring, popup=popup)
                )
        if outdoor_features:
            layers.append(
                {
                    "name": "Scene graph — outdoor zones",
                    "show": True,
                    "style": {
                        "color": "#38bdf8",
                        "weight": 2,
                        "fill": False,
                        "dashArray": "4 4",
                    },
                    "geojson": {
                        "type": "FeatureCollection",
                        "features": outdoor_features,
                    },
                }
            )
        if building_features:
            layers.append(
                {
                    "name": "Scene graph — buildings",
                    "show": True,
                    "style": {
                        "color": "#1d4ed8",
                        "weight": 2,
                        "fill": True,
                        "fillColor": "#1d4ed8",
                        "fillOpacity": 0.25,
                    },
                    "geojson": {
                        "type": "FeatureCollection",
                        "features": building_features,
                    },
                }
            )

    if pathways_geojson is not None:
        path_features = []
        for feature in pathways_geojson.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") != "LineString":
                continue
            props = dict(feature.get("properties") or {})
            props["popup"] = html.escape(
                props.get("name") or props.get("highway") or "pathway"
            )
            path_features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": geom,
                }
            )
        if path_features:
            layers.append(
                {
                    "name": "Walkable pathways",
                    "show": True,
                    "style": {
                        "color": "#ea580c",
                        "weight": 3,
                        "opacity": 0.9,
                    },
                    "geojson": {
                        "type": "FeatureCollection",
                        "features": path_features,
                    },
                }
            )

    return {"center": {"lat": lat, "lon": lon}, "zoom": 16, "layers": layers}


def _folium_kwargs(layer_def: dict, feature: dict) -> dict[str, Any]:
    """Translate layer payload style into Folium keyword arguments."""
    style = dict(layer_def.get("style") or {})
    props = feature.get("properties") or {}
    tag = props.get("primary_tag")
    tag_colors = layer_def.get("tag_colors")
    if tag and tag_colors and tag in tag_colors:
        color = tag_colors[tag]
        style.setdefault("color", color)
        style.setdefault("fillColor", color)
    return {
        folium_key: style[src]
        for src, folium_key in _FOLIUM_STYLE_KEYS.items()
        if src in style
    }


def _write_overlay_map_html(
    out_html: Path,
    *,
    roi: ROI,
    osm_polygons_geojson: dict,
    frame: Frame | None = None,
    graph: SceneGraph | None = None,
    pathways_geojson: dict | None = None,
) -> None:
    """Write a self-contained Folium HTML map with toggleable overlay layers."""
    import folium
    from folium import FeatureGroup, plugins

    payload = _overlay_layers_payload(
        roi=roi,
        osm_polygons_geojson=osm_polygons_geojson,
        frame=frame,
        graph=graph,
        pathways_geojson=pathways_geojson,
    )
    center = payload["center"]
    m = folium.Map(
        location=[center["lat"], center["lon"]],
        zoom_start=payload.get("zoom", 16),
        tiles="OpenStreetMap",
    )

    for layer_def in payload["layers"]:
        fg = FeatureGroup(name=layer_def["name"], show=layer_def.get("show", True))
        for feature in layer_def["geojson"].get("features", []):
            geom = feature.get("geometry", {})
            props = feature.get("properties") or {}
            popup_html = props.get("popup")
            popup = (
                folium.Popup(popup_html, max_width=360) if popup_html else None
            )
            fstyle = _folium_kwargs(layer_def, feature)
            geom_type = geom.get("type")

            if geom_type == "Polygon":
                coords = [(c[1], c[0]) for c in geom["coordinates"][0]]
                folium.Polygon(locations=coords, popup=popup, **fstyle).add_to(fg)
            elif geom_type == "LineString":
                coords = [(c[1], c[0]) for c in geom["coordinates"]]
                folium.PolyLine(
                    locations=coords,
                    popup=popup_html,
                    **fstyle,
                ).add_to(fg)
        fg.add_to(m)

    plugins.MeasureControl(position="topleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def build_map_overlay_ui() -> str:
    """Return Leaflet script for loading overlay layers from ``/overlay/layers.json``."""
    return """
    let overlayLoaded = false;

    function styleForLayer(layerDef, feature) {
      const base = Object.assign({}, layerDef.style || {});
      const tag = feature.properties && feature.properties.primary_tag;
      if (tag && layerDef.tag_colors && layerDef.tag_colors[tag]) {
        const color = layerDef.tag_colors[tag];
        base.color = color;
        base.fillColor = color;
      }
      return base;
    }

    async function loadOverlay() {
      if (overlayLoaded) return;
      const resp = await fetch('/overlay/layers.json');
      if (!resp.ok) return;
      const data = await resp.json();
      overlayLoaded = true;

      if (typeof drawControl !== 'undefined' && drawControl) {
        map.removeControl(drawControl);
        drawControl = null;
      }
      if (typeof drawn !== 'undefined') {
        drawn.clearLayers();
      }

      if (data.center) {
        map.setView([data.center.lat, data.center.lon], data.zoom || 16);
      }

      const overlays = {};
      for (const layerDef of data.layers || []) {
        const group = L.geoJSON(layerDef.geojson, {
          style: (feature) => styleForLayer(layerDef, feature),
          onEachFeature: (feature, layer) => {
            const popup = feature.properties && feature.properties.popup;
            if (popup) layer.bindPopup(popup);
          }
        });
        overlays[layerDef.name] = group;
        if (layerDef.show !== false) {
          group.addTo(map);
        }
      }

      L.control.layers(null, overlays, { collapsed: false }).addTo(map);
      L.control.scale({ imperial: false }).addTo(map);
      setBanner(
        'HERALD layout check — toggle layers (top-right). OSM tiles = ground truth underneath.'
      );
    }

    function startStatusPoll() {
      const poll = setInterval(() => {
        fetch('/status')
          .then(r => r.json())
          .then(st => {
            if (st.message) {
              setBanner('<span style="color:#1e3a8a">' + st.message + '</span>');
            }
            if (st.ready) {
              clearInterval(poll);
              loadOverlay();
            }
          })
          .catch(() => {});
      }, 800);
    }

    fetch('/status')
      .then(r => r.json())
      .then(st => {
        if (st.ready) {
          loadOverlay();
        } else if (st.message) {
          setBanner(st.message);
        }
      })
      .catch(() => {});
    """


class SceneOverlayServer:
    """Single HTTP server: ROI picker → processing status → map overlay."""

    def __init__(self, out_dir: Path, port: int) -> None:
        self.out_dir = out_dir.resolve()
        self._port = port
        self._overlay_ready = False
        self._roi_done = threading.Event()
        self._roi_vertices: list[tuple[float, float]] | None = None
        self._fallback_lat: float | None = None
        self._fallback_lon: float | None = None
        self._overlay_layers: dict | None = None
        self._status_message = "Draw a rectangle over your site, then confirm."
        self._httpd: ThreadingHTTPServer | None = None
        self._url = ""

    @property
    def url(self) -> str:
        return self._url

    def start(self) -> str:
        """Start the server and return the base URL."""
        from herald.ui import build_scene_ui

        out_dir = self.out_dir
        server_ref = self

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(out_dir), **kwargs)

            def log_message(self, *_args) -> None:
                return

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("", "/"):
                    body = build_scene_ui(
                        fallback_lat=server_ref._fallback_lat,
                        fallback_lon=server_ref._fallback_lon,
                    ).encode()
                    _write_response(
                        self,
                        200,
                        body,
                        content_type="text/html; charset=utf-8",
                    )
                    return
                if path == "/status":
                    _write_json(
                        self,
                        200,
                        {
                            "ready": server_ref._overlay_ready,
                            "message": server_ref._status_message,
                        },
                    )
                    return
                if path == "/overlay/layers.json":
                    if (
                        not server_ref._overlay_ready
                        or server_ref._overlay_layers is None
                    ):
                        self.send_error(404)
                        return
                    _write_json(self, 200, server_ref._overlay_layers)
                    return
                super().do_GET()

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] != "/roi":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    data = json.loads(raw.decode("utf-8"))
                    verts = data["vertices"]
                    server_ref._roi_vertices = [
                        (float(v[0]), float(v[1])) for v in verts
                    ]
                except (KeyError, ValueError, json.JSONDecodeError):
                    self.send_error(400)
                    return
                server_ref._status_message = "ROI received — fetching OSM data…"
                _write_json(
                    self,
                    200,
                    {"ok": True, "message": "ROI received — building scene…"},
                )
                server_ref._roi_done.set()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._httpd = ThreadingHTTPServer(("127.0.0.1", self._port), Handler)
        except OSError as exc:
            raise OSError(
                f"Could not bind scene UI server to 127.0.0.1:{self._port} "
                f"(try --port). {exc}"
            ) from exc
        self._url = f"http://127.0.0.1:{self._port}"
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()
        print(f"Scene UI server -> {self._url}/")
        return self._url

    def set_status(self, message: str) -> None:
        self._status_message = message

    def set_overlay_from_scene(
        self,
        *,
        roi: ROI,
        frame: Frame,
        osm_polygons_geojson: dict,
        graph: SceneGraph,
        pathways_geojson: dict,
    ) -> None:
        """Push overlay layer JSON derived from processed scene artifacts."""
        self._overlay_layers = _overlay_layers_payload(
            roi=roi,
            osm_polygons_geojson=osm_polygons_geojson,
            frame=frame,
            graph=graph,
            pathways_geojson=pathways_geojson,
        )
        self._overlay_ready = True
        self._status_message = "Layout ready — loading overlay…"

    def prompt_roi_rectangle(
        self,
        *,
        fallback_lat: float | None = None,
        fallback_lon: float | None = None,
        timeout_s: float = 300.0,
        open_browser: bool = True,
    ) -> ROI:
        """Open the picker in the browser and block until the user submits a ROI."""
        self._fallback_lat = fallback_lat
        self._fallback_lon = fallback_lon
        print(
            f"Draw ROI rectangle at {self._url}/ (timeout {timeout_s:.0f}s) "
            "— allow browser location when prompted"
        )
        if open_browser:
            launch_browser(f"{self._url}/")
        if not self._roi_done.wait(timeout=timeout_s):
            self.shutdown()
            raise ROIPickerError(
                "ROI picker timed out. Draw a rectangle on the map or pass --bbox."
            )
        assert self._roi_vertices is not None
        return ROI.from_polygon(self._roi_vertices)

    def block_until_interrupt(self) -> None:
        print("Map server running — Ctrl+C to exit.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print()
        self.shutdown()

    def shutdown(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


__all__ = ["SceneOverlayServer", "build_map_overlay_ui"]
