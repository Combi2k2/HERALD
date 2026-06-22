"""Tests for ROI picker HTTP handler."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from herald.ui import build_scene_ui


def test_map_html_contains_leaflet_draw():
    html = build_scene_ui(fallback_lat=48.71, fallback_lon=2.20)
    assert "leaflet.draw" in html
    assert "rectangle" in html
    assert "/status" in html
    assert "geolocation" in html
    assert "/overlay/layers.json" in html
    assert "initPicker" in html
    assert "loadOverlay" in html


def test_roi_post_handler_accepts_vertices():
    received: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode())
            received["vertices"] = data["vertices"]
            body = json.dumps({"ok": True, "message": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    import urllib.request

    payload = json.dumps(
        {"vertices": [[48.71, 2.20], [48.71, 2.21], [48.72, 2.21], [48.72, 2.20]]}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/roi",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200

    server.shutdown()
    assert len(received["vertices"]) == 4
