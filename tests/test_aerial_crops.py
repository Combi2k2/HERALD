"""Tests for aerial/OSM polygon crop helpers."""

import math

from PIL import Image

from services.aerial.fetch import (
    TILE_SIZE,
    AerialMeta,
    _latlon_to_tile_xy,
    _tile_xy_to_latlon,
    crop_polygon_views,
    highlight_polygon,
    ring_to_mosaic_px,
)


def _meta_for_bbox(
    south: float, west: float, north: float, east: float, *, zoom: int = 18
) -> AerialMeta:
    x_min, y_max = _latlon_to_tile_xy(south, west, zoom)
    x_max, y_min = _latlon_to_tile_xy(north, east, zoom)
    tile_x0 = int(math.floor(x_min))
    tile_y0 = int(math.floor(y_min))
    tile_x1 = int(math.floor(x_max))
    tile_y1 = int(math.floor(y_max))
    nw_lat, nw_lon = _tile_xy_to_latlon(tile_x0, tile_y0, zoom)
    se_lat, se_lon = _tile_xy_to_latlon(tile_x1 + 1, tile_y1 + 1, zoom)
    return AerialMeta(
        bbox=(se_lat, nw_lon, nw_lat, se_lon),
        zoom=zoom,
        width_px=(tile_x1 - tile_x0 + 1) * TILE_SIZE,
        height_px=(tile_y1 - tile_y0 + 1) * TILE_SIZE,
        provider="test",
    )


def test_ring_to_mosaic_px_non_empty():
    meta = _meta_for_bbox(48.70, 2.20, 48.72, 2.22)
    ring = [(48.715, 2.210), (48.715, 2.211), (48.716, 2.211), (48.716, 2.210), (48.715, 2.210)]
    pts = ring_to_mosaic_px(ring, meta)
    assert len(pts) == 4
    assert all(0 <= x <= meta.width_px for x, _ in pts)
    assert all(0 <= y <= meta.height_px for _, y in pts)


def test_crop_polygon_views_aerial_only_shape():
    meta = _meta_for_bbox(48.70, 2.20, 48.72, 2.22)
    mosaic = Image.new("RGB", (meta.width_px, meta.height_px), color=(80, 80, 80))
    ring = [(48.715, 2.210), (48.715, 2.211), (48.716, 2.211), (48.716, 2.210), (48.715, 2.210)]

    class _FakeResp:
        content = b""

        def raise_for_status(self):
            return None

    class _FakeSession:
        def get(self, *args, **kwargs):
            # Return a minimal valid PNG tile
            from io import BytesIO

            tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), color=(100, 100, 100))
            buf = BytesIO()
            tile.save(buf, format="PNG")
            resp = _FakeResp()
            resp.content = buf.getvalue()
            return resp

    aerial_crop, osm_crop, local_poly = crop_polygon_views(
        mosaic,
        meta,
        ring,
        session=_FakeSession(),
    )
    assert aerial_crop.size == osm_crop.size
    assert len(local_poly) >= 3
    highlighted = highlight_polygon(aerial_crop, local_poly)
    assert highlighted.size == aerial_crop.size
