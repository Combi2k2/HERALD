"""Aerial and map-tile imagery for VLM crops."""

from services.aerial.fetch import (
    AerialMeta,
    DEFAULT_ZOOM,
    TILE_SIZE,
    crop_polygon_views,
    fetch_aerial_for_bbox,
    highlight_polygon,
    ring_centroid_latlon,
    ring_to_mosaic_px,
    save_aerial,
    stitch_osm_tiles,
    tile_origin_for_meta,
)

__all__ = [
    "AerialMeta",
    "DEFAULT_ZOOM",
    "TILE_SIZE",
    "crop_polygon_views",
    "fetch_aerial_for_bbox",
    "highlight_polygon",
    "ring_centroid_latlon",
    "ring_to_mosaic_px",
    "save_aerial",
    "stitch_osm_tiles",
    "tile_origin_for_meta",
]
