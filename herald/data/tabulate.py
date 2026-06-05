"""Tabulate raw ``osm.geojson`` features (tags only, no geometry)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

TABLE_PREFIX_COLUMNS = ("osm_type", "osm_id", "name", "primary_tag", "primary_value")


def osm_features_to_rows(geojson: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten FeatureCollection properties into one dict per feature."""
    rows: list[dict[str, str]] = []
    for feat in geojson.get("features", []):
        row: dict[str, str] = {}
        fid = str(feat.get("id", ""))
        if "/" in fid:
            osm_type, _, osm_id = fid.partition("/")
            row["osm_type"] = osm_type
            row["osm_id"] = osm_id
        for key, value in (feat.get("properties") or {}).items():
            row[str(key)] = "" if value is None else str(value)
        rows.append(row)
    return rows


def table_columns(rows: list[dict[str, str]]) -> list[str]:
    """Stable column order: prefix keys first, then remaining tags sorted."""
    keys: set[str] = set()
    for row in rows:
        keys.update(row)
    rest = sorted(k for k in keys if k not in TABLE_PREFIX_COLUMNS)
    return [c for c in TABLE_PREFIX_COLUMNS if c in keys] + rest


def write_osm_table(rows: list[dict[str, str]], path: Path | str) -> None:
    """Write rows to CSV (no geometry)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = table_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_osm_geojson(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
