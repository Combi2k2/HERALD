"""Tests for osm.geojson tabulation."""

import csv

from herald.data.tabulate import osm_features_to_rows, table_columns, write_osm_table


def test_osm_features_to_rows_drops_geometry():
    geojson = {
        "features": [
            {
                "type": "Feature",
                "id": "way/42",
                "properties": {
                    "building": "yes",
                    "name": "Hall A",
                    "primary_tag": "building",
                    "primary_value": "yes",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
            }
        ]
    }
    rows = osm_features_to_rows(geojson)
    assert len(rows) == 1
    row = rows[0]
    assert row["osm_type"] == "way"
    assert row["osm_id"] == "42"
    assert row["name"] == "Hall A"
    assert row["building"] == "yes"
    assert "geometry" not in row
    assert "coordinates" not in row


def test_write_osm_table(tmp_path):
    rows = [
        {"osm_type": "way", "osm_id": "1", "landuse": "forest", "name": "Woods"},
        {"osm_type": "way", "osm_id": "2", "building": "yes"},
    ]
    out = tmp_path / "osm.csv"
    write_osm_table(rows, out)
    with out.open(encoding="utf-8") as f:
        parsed = list(csv.DictReader(f))
    assert len(parsed) == 2
    assert set(parsed[0]) == set(table_columns(rows))
    assert parsed[0]["landuse"] == "forest"
