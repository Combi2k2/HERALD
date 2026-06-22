"""Unit tests for OSM Overpass client (mocked HTTP)."""

import requests

from services.osm import (
    OSMClient,
    _build_overpass_polygon_query,
    _infer_geometry_kind,
    _is_retryable_http_error,
    _parse_raw_polygon,
    _ring_is_closed,
)


SAMPLE_OVERPASS_RESPONSE = {
    "elements": [
        {
            "type": "way",
            "id": 1,
            "tags": {"building": "university", "name": "Hall A"},
            "geometry": [
                {"lat": 48.71, "lon": 2.20},
                {"lat": 48.71, "lon": 2.21},
                {"lat": 48.72, "lon": 2.21},
                {"lat": 48.72, "lon": 2.20},
                {"lat": 48.71, "lon": 2.20},
            ],
        },
        {
            "type": "way",
            "id": 2,
            "tags": {"highway": "footway", "name": "Path"},
            "geometry": [
                {"lat": 48.71, "lon": 2.20},
                {"lat": 48.715, "lon": 2.205},
            ],
        },
    ]
}


class _MockResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return SAMPLE_OVERPASS_RESPONSE


class _MockSession:
    last_query: str | None = None

    def post(self, *args, **kwargs):
        _MockSession.last_query = kwargs.get("data", b"").decode("utf-8")
        return _MockResponse()

    @property
    def headers(self):
        return {}


def test_parse_building_polygon_and_highway_line():
    client = OSMClient(session=_MockSession())
    result = client._parse_response(48.715, 2.205, 150.0, SAMPLE_OVERPASS_RESPONSE)

    assert len(result.buildings) == 1
    assert len(result.highways) == 1
    assert result.buildings[0].kind == "polygon"
    assert result.buildings[0].name == "Hall A"
    assert result.highways[0].kind == "linestring"
    assert result.highways[0].tags["highway"] == "footway"


def test_ring_closed_detection():
    ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    assert _ring_is_closed(ring) is True
    open_way = [(0.0, 0.0), (1.0, 0.0)]
    assert _ring_is_closed(open_way) is False


SAMPLE_LANDUSE_POLYGON = {
    "type": "way",
    "id": 99,
    "tags": {"landuse": "grass", "name": "Quad"},
    "geometry": [
        {"lat": 48.7112, "lon": 2.2011},
        {"lat": 48.7112, "lon": 2.2014},
        {"lat": 48.7114, "lon": 2.2014},
        {"lat": 48.7114, "lon": 2.2011},
        {"lat": 48.7112, "lon": 2.2011},
    ],
}


def test_parse_raw_polygon_landuse():
    feature = _parse_raw_polygon(SAMPLE_LANDUSE_POLYGON)
    assert feature is not None
    assert feature.primary_tag == "landuse"
    assert feature.primary_value == "grass"


def test_query_raw_polygons_deduplicates():
    payload = {
        "elements": [
            SAMPLE_OVERPASS_RESPONSE["elements"][0],
            SAMPLE_LANDUSE_POLYGON,
            SAMPLE_OVERPASS_RESPONSE["elements"][1],
        ]
    }

    class _RawMockSession:
        def post(self, *args, **kwargs):
            class R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return payload

            return R()

        @property
        def headers(self):
            return {}

    client = OSMClient(session=_RawMockSession())
    verts = [(48.71, 2.20), (48.71, 2.21), (48.72, 2.21), (48.72, 2.20)]
    result = client.query_raw_polygons_in_polygon(verts)
    assert len(result.polygons) == 2
    tags = {p.primary_tag for p in result.polygons}
    assert "building" in tags
    assert "landuse" in tags
    assert len(result.highways) == 1
    assert result.highways[0].tags["highway"] == "footway"


def test_build_overpass_raw_polygons_query_is_unfiltered():
    verts = [(48.71, 2.20), (48.71, 2.21), (48.72, 2.21), (48.72, 2.20)]
    from services.osm import (
        RAW_POLYGON_OVERPASS_TIMEOUT_S,
        _build_overpass_raw_polygons_query,
    )

    query = _build_overpass_raw_polygons_query(verts)
    # Unfiltered single poly scan: fetches all features (incl. untagged
    # polygons) and is faster than a per-tag union.
    assert 'way(poly:"' in query
    assert 'relation(poly:"' in query
    assert 'way["building"]' not in query
    assert f"[timeout:{RAW_POLYGON_OVERPASS_TIMEOUT_S}]" in query


def test_build_overpass_polygon_query_contains_poly_clause():
    verts = [(48.71, 2.20), (48.71, 2.21), (48.72, 2.21), (48.72, 2.20)]
    query = _build_overpass_polygon_query(verts)
    assert 'poly:"48.71 2.2 48.71 2.21 48.72 2.21 48.72 2.2"' in query
    assert 'way["building"](poly:' in query


def test_query_in_polygon_posts_poly_query():
    _MockSession.last_query = None
    client = OSMClient(session=_MockSession())
    verts = [(48.71, 2.20), (48.71, 2.21), (48.72, 2.21), (48.72, 2.20)]
    result = client.query_in_polygon(verts)
    assert _MockSession.last_query is not None
    assert "poly:" in _MockSession.last_query
    assert len(result.buildings) == 1


def test_is_retryable_http_error():
    resp = requests.Response()
    resp.status_code = 504
    assert _is_retryable_http_error(requests.HTTPError(response=resp)) is True
    resp.status_code = 404
    assert _is_retryable_http_error(requests.HTTPError(response=resp)) is False
    # Timeouts / connection errors fail over to the next endpoint rather than
    # retrying the same (likely dead) host in place.
    assert _is_retryable_http_error(requests.Timeout()) is False
    assert _is_retryable_http_error(requests.ConnectionError()) is False


def test_post_overpass_retries_504_then_succeeds(monkeypatch):
    calls: list[int] = []

    class _FlakySession:
        @property
        def headers(self):
            return {}

        def post(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                resp = requests.Response()
                resp.status_code = 504
                raise requests.HTTPError(response=resp)
            return _MockResponse()

    sleeps: list[float] = []
    monkeypatch.setattr("services.osm.time.sleep", lambda s: sleeps.append(s))

    client = OSMClient(
        overpass_urls=("https://example.test/interpreter",),
        session=_FlakySession(),
    )
    payload = client._post_overpass("[out:json];node(1);out;")
    assert payload == SAMPLE_OVERPASS_RESPONSE
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_post_overpass_raises_after_all_retries(monkeypatch):
    class _Always504Session:
        @property
        def headers(self):
            return {}

        def post(self, *args, **kwargs):
            resp = requests.Response()
            resp.status_code = 504
            raise requests.HTTPError(response=resp)

    monkeypatch.setattr("services.osm.time.sleep", lambda _s: None)

    client = OSMClient(
        overpass_urls=("https://a.test/interpreter", "https://b.test/interpreter"),
        session=_Always504Session(),
    )
    try:
        client._post_overpass("[out:json];node(1);out;")
    except RuntimeError as exc:
        assert "All Overpass endpoints failed after retries" in str(exc)
        assert "https://a.test/interpreter" in str(exc)
        assert "https://b.test/interpreter" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_area_highway_treated_as_polygon():
    element = {
        "type": "way",
        "tags": {"highway": "pedestrian", "area:highway": "pedestrian"},
        "geometry": [
            {"lat": 0, "lon": 0},
            {"lat": 0, "lon": 1},
            {"lat": 1, "lon": 1},
            {"lat": 1, "lon": 0},
            {"lat": 0, "lon": 0},
        ],
    }
    verts = [(p["lat"], p["lon"]) for p in element["geometry"]]
    assert _infer_geometry_kind(element, verts) == "polygon"
