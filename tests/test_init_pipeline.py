"""End-to-end tests for scene graph initialization pipeline."""

from services.osm.client import OSMClient, OSMRawPolygon
from services.embeddings.encoder import StubEncoder
from herald.scene.common.geometry import Frame
from herald.scene.init import build_scene_graph
from herald.scene.common.roi import ROI

SAMPLE_OVERPASS_RESPONSE = {
    "elements": [
        {
            "type": "way",
            "id": 1,
            "tags": {"building": "university", "name": "Hall A"},
            "geometry": [
                {"lat": 48.7110, "lon": 2.2010},
                {"lat": 48.7110, "lon": 2.2015},
                {"lat": 48.7115, "lon": 2.2015},
                {"lat": 48.7115, "lon": 2.2010},
                {"lat": 48.7110, "lon": 2.2010},
            ],
        },
        {
            "type": "way",
            "id": 2,
            "tags": {"building": "university", "name": "Hall B"},
            "geometry": [
                {"lat": 48.7116, "lon": 2.2016},
                {"lat": 48.7116, "lon": 2.2020},
                {"lat": 48.7120, "lon": 2.2020},
                {"lat": 48.7120, "lon": 2.2016},
                {"lat": 48.7116, "lon": 2.2016},
            ],
        },
        {
            "type": "way",
            "id": 3,
            "tags": {"highway": "footway"},
            "geometry": [
                {"lat": 48.7105, "lon": 2.2012},
                {"lat": 48.7125, "lon": 2.2012},
            ],
        },
        {
            "type": "way",
            "id": 4,
            "tags": {"highway": "primary", "name": "Road"},
            "geometry": [
                {"lat": 48.7105, "lon": 2.2005},
                {"lat": 48.7125, "lon": 2.2005},
            ],
        },
    ]
}


def _raw_polygon(element: dict) -> OSMRawPolygon:
    tags = {str(k): str(v) for k, v in element["tags"].items()}
    ring = [(float(p["lat"]), float(p["lon"])) for p in element["geometry"]]
    primary = sorted(tags)[0]
    return OSMRawPolygon(
        osm_id=int(element["id"]),
        osm_type="way",
        tags=tags,
        geometry=tuple(ring),
        primary_tag=primary,
        primary_value=tags[primary],
    )


class _MockResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return SAMPLE_OVERPASS_RESPONSE


class _MockSession:
    def post(self, *args, **kwargs):
        return _MockResponse()

    @property
    def headers(self):
        return {}


def _frame_for(roi: ROI) -> Frame:
    lat, lon = roi.latlon_centroid()
    return Frame.from_origin(lat, lon)


def test_build_scene_graph_node_counts():
    roi = ROI.from_polygon(
        [(48.710, 2.200), (48.710, 2.203), (48.713, 2.203), (48.713, 2.200)]
    )
    client = OSMClient(session=_MockSession())
    encoder = StubEncoder()
    raw_polygons = [
        _raw_polygon(el)
        for el in SAMPLE_OVERPASS_RESPONSE["elements"]
        if el["tags"].get("building")
    ]

    result = build_scene_graph(
        roi,
        raw_polygons,
        frame=_frame_for(roi),
        client=client,
        encoder=encoder,
        use_vlm=False,
        show_progress=False,
    )
    graph = result.graph
    site = graph.site_node()

    assert site is not None
    assert result.graph.emb_model_id == "stub-v0"
    assert result.graph.vlm_model_id == "ollama:qwen2.5vl:3b"
    assert len(graph.nodes) == 3
    buildings = [n for n in graph.nodes if n.type == "structure"]
    assert len(buildings) == 2
    assert all(n.geom.type == "polygon" for n in graph.nodes)
    assert all(n.refs for n in buildings)
    assert all(n.txt_embedding_ref is None for n in graph.nodes)
    assert all(n.role == "structure" for n in buildings)
    assert len(result.pathways) == 1


def test_build_scene_graph_containment_edges():
    roi = ROI.from_polygon(
        [(48.710, 2.200), (48.710, 2.203), (48.713, 2.203), (48.713, 2.200)]
    )
    client = OSMClient(session=_MockSession())
    raw_polygons = [
        _raw_polygon(el)
        for el in SAMPLE_OVERPASS_RESPONSE["elements"]
        if el["tags"].get("building")
    ]
    result = build_scene_graph(
        roi,
        raw_polygons,
        frame=_frame_for(roi),
        client=client,
        encoder=StubEncoder(),
        use_vlm=False,
        show_progress=False,
    )
    graph = result.graph
    site = graph.site_node()
    assert site is not None
    contains = [e for e in graph.edges if e.edge_type == "contains"]
    assert len(contains) == 2
    site_children = [e.target_id for e in contains if e.source_id == site.id]
    assert len(site_children) == 2
    for node in graph.nodes:
        if node.pid is not None:
            assert any(
                e.source_id == node.pid and e.target_id == node.id
                for e in contains
            )
