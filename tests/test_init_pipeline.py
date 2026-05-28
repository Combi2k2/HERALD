"""End-to-end tests for scene graph initialization pipeline."""

from herald.osm.client import OSMClient
from herald.scene.embedding import StubEncoder
from herald.scene.init import build_scene_graph
from herald.scene.roi import ROI

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


def test_build_scene_graph_node_counts():
    roi = ROI.from_bbox(48.710, 2.200, 48.713, 2.203)
    client = OSMClient(session=_MockSession())
    encoder = StubEncoder()
    events: list[str] = []

    def on_event(ev):
        events.append(ev.kind)

    result = build_scene_graph(roi, client=client, encoder=encoder, on_event=on_event)
    graph = result.graph

    assert graph.site_id == "site_000"
    assert graph.embedding_model_id == "stub-v0"
    assert len(graph.nodes) >= 3  # site + outdoor + 2 buildings
    buildings = [n for n in graph.nodes if n.level == "building"]
    assert len(buildings) == 2
    assert all(n.embedding is not None for n in graph.nodes)
    assert "pipeline_complete" in events
    assert len(result.pathways) == 1  # footway only


def test_build_scene_graph_containment_edges():
    roi = ROI.from_bbox(48.710, 2.200, 48.713, 2.203)
    client = OSMClient(session=_MockSession())
    result = build_scene_graph(roi, client=client, encoder=StubEncoder())
    graph = result.graph
    contains = [e for e in graph.edges if e.edge_type == "contains"]
    assert len(contains) >= 2
    site_children = [e.target_id for e in contains if e.source_id == "site_000"]
    assert len(site_children) >= 1
