"""Tests for scene graph serialization."""

import numpy as np

from services.osm.client import LatLon
from services.embeddings.encoder import StubEncoder
from herald.scene.common.frame import LocalFrame
from herald.scene.common.graph import SceneEdge, SceneGraph, SceneNode


def _sample_graph() -> SceneGraph:
    frame = LocalFrame(origin=LatLon(lat=48.71, lon=2.20))
    enc = StubEncoder()
    emb = enc.encode(["site"])[0].tolist()
    graph = SceneGraph(
        site_id="site_000",
        frame=frame,
        embedding_model_id=enc.model_id,
        roi_area_m2=10000.0,
    )
    site = SceneNode(
        id="site_000",
        level="site",
        zone_kind=None,
        text="test site",
        geometry_latlon=[(48.71, 2.20), (48.712, 2.20), (48.712, 2.202), (48.71, 2.202), (48.71, 2.20)],
        embedding=emb,
    )
    outdoor = SceneNode(
        id="outdoor_000",
        level="outdoor_region",
        zone_kind="outdoor_region",
        text="plaza",
        geometry_latlon=[(48.7105, 2.2005), (48.7115, 2.2005), (48.7115, 2.2015), (48.7105, 2.2015), (48.7105, 2.2005)],
        embedding=enc.encode(["plaza"])[0].tolist(),
    )
    building = SceneNode(
        id="building_1",
        level="building",
        zone_kind="building",
        text="hall",
        geometry_latlon=[(48.711, 2.201), (48.711, 2.2015), (48.7115, 2.2015), (48.7115, 2.201), (48.711, 2.201)],
        embedding=enc.encode(["hall"])[0].tolist(),
        osm_id=1,
    )
    graph.add_node(site)
    graph.add_node(outdoor)
    graph.add_node(building)
    graph.add_edge("site_000", "outdoor_000")
    graph.add_edge("outdoor_000", "building_1")
    return graph


def test_json_round_trip(tmp_path):
    graph = _sample_graph()
    path = tmp_path / "scene_graph.json"
    graph.to_json(path)
    loaded = SceneGraph.from_json(path)
    assert loaded.site_id == graph.site_id
    assert len(loaded.nodes) == len(graph.nodes)
    assert len(loaded.edges) == len(graph.edges)
    assert loaded.nodes[1].zone_kind == "outdoor_region"
    assert loaded.edges[0].edge_type == "contains"


def test_stub_encoder_deterministic():
    enc = StubEncoder()
    a = enc.encode(["hello"])
    b = enc.encode(["hello"])
    c = enc.encode(["world"])
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_no_point_cloud_in_payload():
    graph = _sample_graph()
    graph.assert_no_point_cloud_payload()
