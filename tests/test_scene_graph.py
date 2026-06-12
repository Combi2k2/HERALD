"""Tests for scene graph serialization."""

import numpy as np

from services.embeddings.encoder import StubEncoder
from herald.scene.common.geometry import Frame, Geometry, Vec3
from herald.scene.common.graph import SceneGraph, SceneNode, SourceRef
from herald.scene.common._legacy_v2 import scene_node_from_v2


def _sample_graph() -> SceneGraph:
    frame = Frame.from_origin(48.71, 2.20)
    graph = SceneGraph(emb_model_id="stub-v0")
    site_ring = [
        (48.71, 2.20),
        (48.712, 2.20),
        (48.712, 2.202),
        (48.71, 2.202),
        (48.71, 2.20),
    ]
    site_coords = np.array(
        [[*frame.wgs2enu(lat, lon), 0.0] for lat, lon in site_ring],
        dtype=np.float64,
    )
    outdoor_coords = np.array(
        [
            [*frame.wgs2enu(lat, lon), 0.0]
            for lat, lon in [
                (48.7105, 2.2005),
                (48.7115, 2.2005),
                (48.7115, 2.2015),
                (48.7105, 2.2015),
                (48.7105, 2.2005),
            ]
        ],
        dtype=np.float64,
    )
    building_coords = np.array(
        [
            [*frame.wgs2enu(lat, lon), 0.0]
            for lat, lon in [
                (48.711, 2.201),
                (48.711, 2.2015),
                (48.7115, 2.2015),
                (48.7115, 2.201),
                (48.711, 2.201),
            ]
        ],
        dtype=np.float64,
    )
    site = SceneNode(
        id="site-uuid",
        type="site",
        pid=None,
        geom=Geometry(type="polygon", coords=site_coords, frame="ENU"),
        desc="test site",
        role="site",
    )
    outdoor = SceneNode(
        id="region-uuid",
        type="region",
        pid="site-uuid",
        geom=Geometry(type="polygon", coords=outdoor_coords, frame="ENU"),
        desc="plaza",
        role="region_use",
        category="plaza",
    )
    building = SceneNode(
        id="structure-uuid",
        type="structure",
        pid="region-uuid",
        geom=Geometry(
            type="polygon",
            coords=building_coords,
            frame="ENU",
            offset=Vec3((0.0, 0.0, 10.0)),
        ),
        refs=[
            SourceRef(
                assigned_by="osm",
                assigned_id="way/1",
                metadata={"building": "yes"},
            )
        ],
        desc="hall",
        role="structure",
        category="building",
    )
    graph.add_node(site)
    graph.add_node(outdoor)
    graph.add_node(building)
    graph.add_edge("site-uuid", "region-uuid")
    graph.add_edge("region-uuid", "structure-uuid")
    return graph


def test_json_round_trip(tmp_path):
    graph = _sample_graph()
    path = tmp_path / "scene_graph.json"
    graph.to_json(path)
    loaded = SceneGraph.from_json(path)
    assert loaded.site_node() is not None
    assert loaded.site_node().id == "site-uuid"
    assert loaded.emb_model_id == "stub-v0"
    assert len(loaded.nodes) == len(graph.nodes)
    assert len(loaded.edges) == len(graph.edges)
    assert loaded.nodes[1].type == "region"
    assert loaded.nodes[2].geom.frame == "ENU"
    assert loaded.nodes[2].geom.offset[2] == 10.0
    assert loaded.nodes[2].refs[0].assigned_id == "way/1"
    assert loaded.edges[0].edge_type == "contains"
    payload = loaded.to_dict()
    assert "site_id" not in payload
    assert "frame_origin" not in payload


def test_v2_node_loads_into_v3_shape():
    legacy = {
        "id": "node_1",
        "type": "building",
        "zone_kind": "building",
        "text": "hall caption",
        "geometry_latlon": [[48.711, 2.201], [48.7115, 2.2015], [48.7115, 2.201]],
        "embedding": [0.1, 0.2],
        "height_m": 12.0,
        "height_source": "default",
        "osm_id": 1,
        "role": "structure",
        "category": "building",
        "function": "academic",
        "name": "Hall A",
        "confidence": 0.95,
        "classification_source": "osm_template",
        "osm_tags": {"building": "university"},
    }
    node = scene_node_from_v2(legacy)
    assert node.type == "structure"
    assert node.name == "Hall A"
    assert node.desc == "hall caption"
    assert node.geom.frame == "WGS"
    assert node.geom.offset[2] == 12.0
    assert node.refs[0].assigned_id == "way/1"
    assert node.refs[0].assigned_by == "osm"
    assert node.txt_embedding_ref is None


def test_stub_encoder_deterministic():
    enc = StubEncoder()
    a = enc.encode(["hello"])
    b = enc.encode(["hello"])
    c = enc.encode(["world"])
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
