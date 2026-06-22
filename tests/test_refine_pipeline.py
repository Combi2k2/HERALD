"""GPU-free tests for Phase-2 RGB-stream refinement.

A ``FakeBackend`` stands in for VGGT (mirroring how ``StubEncoder`` stands in for
a real encoder), feeding a synthetic frontal-plane scene with known intrinsics,
poses, and depth. Tests cover fusion geometry, Umeyama registration, graph
integration, JSON round-trip, the schema guard (no point arrays in the graph),
and sidecar persistence.
"""

from __future__ import annotations

import json

import numpy as np

from herald.data.paths import RunPaths
from herald.scene.common.geometry import Frame, Geometry
from herald.scene.common.graph import SceneGraph, SceneNode
from herald.scene.common.nav import NavGraph
from herald.scene.common.repr import SceneRepr
from herald.scene.refine import (
    Stream,
    fuse_geometries,
    refine_scene_graph,
    register_to_enu,
    umeyama,
)
from herald.scene.refine.io import load_refine_metadata
from services.reconstruction.base import FrameGeometry


class FakeBackend:
    """Returns a precomputed list of FrameGeometry, ignoring the input frames."""

    model_id = "fake"

    def __init__(self, geoms: list[FrameGeometry]) -> None:
        self.geoms = geoms

    def infer(self, frames):  # noqa: ANN001 - matches GeometryBackend protocol
        return self.geoms


def _intrinsics(h: int, w: int, f: float = 80.0) -> np.ndarray:
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def _make_geoms(n: int = 3, h: int = 48, w: int = 48, z: float = 3.0) -> list[FrameGeometry]:
    k = _intrinsics(h, w)
    depth = np.full((h, w), z, np.float32)
    geoms = []
    for i in range(n):
        c2w = np.eye(4)
        c2w[0, 3] = 0.05 * i  # small lateral camera motion
        geoms.append(FrameGeometry(i, k, c2w, depth, np.ones((h, w), np.float32)))
    return geoms


def _graph_with_region() -> SceneGraph:
    g = SceneGraph(emb_model_id="stub-v0")
    g.add_node(
        SceneNode(id="site_000", pid=None, type="site",
                  geom=Geometry(type="polygon", frame="ENU",
                                coords=[[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]]))
    )
    g.add_node(
        SceneNode(id="region_a", pid="site_000", type="region",
                  geom=Geometry(type="polygon", frame="ENU",
                                coords=[[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]]))
    )
    return g


def _stream(n: int = 3, h: int = 48, w: int = 48, reference=None) -> Stream:
    imgs = [np.zeros((h, w, 3), np.uint8) for _ in range(n)]
    return Stream.from_arrays(imgs, reference_positions=reference)


# --- registration ---------------------------------------------------------

def test_umeyama_recovers_known_transform():
    rng = np.random.default_rng(0)
    src = rng.standard_normal((10, 3))
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # 90deg about z
    dst = 2.5 * (src @ r.T) + np.array([4.0, -1.0, 7.0])
    sim = umeyama(src, dst)
    assert sim.registered
    assert abs(sim.scale - 2.5) < 1e-9
    assert np.abs(sim.apply(src) - dst).max() < 1e-9


def test_register_identity_without_reference():
    sim = register_to_enu(_make_geoms(3), None)
    assert sim.registered is False
    assert abs(sim.scale - 1.0) < 1e-12
    assert np.allclose(sim.R, np.eye(3))


def test_register_to_enu_from_reference():
    geoms = _make_geoms(3)
    offset = np.array([10.0, 20.0, 0.0])
    ref = np.array([fg.cam_center for fg in geoms]) + offset
    sim = register_to_enu(geoms, ref, min_points=2)
    assert sim.registered
    assert abs(sim.scale - 1.0) < 1e-6
    assert np.allclose(sim.apply(geoms[0].cam_center.reshape(1, 3))[0], ref[0], atol=1e-6)


# --- fusion ---------------------------------------------------------------

def test_fusion_lands_on_known_plane():
    cloud = fuse_geometries(_make_geoms(1, z=3.0), stride=2, voxel=0.0)
    assert len(cloud) > 0
    assert np.allclose(cloud.points[:, 2], 3.0, atol=1e-5)


# --- pipeline integration -------------------------------------------------

def test_pipeline_adds_object_nodes_under_region():
    graph = _graph_with_region()
    geoms = _make_geoms(3)
    res = refine_scene_graph(graph, _stream(3), backend=FakeBackend(geoms), assoc_dist=0.4)

    assert len(res.object_nodes) >= 2
    for node in res.object_nodes:
        assert node.type == "object"
        assert node.pid == "region_a"
        assert node.refs[0].assigned_by == "rgb_stream"
        meta = node.refs[0].metadata
        assert {"label", "score", "bbox_min", "bbox_max", "size"} <= set(meta)
        assert graph.get_node(node.id) is not None
    # one contains-edge per new object
    obj_edges = [e for e in graph.edges if e.source_id == "region_a" and e.target_id.startswith("obj_")]
    assert len(obj_edges) == len(res.object_nodes)
    # embeddings produced for each object
    assert set(res.embeddings) == {n.id for n in res.object_nodes}


def test_refined_graph_round_trips_and_excludes_point_cloud():
    graph = _graph_with_region()
    res = refine_scene_graph(graph, _stream(3), backend=FakeBackend(_make_geoms(3)), assoc_dist=0.4)
    assert res.cloud is not None and len(res.cloud) > 0

    scene = SceneRepr(frame=Frame.from_origin(48.71, 2.20), graph=graph, nav=NavGraph())
    text = json.dumps(scene.to_dict())
    # schema guard: the render-only cloud never leaks into the graph JSON
    assert "cloud" not in text
    assert "point_cloud" not in text

    loaded = SceneRepr.from_dict(scene.to_dict())
    obj_ids = {n.id for n in res.object_nodes}
    loaded_objs = {n.id for n in loaded.graph.nodes if n.type == "object"}
    assert obj_ids == loaded_objs


def test_save_refine_writes_sidecars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = _graph_with_region()
    geoms = _make_geoms(3)
    ref = np.array([fg.cam_center for fg in geoms]) + np.array([1.0, 2.0, 0.0])
    paths = RunPaths(run_id="testrun")

    res = refine_scene_graph(
        graph, _stream(3, reference=ref), backend=FakeBackend(geoms),
        assoc_dist=0.4, paths=paths, save=True,
    )

    assert paths.refine_scene_graph.is_file()
    assert paths.point_cloud.is_file()  # cloud.ply sidecar
    assert paths.registration.is_file()
    assert paths.refine_metadata.is_file()

    meta = load_refine_metadata(paths)
    assert meta.object_count == len(res.object_nodes)
    assert meta.frame_count == 3
    assert meta.point_count == len(res.cloud)
    # point cloud referenced by URI in metadata only, never inside the graph JSON
    assert meta.point_cloud_uri.endswith("cloud.ply")
    graph_text = paths.refine_scene_graph.read_text()
    assert "cloud.ply" not in graph_text
    assert res.sim3.registered is True
