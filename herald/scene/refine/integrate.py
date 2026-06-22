"""Integrate fused object instances into the scene graph.

Each :class:`ObjectInstance` (in the reconstruction frame) is mapped to the site
ENU frame via the :class:`Sim3`, attached to the nearest containing ``region``
(falling back to nearest region, then the ``site`` root), and added as an
``object`` :class:`SceneNode`. Provenance is recorded with a ``SourceRef`` of
``assigned_by="rgb_stream"`` carrying the 3D bbox + score, so the schema needs no
new fields. Captions are embedded with the shared :class:`Encoder` (stub by
default); embedding *vectors* are returned for an optional sidecar, never inlined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon

from herald.scene.common.geometry import Geometry
from herald.scene.common.graph import SceneGraph, SceneNode, SourceRef
from herald.scene.refine.associate import ObjectInstance
from herald.scene.refine.register import Sim3
from services.embeddings import Encoder, StubEncoder

# Node types an object may be parented to (coarsest-to-finest containers).
CONTAINER_TYPES = ("space", "floor", "structure", "region", "zone")


@dataclass
class IntegrateResult:
    graph: SceneGraph
    object_nodes: list[SceneNode] = field(default_factory=list)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)


def _node_polygon(node: SceneNode) -> Polygon | None:
    """2D ENU footprint of a container node, or None if it has no usable ring."""
    if node.geom.type != "polygon" or node.geom.frame != "ENU":
        return None
    ring = [(float(p[0]), float(p[1])) for p in node.geom.coords]
    if len(ring) < 3:
        return None
    poly = Polygon(ring)
    return poly if poly.is_valid and poly.area > 0 else None


def _aabb_enu(inst: ObjectInstance, sim3: Sim3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid + axis-aligned bbox (min, max) in ENU after applying ``sim3``."""
    lo, hi = inst.aabb
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    corners_enu = sim3.apply(corners)
    centroid = sim3.apply(inst.centroid.reshape(1, 3))[0]
    return centroid, corners_enu.min(axis=0), corners_enu.max(axis=0)


def _find_parent(point_xy: np.ndarray, containers: list[tuple[SceneNode, Polygon]], graph: SceneGraph) -> SceneNode | None:
    """Smallest container that contains the point; else nearest; else site root."""
    pt = Point(float(point_xy[0]), float(point_xy[1]))
    containing = [(poly.area, node) for node, poly in containers if poly.contains(pt)]
    if containing:
        return min(containing, key=lambda t: t[0])[1]
    if containers:
        return min(containers, key=lambda nb: nb[1].distance(pt))[0]
    return graph.site_node()


def _unique_id(graph: SceneGraph, base: str) -> str:
    if graph.get_node(base) is None:
        return base
    i = 1
    while graph.get_node(f"{base}_{i}") is not None:
        i += 1
    return f"{base}_{i}"


def integrate_objects(
    graph: SceneGraph,
    instances: list[ObjectInstance],
    sim3: Sim3,
    *,
    encoder: Encoder | None = None,
    id_prefix: str = "obj",
    source: str = "rgb_stream",
) -> IntegrateResult:
    """Add ``object`` nodes for ``instances`` to ``graph`` (mutates in place)."""
    enc = encoder or StubEncoder()
    containers = [
        (node, poly)
        for node in graph.nodes
        if node.type in CONTAINER_TYPES and (poly := _node_polygon(node)) is not None
    ]

    new_nodes: list[SceneNode] = []
    for k, inst in enumerate(instances):
        centroid, lo, hi = _aabb_enu(inst, sim3)
        parent = _find_parent(centroid, containers, graph)
        nid = _unique_id(graph, f"{id_prefix}_{k:04d}")
        metadata = {
            "label": inst.label,
            "score": float(inst.score),
            "bbox_min": [float(v) for v in lo],
            "bbox_max": [float(v) for v in hi],
            "size": [float(v) for v in (hi - lo)],
            "observation_count": int(inst.observation_count),
            "registered": bool(sim3.registered),
        }
        node = SceneNode(
            id=nid,
            pid=parent.id if parent else None,
            type="object",
            geom=Geometry(type="point", frame="ENU", coords=[[float(centroid[0]), float(centroid[1]), float(centroid[2])]]),
            refs=[SourceRef(assigned_by=source, assigned_id=f"track_{inst.id}", metadata=metadata)],
            observation_count=int(inst.observation_count),
            name=inst.label,
            category=inst.label,
        )
        node.txt_embedding_ref = nid
        graph.add_node(node)
        if parent is not None:
            graph.add_edge(parent.id, nid, edge_type="contains")
        new_nodes.append(node)

    embeddings: dict[str, np.ndarray] = {}
    if new_nodes:
        vecs = enc.encode([n.name for n in new_nodes])
        embeddings = {n.id: vecs[i] for i, n in enumerate(new_nodes)}

    return IntegrateResult(graph=graph, object_nodes=new_nodes, embeddings=embeddings)
