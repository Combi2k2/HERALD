"""Flat plan-view Rerun rendering."""

from __future__ import annotations

import numpy as np

from herald.scene.common.geometry import Frame, Geometry
from herald.scene.common.graph import SceneNode
from herald.scene.common.repr import SceneRepr

_GROUND = 0.0
_LINE_SCENE = 1.5
_LINE_NAV = 1.25
_R_NAV = 2.0

CATEGORY_COLORS: dict[str, list[int]] = {
    "building": [96, 165, 250],
    "vegetation": [74, 222, 128],
    "water": [34, 211, 238],
    "recreation": [192, 132, 252],
    "parking": [250, 204, 21],
    "plaza": [251, 191, 36],
    "service_point": [244, 114, 182],
    "ground_other": [161, 161, 170],
    "unknown": [113, 113, 122],
}
ROLE_COLORS: dict[str, list[int]] = {
    "structure": CATEGORY_COLORS["building"],
    "region_use": CATEGORY_COLORS["vegetation"],
    "facility": CATEGORY_COLORS["service_point"],
    "site": CATEGORY_COLORS["ground_other"],
    "unclassified": CATEGORY_COLORS["unknown"],
}
DISTANCE_COLORS: list[list[int]] = [
    [113, 113, 122],
    [59, 130, 246],
    [34, 197, 94],
    [234, 179, 8],
    [168, 85, 247],
    [239, 68, 68],
]
_NAV_NODE = [250, 204, 21, 255]
_NAV_EDGE = [234, 179, 8, 255]


def _rr():
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError("rerun-sdk required: uv sync --group viewer") from exc
    return rr


def _rgba(c: list[int]) -> list[int]:
    return [*c[:3], 255] if len(c) == 3 else list(c)


def _pt(east: float, north: float, up: float = _GROUND) -> list[float]:
    """ENU → Rerun plan view (north-up; fixes 180° orientation)."""
    return [-float(east), -float(north), up]


def _ring(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0 or np.allclose(xy[0], xy[-1]):
        return xy
    return np.vstack([xy, xy[0:1]])


def _xy(frame: Frame | None, geom: Geometry) -> np.ndarray:
    if geom.frame == "ENU":
        return geom.coords[:, :2]
    if frame is None:
        return geom.coords[:, :2]
    return np.array([frame.wgs2enu(float(r[0]), float(r[1])) for r in geom.coords])


def _children(graph) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in graph.edges:
        if e.edge_type == "contains":
            out.setdefault(e.source, []).append(e.target)
    return out


def _dist_to_leaf(node_id: str, tree: dict[str, list[str]], cache: dict[str, int]) -> int:
    if node_id in cache:
        return cache[node_id]
    kids = tree.get(node_id, [])
    d = 0 if not kids else 1 + max(_dist_to_leaf(c, tree, cache) for c in kids)
    cache[node_id] = d
    return d


def semantic_color(node: SceneNode) -> list[int]:
    if node.attrs.get("category") in CATEGORY_COLORS:
        return list(CATEGORY_COLORS[node.attrs["category"]])
    if node.attrs.get("role") in ROLE_COLORS:
        return list(ROLE_COLORS[node.attrs["role"]])
    return list(CATEGORY_COLORS["building" if node.level == "structure" else "unknown"])


def hierarchy_color(dist: int) -> list[int]:
    return list(DISTANCE_COLORS[min(dist, len(DISTANCE_COLORS) - 1)])


def render_scene(scene, *, spawn: bool = True, static: bool = True) -> None:
    """Log semantic / hierarchy / nav under ``/``; toggle layers in the entity tree."""
    rr = _rr()
    rr.init("herald-scene", spawn=spawn)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    tree = _children(scene.graph)
    dist_cache: dict[str, int] = {}
    span = 500.0
    lw = rr.Radius.ui_points(_LINE_SCENE)

    for node in scene.graph.nodes:
        if node.level == "site" or node.geom.type != "polygon":
            continue
        strip = [_pt(x, y) for x, y in _ring(_xy(scene.frame, node.geom))]
        if len(strip) < 3:
            continue
        for x, y, _ in strip:
            span = max(span, abs(x), abs(y), 1.0)
        dist = _dist_to_leaf(node.uid, tree, dist_cache)
        attrs = dict(
            uid=node.uid,
            level=node.level,
            role=node.attrs.get("role", ""),
            category=node.attrs.get("category", ""),
            max_distance_to_leaf=dist,
        )
        for layer, rgb in (
            ("semantic", semantic_color(node)),
            ("hierarchy", hierarchy_color(dist)),
        ):
            base = f"{layer}/{node.uid}"
            rr.log(base, rr.LineStrips3D([strip], colors=[_rgba(rgb)], radii=[lw]), static=static)
            rr.log(f"{base}/attrs", rr.AnyValues(**attrs), static=static)

    nav = scene.nav
    if nav.nodes:
        pos = [_pt(*n.pos) for n in nav.nodes]
        for x, y, _ in pos:
            span = max(span, abs(x), abs(y), 1.0)
        rr.log("nav/nodes", rr.Points3D(pos, colors=[_NAV_NODE], radii=_R_NAV), static=static)
        by_id = {n.id: n.pos for n in nav.nodes}
        strips = [
            [_pt(*by_id[e.source_id]), _pt(by_id[e.target_id][0], by_id[e.target_id][1], _GROUND + 0.05)]
            for e in nav.edges
            if e.source_id in by_id and e.target_id in by_id
        ]
        if strips:
            rr.log(
                "nav/edges",
                rr.LineStrips3D(strips, colors=[_NAV_EDGE], radii=[rr.Radius.ui_points(_LINE_NAV)]),
                static=static,
            )

    try:
        import rerun.blueprint as rrb
    except (ImportError, ModuleNotFoundError, AttributeError):
        return

    eye = max(800.0, span * 1.4)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Spatial3DView(
                origin="/",
                name="HERALD",
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=(0.0, 0.0, eye),
                    look_target=(0.0, 0.0, 0.0),
                ),
            ),
            collapse_panels=True,
        )
    )
