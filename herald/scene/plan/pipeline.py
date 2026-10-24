"""Planner, end to end: query -> descent shortlist -> LLM-ordered area sequence -> POI route.

Ties together descent.descend() (stage 1: hierarchical top-K scoring), order.order_route()
(stage 2: one LLM call that orders the shortlist), and RouteGraph (stage 3: shortest-path
between each ordered stop's POI). Each stage stays independently callable/testable -- this is
just the composition, so evaluation can isolate any one stage (see
experiments/planner-eval-01/idea.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from herald.scene.common.graph import SceneGraph, SceneNode
from herald.scene.common.route import RouteGraph
from herald.scene.plan.descent import DEFAULT_BETA, DEFAULT_SIGMA, descend
from herald.scene.plan.order import DONE_ID, REQUESTER_ID, order_route, pick_next_stop, real_contains


@dataclass
class PlanResult:
    shortlist: list[SceneNode]           # descent stage output
    sequence: list                        # order_route stage output (uids / REQUESTER_ID)
    stops: list[str]                      # resolved route-node ids, one per sequence entry
    path_nodes: list[str]                 # full nav-node path visiting every stop in order
    path_length_m: float


def _area_text(n: SceneNode, graph_by_uid: dict[int, SceneNode] | None = None) -> str:
    contains = n.attrs.get("contains") or ""
    real = real_contains(n, graph_by_uid) if graph_by_uid is not None else ""
    combined = ", ".join(x for x in (real, contains) if x)
    base = f"{n.name}. {n.desc}" if n.name else f"area {n.uid}"
    return f"{base} Contains: {combined}." if combined else base


def nearest_area_desc(pos, areas: list[SceneNode]) -> str | None:
    """A natural-language landmark phrase for `pos` (e.g. "near the Nurses' Station"), for
    order_route()'s requester_desc/robot_desc -- an LLM can reason over a named landmark, not
    over a raw [x, y, z] triple. None if no named/geometry area exists to anchor to (order_route
    then falls back to the raw-position phrasing)."""
    named = [n for n in areas if n.geom is not None and n.name]
    if not named:
        return None
    p = np.asarray(pos, np.float64)
    d = [float(np.linalg.norm(np.asarray(n.geom.offset, np.float64) - p)) for n in named]
    return f"near the {named[int(np.argmin(d))].name}"


def _resolve_stops_and_path(sequence: list, by_uid: dict[int, SceneNode], route: RouteGraph,
                             requester_pos, robot_pos) -> tuple[list[str], list[str], float]:
    """Shared by plan_route() and plan_route_iterative(): turn a sequence of area uids /
    REQUESTER_ID into POI route-nodes, then the actual nav-graph path + length visiting them in
    order. `by_uid` must cover every uid the sequence can contain -- plan_route() passes just its
    one shortlist (fixed for the whole call); plan_route_iterative() must pass the full graph's
    areas, since each step can draw from a DIFFERENT shortlist."""
    stops: list[str] = []
    for entry in sequence:
        if entry == REQUESTER_ID:
            pos = requester_pos
        else:
            node = by_uid.get(entry)
            if node is None or node.geom is None:
                continue
            pos = node.geom.offset
        poi_id = f"poi_{entry}"
        if not any(n.id == poi_id for n in route.nodes):
            route.add_poi(poi_id, entry if isinstance(entry, int) else -1, pos)
        stops.append(poi_id)

    start = route.nearest_node(robot_pos, type="nav")
    full_path: list[str] = [start.id] if start is not None else []
    total = 0.0
    prev = start.id if start is not None else None
    for stop_id in stops:
        if prev is None:
            break
        seg = route.shortest_path_nodes(prev, stop_id)
        if not seg:
            continue
        total += route.shortest_path(prev, stop_id)
        full_path.extend(seg[1:])
        prev = stop_id
    return stops, full_path, total


def plan_route(query: str, graph: SceneGraph, route: RouteGraph, encoder, *, robot_pos,
                requester_pos, model: str, beta: float = DEFAULT_BETA, sigma: float = DEFAULT_SIGMA,
                k: int = 5, node_emb: dict[int, np.ndarray] | None = None,
                distance_fn: Callable[[object], float] | None = None,
                use_landmark_desc: bool = False) -> PlanResult:
    """encoder: a services.embeddings.Encoder (area text -> embedding, MiniLM by default
    elsewhere in this codebase). node_emb, if given, skips re-encoding every area on every
    call -- pass a precomputed {uid: embedding} dict when planning many queries against the
    same graph (see the eval harness). distance_fn: passed straight through to
    herald.scene.plan.descent.descend() -- see its docstring (None = today's exact Euclidean
    proximity behavior). use_landmark_desc: whether to phrase robot/requester position as "near
    the X" (nearest_area_desc()) instead of raw [x,y,z] in the order_route() prompt -- despite
    seeming like an obvious improvement, this measurably HURT oracle-ordering coverage in
    controlled testing (0.794/0.600 -> ~0.53/0.25, see planner-eval-01/result.md's addendum), so
    it defaults to OFF (raw positions) until that regression is root-caused."""
    areas = [n for n in graph.nodes if n.level == "area" and n.geom is not None]
    graph_by_uid = {n.uid: n for n in graph.nodes}
    if node_emb is None:
        embs = encoder.encode([_area_text(n, graph_by_uid) for n in areas], is_query=False)
        node_emb = {n.uid: e for n, e in zip(areas, embs)}

    q_emb = encoder.encode([query], is_query=True)[0]
    shortlist = descend(graph, q_emb, node_emb, robot_pos, beta=beta, sigma=sigma, k=k,
                         distance_fn=distance_fn)
    requester_desc = nearest_area_desc(requester_pos, areas) if use_landmark_desc else None
    robot_desc = nearest_area_desc(robot_pos, areas) if use_landmark_desc else None
    sequence = order_route(query, shortlist, requester_pos, robot_pos, model,
                            requester_desc=requester_desc, robot_desc=robot_desc,
                            graph_by_uid=graph_by_uid)

    by_uid = {n.uid: n for n in shortlist}
    stops, full_path, total = _resolve_stops_and_path(sequence, by_uid, route, requester_pos, robot_pos)
    return PlanResult(shortlist=shortlist, sequence=sequence, stops=stops,
                       path_nodes=full_path, path_length_m=total)


def _progress_desc(sequence: list, graph_by_uid: dict[int, SceneNode]) -> str:
    if not sequence:
        return "Nothing visited yet."
    parts = []
    for entry in sequence:
        if entry == REQUESTER_ID:
            parts.append("returned to the person")
        else:
            node = graph_by_uid.get(entry)
            parts.append(node.name if node is not None and node.name else f"area {entry}")
    return "Already visited, in order: " + "; then ".join(parts) + "."


def plan_route_iterative(query: str, graph: SceneGraph, route: RouteGraph, encoder, *, robot_pos,
                          requester_pos, model: str, beta: float = DEFAULT_BETA,
                          sigma: float = DEFAULT_SIGMA, k: int = 5,
                          node_emb: dict[int, np.ndarray] | None = None,
                          distance_fn_factory: Callable[[object], Callable[[object], float]] | None = None,
                          max_steps: int = 8, use_landmark_desc: bool = False) -> PlanResult:
    """User-directed alternative to plan_route()'s one-shot full-sequence call: resolve ONE
    destination at a time (herald.scene.plan.order.pick_next_stop()), re-augmenting the prompt
    with a running "already visited" summary and re-running descend() each step, until the LLM
    itself says the request is satisfied (DONE_ID) or `max_steps` is hit (safety cap against a
    model that never says DONE). Motivation: on the real Hospital benchmark, single-destination
    queries resolve far more reliably than multi-destination ones in one shot (strict "all
    covered" rate 0.550 at N=1 vs. 0.083 at N=2, planner-eval-01/result.md's addendum) --
    decomposing a multi-destination request into repeated single-destination decisions tests
    whether that per-step reliability compounds better than asking for the whole sequence at
    once. Unlike plan_route(), `robot_pos` genuinely updates to the last resolved stop between
    steps, so the geodesic/proximity term reflects where the robot actually is by that point,
    not a single static position for the whole multi-step plan. distance_fn_factory (NOT a
    static distance_fn like plan_route() takes): a callable `pos -> distance_fn` returning a
    fresh distance_fn for descend(), called EVERY step with the CURRENT robot position -- a
    geodesic distance_fn built once from the initial position would go stale the moment the
    robot "moves" to its first resolved stop, since the Dijkstra source needs to move with it.
    None keeps Euclidean-to-cur_pos (descend()'s default) at every step."""
    areas = [n for n in graph.nodes if n.level == "area" and n.geom is not None]
    graph_by_uid = {n.uid: n for n in graph.nodes}
    if node_emb is None:
        embs = encoder.encode([_area_text(n, graph_by_uid) for n in areas], is_query=False)
        node_emb = {n.uid: e for n, e in zip(areas, embs)}
    q_emb = encoder.encode([query], is_query=True)[0]
    requester_desc = nearest_area_desc(requester_pos, areas) if use_landmark_desc else None

    cur_pos = robot_pos
    sequence: list = []
    last_shortlist: list[SceneNode] = []
    for step in range(max_steps):
        distance_fn = distance_fn_factory(cur_pos) if distance_fn_factory else None
        shortlist = descend(graph, q_emb, node_emb, cur_pos, beta=beta, sigma=sigma, k=k,
                             distance_fn=distance_fn)
        last_shortlist = shortlist
        robot_desc = nearest_area_desc(cur_pos, areas) if use_landmark_desc else None
        next_id = pick_next_stop(query, _progress_desc(sequence, graph_by_uid), shortlist,
                                  requester_pos, cur_pos, model,
                                  requester_desc=requester_desc, robot_desc=robot_desc,
                                  graph_by_uid=graph_by_uid, allow_done=step > 0)
        if next_id == DONE_ID:
            break
        entry = REQUESTER_ID if next_id == REQUESTER_ID else int(next_id)
        sequence.append(entry)
        cur_pos = requester_pos if entry == REQUESTER_ID else graph_by_uid[entry].geom.offset

    stops, full_path, total = _resolve_stops_and_path(sequence, graph_by_uid, route, requester_pos, robot_pos)
    return PlanResult(shortlist=last_shortlist, sequence=sequence, stops=stops,
                       path_nodes=full_path, path_length_m=total)
