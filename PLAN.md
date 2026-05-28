# Implementation Plan: Site-Scale Semantic Navigation Framework

## Overview

Two-component framework for campus-scale robot navigation:
- **Global planner**: hierarchical 3D scene graph + language-grounded path planning
- **Local motion**: waypoint-conditioned diffusion policy with SAM2 + VGGT features

Base codebase: HOV-SG. Novel additions: OSM initialization, ego-vehicle trajectory nodes,
Voronoi-augmented nav graph, LLM query decomposition, experience-based edge weights.

---

## Prioritized Implementation Steps

### Priority 1 — OSM Initialization (`osmnx`)

**Goal**: Replace empty graph initialization with a structurally meaningful scaffold
derived from OpenStreetMap before any robot exploration begins.

**Tool**: `osmnx` (5.7k★, actively maintained)

```python
import osmnx as ox
import networkx as nx

# Zone nodes: building footprints → V_Z
buildings = ox.features_from_place("IP Paris, Palaiseau", tags={"building": True})
# → extract centroid, polygon, metadata text → precompute CLIP embedding

# Nav graph backbone: walkable pathways → G_nav
G_walk = ox.graph_from_place("IP Paris, Palaiseau", network_type="walk")
# Already a networkx graph → Dijkstra is one call:
path = nx.shortest_path(G_walk, source, target, weight="length")
```

**Deliverable**: Site node + Zone nodes populated with geometry, text descriptions,
and CLIP embeddings. Pathway network stored as networkx graph for later nav graph construction.

---

### Priority 2 — Online Object Node Refinement (ConceptGraphs real-time branch)

**Goal**: Incremental object node construction from live RGB-D stream using
SAM2 segmentation + CLIP embeddings + 3D projection.

**Tool**: `concept-graphs/concept-graphs` (877★), `ali-dev` branch for real-time mode.

**Pipeline per frame**:
1. SAM2 automatic mode → instance masks (no predefined categories)
2. CLIP ViT-L/14 crop encoding → embedding `e_i ∈ R^d`
3. Project mask onto LiDAR point cloud → centroid `p_i ∈ R^3`
4. Spatial gating: find candidates `N_r = {j | ‖p_i - p_j‖ < r}`
5. Semantic matching: `ssim_ij = cos(e_i, e_j)`
   - If `max ssim > τ`: EMA update on embedding + geometry
   - Else: insert new node

**Key parameters**: radius `r`, similarity threshold `τ`, EMA decay factor.

**Deliverable**: Object nodes (`V_O`) continuously updated during exploration,
with stable embeddings and 3D positions.

---

### Priority 3 — FAISS Retrieval Index

**Goal**: Efficient CLIP embedding search over `V_O` nodes at campus scale.
Brute-force cosine search will not scale beyond ~1000 nodes.

**Tool**: `faiss-gpu` (Meta, 33k★)

```python
import faiss
import numpy as np

# Build index over all node embeddings
d = 512  # CLIP embedding dim
index = faiss.IndexFlatIP(d)  # inner product = cosine on normalized vecs
embeddings = np.stack([v.embedding for v in graph.nodes])
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Query
query_vec = clip_encode(text_description)
faiss.normalize_L2(query_vec)
D, I = index.search(query_vec, k=5)  # top-5 candidates
```

**Deliverable**: Sub-millisecond ANN retrieval over scene graph nodes.
Update index incrementally as new nodes are inserted.

---

### Priority 4 — Voronoi Navigation Graph Augmentation

**Goal**: Extend OSM pathway backbone with Voronoi-sampled waypoints to cover
open spaces (plazas, courtyards) not represented in OSM.

**Tool**: `scipy.spatial.Voronoi` (simpler and more controllable than VoronoiPlanner3D)

```python
from scipy.spatial import Voronoi
import numpy as np

# Sample free-space points from 2D occupancy grid
free_points = get_free_space_samples(occupancy_grid, n=500)

# Compute Voronoi diagram
vor = Voronoi(free_points)

# Filter edges that intersect obstacles
valid_edges = [
    (vor.vertices[e[0]], vor.vertices[e[1]])
    for e in vor.ridge_vertices
    if e[0] >= 0 and e[1] >= 0
    and not intersects_obstacle(vor.vertices[e[0]], vor.vertices[e[1]], occupancy_grid)
]

# Merge with OSM pathway graph
G_nav = merge_graphs(G_osm_walk, valid_edges)
```

**Deliverable**: Dense `G_nav = (W, P)` covering both structured pathways and open spaces.
Experience-based edge weights: lower cost for previously traversed edges.

---

### Priority 5 — LLM Query Decomposition + Hierarchical Retrieval

**Goal**: Map natural language goal `q` → sequence of keyframes
`{(d_k^Z, d_k^A, d_k^O)}` → resolved graph nodes → waypoint sequence.

**Tool**: Any LLM API (structured JSON output mode). Reference: OVSG (`changhaonan/OVSG`, CoRL 2023)
for prompt design patterns.

**Prompt template structure**:
```
System: You are a navigation assistant for a robot on a university campus.
Given a task query and the robot's current context (zone, area), decompose
the query into a sequence of hierarchical location descriptors.

Output JSON:
{
  "keyframes": [
    {"zone": "...", "area": "...", "object": "..."},
    ...
  ]
}

Context: Robot is currently in {current_zone}, {current_area}.
Query: {q}
```

**Hierarchical retrieval** (top-down, FAISS-accelerated):
```python
v_Z = argmax_{v in V_Z} cos(e_v, CLIP(d_k^Z))
v_A = argmax_{v in V_A, parent=v_Z} cos(e_v, CLIP(d_k^A))
v_O = argmax_{v in V_O, parent=v_A} cos(e_v, CLIP(d_k^O))
```

Hybrid re-ranking when robot pose is available:
```
η(v) = β · spatial(v, p_ego) + (1 - β) · cos(e_v, CLIP(d_k))
```

**Deliverable**: End-to-end query → waypoint sequence pipeline.

---

## Tool Stack Summary

| Component | Tool | Repo | Stars |
|---|---|---|---|
| OSM initialization | osmnx | `gboeing/osmnx` | 5.7k |
| Scene graph base | HOV-SG | `hovsg/HOV-SG` | 477 |
| Online object refinement | ConceptGraphs (ali-dev) | `concept-graphs/concept-graphs` | 877 |
| Streaming 3D segmentation | EmbodiedSAM | `xuxw98/EmbodiedSAM` | 632 |
| Embedding retrieval | FAISS | `facebookresearch/faiss` | 33k |
| Voronoi nav graph | scipy.spatial.Voronoi | stdlib | — |
| Path planning | networkx Dijkstra | stdlib | — |
| Query retrieval reference | OVSG | `changhaonan/OVSG` | 72 |
| SAM2 features (local motion) | SAM2 | `facebookresearch/sam2` | — |
| Geometry features (local motion) | VGGT | — | — |

---

## Literature Gaps to Address

- **DovSG** (RA-L 2025): "Dynamic Open-Vocabulary 3D Scene Graphs for Long-Term
  Language-Guided Mobile Manipulation" — addresses long-term map maintenance (stale
  node detection), which is identified as a limitation. Add to related work section.

- **CausalNav** (Duan 2026): No public repo found as of May 2026. Note in report
  that this dependency is not publicly reproducible.

---

## Key Design Decisions

### SAM vs SAM2
- Current HOV-SG experiments use SAM (per-frame, no temporal consistency)
- Transition to SAM2 for online refinement enforces temporal instance tracking
  across the onboarding video stream — stabilizes EMA node updates

### Online vs Offline Processing
- Onboarding phase: chunk-based incremental processing (not full-batch)
  → scene graph updated iteratively during live exploration
- Production: async pipeline — SAM2 encoder on separate thread from graph fusion

### Safety
- Diffusion policy alone is insufficient for safety guarantees
- Future: DWA or CBF safety shield running at high frequency, overriding
  diffusion output on imminent collision detection

---

## Experiment Sequence

1. Validate OSM initialization on IP Paris campus (osmnx → zone nodes + nav graph)
2. Validate online refinement on Habitat-Sim (extending current HOV-SG experiments)
3. Validate FAISS retrieval matches brute-force baseline
4. Validate Voronoi nav graph covers open-space gaps in OSM data
5. End-to-end: natural language query → waypoint sequence on Habitat-Sim
6. Real-world onboarding phase on IP Paris campus (teleoperation)
