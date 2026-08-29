# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project rule

- Do NOT create a `tests/` folder or write test files unless the user explicitly asks for tests. (`scripts/test_*.py` are runnable diagnostics/demos, not pytest suites.)

## Environment & commands

Dependencies are managed with **uv** (never `pip`/`venv` directly); `.venv/` is created on first `uv sync`.

```bash
uv sync                          # default groups only: dev, viz, vlm
uv sync --group recon            # + VGGT-Omega geometry + torch/opencv (heavy, CUDA)
uv sync --group seg              # + SAM2/SigLIP via transformers>=4.56 (heavy, CUDA)
uv sync --group viewer           # + rerun-sdk
uv run python scripts/scene_init.py --serve
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest   # disable stray system plugins (e.g. ROS)
```

**On a fresh clone**, the Boxer engine (`third_party/boxer`) is a pinned git submodule and its
checkpoints (~1.1 GB) are *not* in git — fetch both once:

```bash
git submodule update --init third_party/boxer          # or: git clone --recurse-submodules ...
bash third_party/boxer/scripts/download_ckpts.sh        # BoxerNet + DINOv3 + OWLv2 from HuggingFace
```

`recon` and `seg` are deliberately **out of default-groups** (large, CUDA-bound) — install explicitly. `vggt-omega` ships from GitHub, not PyPI, imported as `vggt_omega`.

### Cluster / GPU (SLURM)

Heavy jobs run via the SLURM templates in `scripts/*.slurm` (submit with `sbatch`, override config via `sbatch --export=ALL,TRAJ=P0001,...`). Hard-won constraints:

- **Use `--partition=3090`, not the default P100.** The cu130 torch build crashes on the P100 (compute capability 6.0); cuML/RAPIDS also needs cc ≥ 7.0. The 3090 partition caps at 4 CPUs per GPU.
- **Clear `PYTHONPATH`** (`unset PYTHONPATH` in SLURM, or prefix `PYTHONPATH= uv run ...`). A leaked system site-packages path lands ahead of the uv venv and shadows it (e.g. system `huggingface_hub` breaks `transformers`).
- Login nodes have no usable GPU; segmentation/embedding will silently fall to CPU there.

## Architecture

HERALD is a **site-scale semantic navigation framework** (see `PLAN.md` for the full roadmap). Two phases, both building/refining one hierarchical scene graph (`site → region/zone → area → object → ego`):

### Phase 1 — Offline scene-graph init (`herald/scene/init/`)

`build_scene_graph()` (`init/pipeline.py`) turns OpenStreetMap polygons into a `SceneRepr` (graph + nav graph): Overpass fetch (`services/osm.py`) → `seed_graph` (site + region nodes) → `build_tree` containment hierarchy → `build_feat` classification (optionally a vision-LLM per polygon via `services/vlm.py`) → `build_path` pathway nav graph. Entry: `scripts/scene_init.py` (browser GPS/ROI picker, Rerun/Folium validation), viewed with `scripts/scene_view.py`. Runs are written to `data/{run_id}/{raw,phase1}/`.

### Phase 2 — Online RGB(-D) reconstruction (two-tier)

Turns a session's RGB(-D) frames into **static object OBBs + a colored scene cloud**, then (eventually) reconciles many sessions into one persistent map. Two tiers, joined by a `SessionResult` firewall (Tier 2 consumes `SessionResult`, never raw frames):

**Tier 1 — `herald/scene/recon/` (per session, one video).** Six files: `pipeline.py` (`SessionRecon`), `geometry.py` (`GtGeometry`; `VggtGeometry` wired in from `services.vggt`), `corridor.py` (`span`, `classify_static`), `types.py` (`SceneMap` + its `SessionResult` alias, `SceneObject`), `utils.py` (geom/cluster/IO), `__init__.py`. **Unified scene structure:** a per-session recon and a merged multi-session map are the same `SceneMap` (cloud + `SceneObject`s); a `SceneObject` carries a persistent `uid` (stable across merges; the per-video `track_id` lives in per-session `sessions` provenance), OBB, `label`/`labels`, `conf`, `support`, `embedding` (reserved), and `crops`. `save_session`/`load_session` round-trip every field (backward-compatible with pre-unification dumps). `SessionRecon.add_frame` streams frames through the Boxer engine, accumulates the scene cloud (`utils.SceneCloud`), and `finalize()` returns a `SessionResult`. Finalize order: **`suppress_contained`** (drop OBBs ≥`contain_thr` nested inside a larger one — part-of-a-whole boxes) → **`classify_static`** (corridor split). Object existence is gated per-frame (`conf_thr2d`/`conf_thr3d`) and by track support (`min_obs`) — there is no lifetime-confidence filter; the fused `(score2d+score3d)/2` is carried as an informational `conf` attribute only. `__init__` loads `SessionRecon` lazily so importing the package stays torch-free (though importing `herald.scene` at all pulls torch via the Phase-1 chain). `utils.py` = `unproject_labeled`/`unproject_rgb`, `SceneCloud`, `filter_clusters` (DBSCAN cuML→sklearn), `quat_to_R`, `w2c_to_c2w`, `write_ply`/`save_cloud`/`load_cloud`.

**Boxer engine — `services/boxer.py` (`BoxerPipeline`).** Boxer's *complete* model, OWLv2-only: OWLv2 open-vocab detect → BoxerNet 3D lift (labels stamped via `set_text` so the tracker's semantic merge can fire; confidence = `(score2d+score3d)/2`) → Boxer's online `BoundingBox3DTracker` (Hungarian 3D-IoU match on *visible* tracks + occlusion-aware aging + semantic-gated duplicate merge). It stays a **generic** engine — no dynamic logic; `add_frame` returns the per-frame track association `{match_tids, match_boxes, match_centers}`, `objects()` returns fused OBBs `{center, size, quat, conf, support, label, labels, crops}` in NED. Boxer is a **pinned git submodule** at `third_party/boxer` (facebookresearch/boxer, CC-BY-NC 4.0; not pip-installable — `sys.path` is prepended and colliding `utils`/`loaders` modules evicted, see the file header) and needs downloaded checkpoints (`third_party/boxer/scripts/download_ckpts.sh`; ~1.1 GB, gitignored).

**Corridor filter (dynamic removal) lives entirely in recon.** A Boxer track *is* a spatial-temporal corridor (CausalNav-style); `SessionRecon` builds per-track trajectories from `add_frame`'s match arrays. `corridor.span` = AABB diagonal of the world centroids. One threshold `corridor_span` drives two effects: (1) **online** — once a track's span crosses it, that track's 2D box is cut from the scene-cloud unprojection so a mover never smears the static cloud (span only grows → no flip-back); (2) **finalize** — `classify_static` drops movers (gated by `dyn_min_support` so a low-support flaky track isn't trusted as dynamic). Catches *in-video* movers only.

**Tier 2 — `herald/scene/refine/` (multi-session): update/insert built, retire not.** `align.py` registers a session cloud onto the canonical map by energy-based Sim3 (RANSAC ground-level → scale/yaw/translation). `reconcile.py` then merges objects: each cross-session pair fuses via a **two-pass gate** (pass 1 `IoS ≥ ios_trust`, label-agnostic; pass 2 `cdiag < cdiag_gate` AND `labels_agree`) and a **union-find** collapses each connected component into one object (`_fuse_many`: centre confidence-averaged, orientation+extent from the lead box — never average extents across frames, it inflates thin objects; support summed, provenance unioned). `labels_agree` is an isolated swap point (currently label-vote overlap; intended to become embedding-similarity). Fused/only-A keep the canonical A `uid`; only-B get fresh uids. **Persistent map:** a single accumulating `SceneMap` on disk (`persistent.npz`) that each `scene_track_obj` run updates in place (reconcile the new session's objects in, then `overlap_merge` label-agnostic dedup). **Route graph (`route.py`):** `RouteGraph` is a persistent node+edge lattice — `build_route` (`refine/route.py`) turns one trajectory into nodes (Taubin smooth → 1 m arc-length resample → sequential edges), `absorb` appends another graph (ids stay stable, append-only so object attachments never remap), and `add_proximity_edges(radius, min_seq_gap)` links intra-session loop closures + cross-session pairs over the whole graph. **Areas (`scene_area_cluster`):** groups objects into portal-bounded `area` nodes — object↔object distance = observing-nav-node→object offset + route geodesic + offset, minimised over the two anchor sets, with any offset hop or route edge crossing a **portal** gate (`refine/portals.edge_crosses_gate`) set to ∞ (no Euclidean floor); agglomerative within `d_max`. **Portals** = `entrance` objects plus mirrors the camera walked through or sitting near a door (marked `attrs["is_portal"]`). **Evidence-gated retire** (dropping a canonical object a new visit proves is gone, gated on frustum + free-space, not a timeout — the fix for Boxer's ghost-box problem) is **not yet built**. Entry: `scene_align.py` (Sim3; `--no-align` for pre-registered datasets) → `scene_track_obj.py` → `scene_merge_route.py` (route + attachment) → `scene_area_cluster.py` → `scene_area_classify.py` (VLM area captions).

**Phase-2 entry points (persistent pipeline).** Per session T: `scripts/scene_recon.py` (Tier-1 recon → `SessionResult` dump; GPU) → `scene_align.py` → `scene_track_obj.py` → `scene_merge_route.py`; then `scene_area_cluster.py` once, optional `scene_area_classify.py`, and `scene_render.py` to view. Scripts only write to disk — rendering is a separate step. The persistent representation lives under `data/tartanground/<Env>/scene/`: `persistent.npz` (accumulating `SceneMap`), `route.json` (route graph), `attach.json` (object→observing-node attachment), `scene_graph.json` (area/object `SceneGraph`), `sim3_<T>.json`. The whole flow runs via `scripts/scene_pipeline.slurm` (recon on GPU ×N → integration on CPU); recon alone via `scripts/scene_recon.slurm`. Data lives under `data/tartanground/` (fetch with `scripts/download_tartanground.py`); job outputs go to `runs/{jobid}/`.

### Supporting packages

- `herald/scene/common/` — shared domain types (`SceneGraph`/`SceneNode` (int-uid, OBB-capable `Geometry`), `SourceRef`, `RouteGraph`/`RouteNode`/`RouteEdge`, `Frame`/coordinate transforms WGS↔UTM↔ENU, `SceneRepr`).
- `herald/datasets/` — dataset adapters (`TartanGroundTraj`/`...Geometry`/`...Seg`) exposing stored streams as geometry/segmentation sources.
- `services/` — external integrations & model wrappers: `boxer.py` (`BoxerPipeline`, the Phase-2 recon engine over the Boxer submodule), `vggt.py` (`VggtStream` monocular geometry), `osm.py` (Overpass), `vlm.py` (LangChain/Ollama or OpenAI), `sam2.py` (SAM2 — kept for future semantic routing, no longer on the recon path), `aerial.py` (tiles), `embeddings/` (`Encoder`, `StubEncoder`).
- `herald/ui/`, `herald/viz/`, `herald/browser.py` — Rerun/Folium rendering, ROI picker, browser launch.
- `utils/` — OSM tag filtering/helpers used by Phase 1 classification.

The three installable top-level packages are `herald*`, `services*`, `utils*` (see `pyproject.toml`).
