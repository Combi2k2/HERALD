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

### Phase 2 — Online RGB(-D) reconstruction (`herald/scene/recon/`)

Streams RGB frames into a **class-agnostic labeled point cloud**. Three *independent* submodules, wired by `ReconStream` (`recon/pipeline.py`), passing plain numpy:

- `vggt.py` (`VggtStream`) — monocular geometry `{K, c2w, depth, conf}`; may buffer frames into chunks. Optional: dataset GT geometry can be supplied instead.
- `sam2.py` (`Sam2Segmenter`) — per-frame instance masks (automatic mode, no categories). Deliberately **stateless**; cross-frame identity was moved into the Fuser.
- `fusion.py` (`Fuser`) — associates masks into objects by **appearance** (SigLIP crop embedding, `embed.py`) **+ occupancy** (voxel overlap), then votes them into the cloud. Two stages: (1) per-frame `push()`→`_match()` (AABB prefilter → semantic cosine gate → spatial overlap gate); (2) flush-time `_merge()` union-find pass that bridges fragments of the same object.

`ReconStream` drains the geometry and mask queues in lockstep (VGGT chunking vs per-frame SAM2), so each fused frame pairs the right geometry with the right masks. `flush()` is a non-destructive snapshot (watch the cloud build up in Rerun); `finish()` drains and returns the final cloud.

`recon/__init__.py` lazily re-exports `Sam2Segmenter`/`SiglipEmbedder` and does **not** re-export `vggt`, so fusion-only / geometry-only users don't pay the torch + transformers import cost. `recon/utils.py` holds the shared geometry/IO helpers (`unproject_labeled`, `bbox_crop`, `filter_clusters` — DBSCAN via cuML on GPU else sklearn — `write_ply`, etc.).

**Recon entry points:** `scripts/test_recon.py` (full streaming pipeline on TartanGround GT geometry, renders evolving in Rerun), `scripts/fuse_gt.py` (GT geometry **and** GT segmentation — upper bound), `scripts/test_fuser.py` (synthetic, CPU-only), plus diagnostics (`dump_crops.py`, `csim_stats.py`, `text_probe.py`) for the appearance-gate investigation. Data lives under `data/tartanground/` (fetch with `scripts/download_tartanground.py`); job outputs go to `runs/`.

### Supporting packages

- `herald/scene/common/` — shared domain types (`SceneGraph`, `SceneNode`, `Frame`/coordinate transforms WGS↔UTM↔ENU, `NavGraph`, `Trajectory`, `SceneRepr`).
- `herald/datasets/` — dataset adapters (`TartanGroundTraj`/`...Geometry`/`...Seg`) exposing stored streams as geometry/segmentation sources.
- `services/` — external integrations: `osm.py` (Overpass), `vlm.py` (LangChain/Ollama or OpenAI), `aerial.py` (tiles), `embeddings/` (`Encoder`, `StubEncoder`).
- `herald/ui/`, `herald/viz/`, `herald/browser.py` — Rerun/Folium rendering, ROI picker, browser launch.
- `utils/` — OSM tag filtering/helpers used by Phase 1 classification.

The three installable top-level packages are `herald*`, `services*`, `utils*` (see `pyproject.toml`).
