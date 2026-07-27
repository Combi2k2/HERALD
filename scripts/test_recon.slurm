#!/bin/bash
#SBATCH --job-name=herald-recon
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4    # 3090 partition allows max 4 CPUs per GPU
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# 3090 partition = RTX 3090 (compute capability 8.6). Required for cuML/RAPIDS
# (needs cc >= 7.0); the default P100 partition is cc 6.0 and will NOT run cuML.
# SAM2 itself only needs any CUDA GPU, so this job runs on the 3090 regardless.

set -euo pipefail

# ------------------------------------------------------------------ config ---
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
ENV_NAME=${ENV_NAME:-Hospital}   # TartanGround environment (dir under data/tartanground)
TRAJ=${TRAJ:-P0000}              # trajectory (override: sbatch --export=ALL,TRAJ=P0001 ...)
CAMERA=${CAMERA:-lcam_front}     # camera  (override: sbatch --export=ALL,CAMERA=rcam_front ...)
MAX_FRAMES=${MAX_FRAMES:-512}          # 0 = all frames (P0000 has 2012); bounded for a first perf run
SNAPSHOT_EVERY=64          # flush + log the evolving cloud every N frames
SAM2_MODEL=facebook/sam2.1-hiera-base-plus   # base+ (heavier/better than -tiny)
SIGLIP_MODEL=google/siglip2-base-patch16-224 # appearance embedder for object association
POINTS_PER_SIDE=8          # auto-mask seed grid (NxN). 16 seeds far too many objects for a 640px scene
MIN_AREA=800               # drop tiny masks (per frame) before embedding
FRAME_STRIDE=${FRAME_STRIDE:-1}        # feed every k-th frame. k>1 = ~k x faster, but more inter-frame motion
VOXEL=0.1
PIXEL_STRIDE=2
MAX_DEPTH=60.0
# --- object association (Fuser) ---
BG_FADE=${BG_FADE:-0.5}    # fade non-mask crop pixels toward white before embedding
RADIUS=1.0                 # AABB prefilter slack (m)
MIN_OVERLAP=0.2            # min fraction of a mask's voxels overlapping an object to associate
MIN_CSIM=0.5               # min SigLIP embedding cosine to associate
MERGE_DILATE=2             # voxel reach for the object<->object merge pass (bridges fragments)
MERGE_CSIM=0.5             # min cosine to merge two existing objects
MERGE_OVERLAP=${MERGE_OVERLAP:-0.25}   # min contact frac to merge; low collapses connected scenes

RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"
RRD=$RUN_DIR/recon_${ENV_NAME}_${TRAJ}_${CAMERA}_stride${FRAME_STRIDE}.rrd

# reuse the already-downloaded SAM2 weights instead of re-fetching to $HOME
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
# keep sklearn/numpy from oversubscribing the allocated cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ------------------------------------------------------------- diagnostics ---
echo "=================================================================="
echo "job:        ${SLURM_JOB_ID:-manual}  on host $(hostname)"
echo "started:    $(date -Is)"
echo "git commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "target:     $ENV_NAME/$TRAJ/$CAMERA  max_frames=$MAX_FRAMES  snapshot_every=$SNAPSHOT_EVERY"
echo "rrd out:    $RRD"
echo "------------------------------------------------------------------"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || echo "nvidia-smi unavailable"
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PYEOF
echo "=================================================================="

# ------------------------------------------------------------------- run -----
cd "$REPO"
MAXF_ARG=""
[ "$MAX_FRAMES" -gt 0 ] && MAXF_ARG="--max-frames $MAX_FRAMES"

# GNU time isn't on every node (nodemm01 lacks it); use it if present, else the
# script self-reports wall time / peak RSS via its own PERF line + sacct MaxRSS.
TIMER=""
[ -x /usr/bin/time ] && TIMER="/usr/bin/time -v"

$TIMER "$PY" scripts/test_recon.py \
    --root "$REPO/data/tartanground" \
    --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
    $MAXF_ARG \
    --snapshot-every "$SNAPSHOT_EVERY" \
    --sam2-model "$SAM2_MODEL" \
    --siglip-model "$SIGLIP_MODEL" \
    --points-per-side "$POINTS_PER_SIDE" \
    --min-area "$MIN_AREA" \
    --frame-stride "$FRAME_STRIDE" \
    --voxel "$VOXEL" \
    --pixel-stride "$PIXEL_STRIDE" \
    --max-depth "$MAX_DEPTH" \
    --bg-fade "$BG_FADE" \
    --radius "$RADIUS" \
    --min-overlap "$MIN_OVERLAP" \
    --min-csim "$MIN_CSIM" \
    --merge-dilate "$MERGE_DILATE" \
    --merge-csim "$MERGE_CSIM" \
    --merge-overlap "$MERGE_OVERLAP" \
    --device cuda \
    --out "$RRD"

echo "------------------------------------------------------------------"
echo "finished:   $(date -Is)"
echo "rrd:        $RRD  ($(du -h "$RRD" 2>/dev/null | cut -f1))"
echo "render locally with:  rerun $RRD"
echo "=================================================================="
