#!/bin/bash
#SBATCH --job-name=herald-track
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4    # 3090 partition allows max 4 CPUs per GPU
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# SAM2 video tracking + merge-free cloud (scripts/test_track_recon.py).
# 3090 (cc 8.6): the P100 (cc 6.0) crashes the cu130 torch build. SAM2 needs any
# CUDA GPU; the video model + AMG both live on the GPU, so give it the 3090.

set -euo pipefail

# ------------------------------------------------------------------ config ---
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
ENV_NAME=${ENV_NAME:-Hospital}   # TartanGround environment (dir under data/tartanground)
TRAJ=${TRAJ:-P0000}              # trajectory (override: sbatch --export=ALL,TRAJ=P0001 ...)
CAMERA=${CAMERA:-lcam_front}
MAX_FRAMES=${MAX_FRAMES:-0}      # 0 = whole video (chunking bounds memory regardless)
FRAME_STRIDE=${FRAME_STRIDE:-1}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-64}
SAM2_MODEL=facebook/sam2.1-hiera-base-plus
CHUNK=${CHUNK:-16}                      # frames per SAM2 session (bounds memory)
POINTS_PER_SIDE=${POINTS_PER_SIDE:-16}  # AMG seed grid; higher => finer masks
MIN_AREA=${MIN_AREA:-400}
OFFLOAD=${OFFLOAD:-0}                    # 1 = offload SAM2 state to CPU (slow on base-plus); 0 = keep on GPU
# --- cross-chunk 3D voxel-overlap merge ---
MERGE_VOXEL=${MERGE_VOXEL:-0.1}         # world voxel size for co-occupancy matching
MERGE_OVERLAP=${MERGE_OVERLAP:-0.2}     # min fraction of a chunk object's voxels on a global to merge
# --- flicker control / voting ---
BORDER=${BORDER:-12}                    # drop pixels within N px of the frame edge
EDGE_RTOL=${EDGE_RTOL:-0.03}            # depth-discontinuity (flying-pixel) filter; 0=off
ERODE=${ERODE:-1}
SEM_CONF=${SEM_CONF:-0.0}               # min SAM2 mask confidence to vote; 0=off
MIN_CLUSTER=${MIN_CLUSTER:-10}
MIN_FRAMES=${MIN_FRAMES:-1}
VOXEL=${VOXEL:-0.05}
PIXEL_STRIDE=${PIXEL_STRIDE:-2}
MAX_DEPTH=${MAX_DEPTH:-60.0}

RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"
RRD=$RUN_DIR/track_${ENV_NAME}_${TRAJ}_${CAMERA}_stride${FRAME_STRIDE}.rrd

# clear the leaked system PYTHONPATH so the uv venv's packages win
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# reduce fragmentation of the growing per-object video memory (see OOM guidance)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ------------------------------------------------------------- diagnostics ---
echo "=================================================================="
echo "job:        ${SLURM_JOB_ID:-manual}  on host $(hostname)"
echo "started:    $(date -Is)"
echo "git commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "target:     $ENV_NAME/$TRAJ/$CAMERA  max_frames=$MAX_FRAMES chunk=$CHUNK pps=$POINTS_PER_SIDE voxel=$VOXEL"
echo "rrd out:    $RRD"
echo "------------------------------------------------------------------"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "nvidia-smi unavailable"
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
OFFLOAD_ARG=""
[ "$OFFLOAD" = "0" ] && OFFLOAD_ARG="--no-offload"

"$PY" scripts/test_track_recon.py \
    --root "$REPO/data/tartanground" \
    --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
    $MAXF_ARG \
    --frame-stride "$FRAME_STRIDE" \
    --snapshot-every "$SNAPSHOT_EVERY" \
    --sam2-model "$SAM2_MODEL" \
    --chunk "$CHUNK" \
    --points-per-side "$POINTS_PER_SIDE" \
    --min-area "$MIN_AREA" \
    --merge-voxel "$MERGE_VOXEL" \
    --merge-overlap "$MERGE_OVERLAP" \
    $OFFLOAD_ARG \
    --border "$BORDER" \
    --edge-rtol "$EDGE_RTOL" \
    --erode "$ERODE" \
    --sem-conf "$SEM_CONF" \
    --min-cluster "$MIN_CLUSTER" \
    --min-frames "$MIN_FRAMES" \
    --voxel "$VOXEL" \
    --pixel-stride "$PIXEL_STRIDE" \
    --max-depth "$MAX_DEPTH" \
    --device cuda \
    --out "$RRD"

echo "------------------------------------------------------------------"
echo "finished:   $(date -Is)"
echo "rrd:        $RRD  ($(du -h "$RRD" 2>/dev/null | cut -f1))"
echo "render locally with:  rerun $RRD"
echo "=================================================================="
