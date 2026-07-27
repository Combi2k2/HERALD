#!/bin/bash
#SBATCH --job-name=herald-seg
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4    # 3090 partition allows max 4 CPUs per GPU
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# SAM2 segmentation ONLY (services.Sam2) over a whole video -> masks .rrd.
# No geometry, no reconstruction. Default per-frame AMG (track=False): no OOM.
# 3090 (cc 8.6): the P100 (cc 6.0) crashes the cu130 torch build.

set -euo pipefail

# ------------------------------------------------------------------ config ---
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
ENV_NAME=${ENV_NAME:-Hospital}
TRAJ=${TRAJ:-P0000}
CAMERA=${CAMERA:-lcam_front}
MAX_FRAMES=${MAX_FRAMES:-0}      # 0 = whole video
FRAME_STRIDE=${FRAME_STRIDE:-1}
SAM2_MODEL=${SAM2_MODEL:-facebook/sam2.1-hiera-base-plus}
POINTS_PER_SIDE=${POINTS_PER_SIDE:-16}
POINTS_PER_BATCH=${POINTS_PER_BATCH:-32}
MIN_AREA=${MIN_AREA:-200}
IOU_THRESH=${IOU_THRESH:-0.7}          # AMG pred_iou_thresh (lib default 0.88)
STABILITY_THRESH=${STABILITY_THRESH:-0.9}   # AMG stability_score_thresh (lib default 0.95)
TRACK=${TRACK:-0}               # 1 = video tracker (track=True, can OOM); 0 = per-frame AMG
CHUNK=${CHUNK:-16}
OFFLOAD=${OFFLOAD:-0}           # track only: 1=state on CPU, 0=on GPU

RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"
RRD=$RUN_DIR/seg_${ENV_NAME}_${TRAJ}_${CAMERA}_stride${FRAME_STRIDE}.rrd

# clear the leaked system PYTHONPATH so the uv venv's packages win
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ------------------------------------------------------------- diagnostics ---
echo "=================================================================="
echo "job:        ${SLURM_JOB_ID:-manual}  on host $(hostname)"
echo "started:    $(date -Is)"
echo "git commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "target:     $ENV_NAME/$TRAJ/$CAMERA  max_frames=$MAX_FRAMES track=$TRACK pps=$POINTS_PER_SIDE"
echo "rrd out:    $RRD"
echo "------------------------------------------------------------------"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "nvidia-smi unavailable"
echo "=================================================================="

# ------------------------------------------------------------------- run -----
cd "$REPO"
MAXF_ARG=""
[ "$MAX_FRAMES" -gt 0 ] && MAXF_ARG="--max-frames $MAX_FRAMES"
TRACK_ARG=""
[ "$TRACK" = "1" ] && TRACK_ARG="--track"
OFFLOAD_ARG=""
[ "$OFFLOAD" = "0" ] && OFFLOAD_ARG="--no-offload"

"$PY" scripts/segment_video.py \
    --root "$REPO/data/tartanground" \
    --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
    $MAXF_ARG \
    --frame-stride "$FRAME_STRIDE" \
    --sam2-model "$SAM2_MODEL" \
    --points-per-side "$POINTS_PER_SIDE" \
    --points-per-batch "$POINTS_PER_BATCH" \
    --min-area "$MIN_AREA" \
    --iou-thresh "$IOU_THRESH" \
    --stability-thresh "$STABILITY_THRESH" \
    $TRACK_ARG \
    --chunk "$CHUNK" \
    $OFFLOAD_ARG \
    --device cuda \
    --out "$RRD"

echo "------------------------------------------------------------------"
echo "finished:   $(date -Is)"
echo "rrd:        $RRD  ($(du -h "$RRD" 2>/dev/null | cut -f1))"
echo "render locally with:  rerun $RRD"
echo "=================================================================="
