#!/bin/bash
#SBATCH --job-name=herald-det
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4    # 3090 partition allows max 4 CPUs per GPU
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Open-vocab detection ONLY (YOLO-World / ultralytics) over a whole video -> boxes .rrd.
# No geometry, no segmentation, no reconstruction. 3090 (cc 8.6); P100 crashes cu130 torch.

set -euo pipefail

REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
ENV_NAME=${ENV_NAME:-Hospital}
TRAJ=${TRAJ:-P0000}
CAMERA=${CAMERA:-lcam_front}
MAX_FRAMES=${MAX_FRAMES:-0}      # 0 = whole video
FRAME_STRIDE=${FRAME_STRIDE:-1}
MODEL=${MODEL:-yolov8x-worldv2.pt}
CONF=${CONF:-0.1}
VOCAB_FILE=${VOCAB_FILE:-}       # optional: one class name per line (else script default)

RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"
RRD=$RUN_DIR/detect_${ENV_NAME}_${TRAJ}_${CAMERA}_stride${FRAME_STRIDE}.rrd

unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-$HOME/.config/Ultralytics}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=================================================================="
echo "job:        ${SLURM_JOB_ID:-manual}  on host $(hostname)"
echo "started:    $(date -Is)"
echo "target:     $ENV_NAME/$TRAJ/$CAMERA  max_frames=$MAX_FRAMES model=$MODEL conf=$CONF"
echo "rrd out:    $RRD"
echo "------------------------------------------------------------------"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "nvidia-smi unavailable"
echo "=================================================================="

cd "$REPO"
MAXF_ARG=""
[ "$MAX_FRAMES" -gt 0 ] && MAXF_ARG="--max-frames $MAX_FRAMES"
VOCAB_ARG=""
[ -n "$VOCAB_FILE" ] && VOCAB_ARG="--vocab-file $VOCAB_FILE"

"$PY" scripts/detect_video.py \
    --root "$REPO/data/tartanground" \
    --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
    $MAXF_ARG \
    --frame-stride "$FRAME_STRIDE" \
    --model "$MODEL" \
    --conf "$CONF" \
    $VOCAB_ARG \
    --device cuda \
    --out "$RRD"

echo "------------------------------------------------------------------"
echo "finished:   $(date -Is)"
echo "rrd:        $RRD  ($(du -h "$RRD" 2>/dev/null | cut -f1))"
echo "render locally with:  rerun $RRD"
echo "=================================================================="
