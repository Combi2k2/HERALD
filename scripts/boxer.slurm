#!/bin/bash
#SBATCH --job-name=herald-boxer
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Tier-1 per-session recon (scripts/session_recon.py): Boxer's complete model
# (OWLv2 detector -> BoxerNet 3D lift -> online tracker w/ semantic merge) ->
# static scene cloud + static object OBBs, with in-video movers dropped by the
# corridor filter. Renders to Rerun.
# Boxer runs from HERALD's .venv (self-contained models; herald importable).

set -euo pipefail

REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
SCRIPT=${SCRIPT:-session_recon.py}   # or test_recon.py for the layered-panel diagnostic
ENV_NAME=${ENV_NAME:-Hospital}
TRAJ=${TRAJ:-P0001}
CAMERA=${CAMERA:-lcam_front}
MAX_FRAMES=${MAX_FRAMES:-0}
FRAME_STRIDE=${FRAME_STRIDE:-1}
CONF_THR2D=${CONF_THR2D:-0.40}
CONF_THR3D=${CONF_THR3D:-0.50}
IOU_THR=${IOU_THR:-0.3}
MIN_OBS=${MIN_OBS:-4}
MAX_DEPTH=${MAX_DEPTH:-60.0}
CORRIDOR_STEP=${CORRIDOR_STEP:-0.2}
MAX_RANGE=${MAX_RANGE:-15.0}
MIN_VISIBLE=${MIN_VISIBLE:-0.0}

RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"
RRD=$RUN_DIR/session_${ENV_NAME}_${TRAJ}_${CAMERA}_stride${FRAME_STRIDE}.rrd

unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=================================================================="
echo "job:      ${SLURM_JOB_ID:-manual} on $(hostname)   $(date -Is)"
echo "target:   $ENV_NAME/$TRAJ/$CAMERA  max=$MAX_FRAMES conf2d=$CONF_THR2D conf3d=$CONF_THR3D iou=$IOU_THR min_obs=$MIN_OBS range=$MAX_RANGE vis=$MIN_VISIBLE"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="

cd "$REPO"
MAXF_ARG=""
[ "$MAX_FRAMES" -gt 0 ] && MAXF_ARG="--max-frames $MAX_FRAMES"
DUMP_ARG=""
[ -n "${DUMP:-}" ] && DUMP_ARG="--dump $DUMP"     # session_recon.py: also save objects + cloud

"$PY" scripts/"$SCRIPT" \
    --root "$REPO/data/tartanground" \
    --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
    $MAXF_ARG --frame-stride "$FRAME_STRIDE" \
    --conf-thr2d "$CONF_THR2D" \
    --conf-thr3d "$CONF_THR3D" --iou-thr "$IOU_THR" --min-obs "$MIN_OBS" \
    --max-depth "$MAX_DEPTH" --corridor-step "$CORRIDOR_STEP" \
    --max-range "$MAX_RANGE" --min-visible "$MIN_VISIBLE" \
    --device cuda --out "$RRD" $DUMP_ARG

echo "------------------------------------------------------------------"
echo "finished: $(date -Is)   rrd: $RRD  ($(du -h "$RRD" 2>/dev/null | cut -f1))"
echo "=================================================================="
