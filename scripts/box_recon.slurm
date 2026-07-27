#!/bin/bash
#SBATCH --job-name=herald-objrec
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Object-level reconstruction: detection boxes -> bg-removed clouds -> 3D-overlap
# merge -> per-object OBB on the fly (scripts/box_recon.py). No masks, no SAM2.

set -euo pipefail

REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
ENV_NAME=${ENV_NAME:-Hospital}
TRAJ=${TRAJ:-P0001}
CAMERA=${CAMERA:-lcam_front}
MAX_FRAMES=${MAX_FRAMES:-0}
FRAME_STRIDE=${FRAME_STRIDE:-1}
MODEL=${MODEL:-yolov8x-worldv2.pt}
CONF=${CONF:-0.1}
PIXEL_STRIDE=${PIXEL_STRIDE:-3}
DBSCAN_EPS=${DBSCAN_EPS:-0.15}
MIN_SAMPLES=${MIN_SAMPLES:-4}
MIN_PTS=${MIN_PTS:-30}
FLOOR_MARGIN=${FLOOR_MARGIN:-0.08}
MAX_DEPTH=${MAX_DEPTH:-60.0}
MERGE_VOXEL=${MERGE_VOXEL:-0.1}
MERGE_OVERLAP=${MERGE_OVERLAP:-0.2}
MIN_VOX=${MIN_VOX:-5}
MIN_OBS=${MIN_OBS:-2}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-64}
CLASS_GATE=${CLASS_GATE:-1}          # 0 => --no-class-gate

RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"
RRD=$RUN_DIR/objrecon_${ENV_NAME}_${TRAJ}_${CAMERA}_stride${FRAME_STRIDE}.rrd

unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-$HOME/.config/Ultralytics}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=================================================================="
echo "job:        ${SLURM_JOB_ID:-manual}  on host $(hostname)"
echo "started:    $(date -Is)"
echo "target:     $ENV_NAME/$TRAJ/$CAMERA  max_frames=$MAX_FRAMES conf=$CONF merge_overlap=$MERGE_OVERLAP class_gate=$CLASS_GATE"
echo "rrd out:    $RRD"
echo "------------------------------------------------------------------"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "nvidia-smi unavailable"
echo "=================================================================="

cd "$REPO"
MAXF_ARG=""
[ "$MAX_FRAMES" -gt 0 ] && MAXF_ARG="--max-frames $MAX_FRAMES"
CLASS_ARG=""
[ "$CLASS_GATE" = "0" ] && CLASS_ARG="--no-class-gate"

"$PY" scripts/box_recon.py \
    --root "$REPO/data/tartanground" \
    --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
    $MAXF_ARG \
    --frame-stride "$FRAME_STRIDE" \
    --model "$MODEL" --conf "$CONF" \
    --pixel-stride "$PIXEL_STRIDE" --max-depth "$MAX_DEPTH" \
    --dbscan-eps "$DBSCAN_EPS" --min-samples "$MIN_SAMPLES" --min-pts "$MIN_PTS" \
    --floor-margin "$FLOOR_MARGIN" \
    --merge-voxel "$MERGE_VOXEL" --merge-overlap "$MERGE_OVERLAP" \
    --min-vox "$MIN_VOX" --min-obs "$MIN_OBS" \
    --snapshot-every "$SNAPSHOT_EVERY" \
    $CLASS_ARG \
    --device cuda \
    --out "$RRD"

echo "------------------------------------------------------------------"
echo "finished:   $(date -Is)"
echo "rrd:        $RRD  ($(du -h "$RRD" 2>/dev/null | cut -f1))"
echo "render locally with:  rerun $RRD"
echo "=================================================================="
