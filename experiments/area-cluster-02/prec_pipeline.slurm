#!/bin/bash
#SBATCH --job-name=herald-precpipe
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Precision-tuned end-to-end (tag `prec5`): re-recon P0000/P0001/P0002 with a
# RAISED confidence floor (0.40/0.50, up from frustum5's 0.30/0.40), min_obs 5,
# 5m frustum, and the new visibility gate (>=60% of the reprojected 3D OBB must
# sit in-frame -> drops truncated edge lifts). Then object-merge -> route-merge ->
# overlap-dedup -> area cluster. One GPU job, serial.

set -euo pipefail
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

ENV_NAME=${ENV_NAME:-Hospital}
CAMERA=${CAMERA:-lcam_front}
TAG=${TAG:-prec5}
CONF_THR2D=${CONF_THR2D:-0.40}
CONF_THR3D=${CONF_THR3D:-0.50}
MIN_OBS=${MIN_OBS:-5}
MAX_RANGE=${MAX_RANGE:-5.0}
MIN_VISIBLE=${MIN_VISIBLE:-0.6}
IOS=${IOS:-0.5}
D_MAX=${D_MAX:-5}
LINKAGE=${LINKAGE:-average}

REC=$REPO/data/tartanground/$ENV_NAME/recon
MERGED=$REC/merged_${TAG}.npz
DEDUP=$REC/merged_${TAG}_dedup.npz
ART=$REPO/experiments/area-cluster-02/out/$TAG
ARTD=$REPO/experiments/area-cluster-02/out/${TAG}_dedup

echo "=================================================================="
echo "job ${SLURM_JOB_ID:-manual} on $(hostname) $(date -Is)"
echo "tag=$TAG conf=$CONF_THR2D/$CONF_THR3D min_obs=$MIN_OBS range=${MAX_RANGE}m vis=$MIN_VISIBLE | ios=$IOS d_max=$D_MAX linkage=$LINKAGE"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="
cd "$REPO"
mkdir -p "$ART" "$REC"

for TRAJ in P0000 P0001 P0002; do
  echo "----- RECON $TRAJ (tag $TAG) -----"
  "$PY" scripts/session_recon.py \
      --root "$REPO/data/tartanground" --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
      --conf-thr2d "$CONF_THR2D" --conf-thr3d "$CONF_THR3D" --min-obs "$MIN_OBS" \
      --max-range "$MAX_RANGE" --min-visible "$MIN_VISIBLE" \
      --device cuda --out "$REC/session_${TRAJ}_${TAG}.rrd" \
      --dump "$REC/session_${TRAJ}_${TAG}.npz"
done

echo "----- STAGE 0: object merge (3 prec dumps -> $MERGED) -----"
"$PY" scripts/merge_multi.py --no-embed \
    --sessions "$REC/session_P0000_${TAG}.npz" "$REC/session_P0001_${TAG}.npz" "$REC/session_P0002_${TAG}.npz" \
    --out "$MERGED"

echo "----- STAGE 1: route-graph merge (persist fused route map) -----"
"$PY" experiments/area-cluster-02/merge_routes.py --merged "$MERGED" --dump-tag "$TAG" --out-dir "$ART" --device cuda

echo "----- STAGE 2: overlap-dedup (fuse intersecting OBBs, ios>=$IOS) -----"
"$PY" experiments/area-cluster-02/dedup_objects.py --merged "$MERGED" --art-dir "$ART" --ios "$IOS" \
    --out "$DEDUP" --out-art "$ARTD"

echo "----- STAGE 3: area clustering (consume deduped artifacts) -----"
"$PY" experiments/area-cluster-02/cluster_areas.py --merged "$DEDUP" --art-dir "$ARTD" \
    --d-max "$D_MAX" --linkage "$LINKAGE" \
    --rrd "$ARTD/area_cluster_02_${TAG}_dedup_${LINKAGE}_d${D_MAX}.rrd"
echo "finished: $(date -Is)"
