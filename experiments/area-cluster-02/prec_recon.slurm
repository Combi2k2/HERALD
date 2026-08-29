#!/bin/bash
#SBATCH --job-name=herald-precrecon
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Precision-tuned DETECTION only (tag `prec5`), the GPU-bound half.
# Re-recon P0000/P0001/P0002 with raised confidence (0.40/0.50), min_obs 5, 5m
# frustum, visibility gate (>=0.6 of the reprojected 3D OBB in-frame), and now
# per-track observing-frame persistence (sessions[sid]["frames"]) for multi-anchor
# attachment. Dumps session_<T>_prec5.npz; merge/cluster run separately on CPU.

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
REC=$REPO/data/tartanground/$ENV_NAME/recon

echo "=================================================================="
echo "job ${SLURM_JOB_ID:-manual} on $(hostname) $(date -Is)"
echo "tag=$TAG conf=$CONF_THR2D/$CONF_THR3D min_obs=$MIN_OBS range=${MAX_RANGE}m vis=$MIN_VISIBLE"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="
cd "$REPO"
mkdir -p "$REC"

for TRAJ in P0000 P0001 P0002; do
  echo "----- RECON $TRAJ (tag $TAG) -----"
  "$PY" scripts/session_recon.py \
      --root "$REPO/data/tartanground" --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
      --conf-thr2d "$CONF_THR2D" --conf-thr3d "$CONF_THR3D" --min-obs "$MIN_OBS" \
      --max-range "$MAX_RANGE" --min-visible "$MIN_VISIBLE" \
      --device cuda --out "$REC/session_${TRAJ}_${TAG}.rrd" \
      --dump "$REC/session_${TRAJ}_${TAG}.npz"
done
echo "finished recon: $(date -Is)"
