#!/bin/bash
#SBATCH --job-name=herald-merge
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Tier-2 refine on two real Hospital sessions: build each cloud from RGB-D (no Boxer),
# then energy-field align P2->P1, animate the Sim(3) optimisation, and merge+voxelise.

set -euo pipefail

REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
ENV_NAME=${ENV_NAME:-Hospital}
TRAJ1=${TRAJ1:-P0000}          # target
TRAJ2=${TRAJ2:-P0001}          # source (aligned onto target)
CAMERA=${CAMERA:-lcam_front}
STRIDE=${STRIDE:-2}
MAX_DEPTH=${MAX_DEPTH:-40.0}
VOXEL=${VOXEL:-0.1}
ITERS=${ITERS:-80}
EXTRA=${EXTRA:-}

CDIR=$REPO/data/tartanground/$ENV_NAME/recon
RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$CDIR" "$RUN_DIR"

unset PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=================================================================="
echo "job:  ${SLURM_JOB_ID:-manual} on $(hostname)  $(date -Is)"
echo "pair: $ENV_NAME $TRAJ1 (target) <- $TRAJ2 (source)  stride=$STRIDE voxel=$VOXEL"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="
cd "$REPO"

for T in "$TRAJ1" "$TRAJ2"; do
    if [ ! -f "$CDIR/$T.npz" ]; then
        echo ">>> building cloud for $T ..."
        "$PY" scripts/build_cloud.py --env "$ENV_NAME" --traj "$T" --camera "$CAMERA" \
            --stride "$STRIDE" --max-depth "$MAX_DEPTH" --voxel "$VOXEL" --out "$CDIR/$T.ply"
    else
        echo ">>> cloud for $T exists, skipping build"
    fi
done

echo ">>> aligning + merging ..."
"$PY" scripts/merge_sessions.py \
    --p1 "$CDIR/$TRAJ1.npz" --p2 "$CDIR/$TRAJ2.npz" \
    --voxel "$VOXEL" --iters "$ITERS" \
    --device cuda --out "$RUN_DIR/merged_${TRAJ1}_${TRAJ2}.ply" \
    --rrd "$RUN_DIR/merge_${TRAJ1}_${TRAJ2}.rrd" $EXTRA

echo "------------------------------------------------------------------"
echo "finished: $(date -Is)   out: $RUN_DIR"
echo "=================================================================="
