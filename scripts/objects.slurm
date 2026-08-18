#!/bin/bash
#SBATCH --job-name=herald-objmerge
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Tier-2 object reconciliation (scripts/merge_objects.py): align two recon'd sessions and
# fuse their OBBs into one persistent set.

set -euo pipefail
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
CDIR=$REPO/data/tartanground/Hospital/recon
A=${A:-$CDIR/session_P0000.npz}
B=${B:-$CDIR/session_P0001.npz}
EXTRA=${EXTRA:-}
RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"

unset PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "job ${SLURM_JOB_ID:-manual} on $(hostname)  $(date -Is)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"

cd "$REPO"
"$PY" scripts/merge_objects.py --a "$A" --b "$B" --device cuda \
    --out "$RUN_DIR/merged_objects.npz" --rrd "$RUN_DIR/merged_objects.rrd" $EXTRA

echo "finished: $(date -Is)   out: $RUN_DIR"
