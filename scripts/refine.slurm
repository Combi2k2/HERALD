#!/bin/bash
#SBATCH --job-name=herald-refine
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Tier-2 refine cloud alignment (scripts/test_refine.py): energy-based Sim(3)
# registration of two session clouds. Default = synthetic closed-loop self-test;
# pass P1=... P2=... to align two real .npz/.ply clouds.

set -euo pipefail

REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
ITERS=${ITERS:-80}
MAX_POINTS=${MAX_POINTS:-12000}
SEED=${SEED:-0}
EXTRA=${EXTRA:-}                       # e.g. "--optimize-scale --tilt 8"

RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"
RRD=$RUN_DIR/test_refine.rrd

unset PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=================================================================="
echo "job:      ${SLURM_JOB_ID:-manual} on $(hostname)   $(date -Is)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="

cd "$REPO"
PAIR_ARGS=""
[ -n "${P1:-}" ] && [ -n "${P2:-}" ] && PAIR_ARGS="--p1 $P1 --p2 $P2"

"$PY" scripts/test_refine.py \
    $PAIR_ARGS \
    --iters "$ITERS" --max-points "$MAX_POINTS" \
    --seed "$SEED" --device cuda --out "$RRD" $EXTRA

echo "------------------------------------------------------------------"
echo "finished: $(date -Is)   rrd: $RRD"
echo "=================================================================="
