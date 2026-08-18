#!/bin/bash
#SBATCH --job-name=herald-align
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Tier-2 refine alignment test (scripts/test_align.py): perturb the source cloud by a
# known Sim3 (yaw + xy-tilt + translation + scale) and recover it. Override the
# perturbation / clouds via EXTRA, e.g.:
#   sbatch --export=ALL,EXTRA="--yaw 60 --tilt-x 8 --scale 1.2" scripts/align.slurm

set -euo pipefail
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
EXTRA=${EXTRA:-}
RUN_DIR=$REPO/runs/${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR"

unset PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "job ${SLURM_JOB_ID:-manual} on $(hostname)  $(date -Is)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"

cd "$REPO"
"$PY" scripts/test_align.py --device cuda --out "$RUN_DIR/test_align.rrd" $EXTRA

echo "finished: $(date -Is)   rrd: $RUN_DIR/test_align.rrd"
