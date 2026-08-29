#!/bin/bash
#SBATCH --job-name=herald-routemerge
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Stage 1 (GPU): merge_routes.py -> persist fused RouteGraph + object attachment + Sim3.
# Stage 2 (CPU): cluster_areas.py consumes those artifacts -> area clustering + rrd.

set -euo pipefail
REPO=/home/infres/dnguyen-25/HERALD
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
D_MAX=${D_MAX:-8}
PY=$REPO/.venv/bin/python

echo "=================================================================="
echo "job ${SLURM_JOB_ID:-manual} on $(hostname) $(date -Is)  d_max=$D_MAX"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="
cd "$REPO"
echo "----- STAGE 1: route-graph merge (persist artifacts) -----"
"$PY" experiments/area-cluster-02/merge_routes.py --device cuda
echo "----- STAGE 2: area clustering (consume artifacts) -----"
"$PY" experiments/area-cluster-02/cluster_areas.py --d-max "$D_MAX"
echo "finished: $(date -Is)"
