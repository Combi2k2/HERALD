#!/bin/bash
#SBATCH --job-name=herald-areamulti
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Multi-session area clustering: Sim3-align P0001/P0002 onto P0000, fuse route graphs,
# attach the merged map's 307 objects, complete-linkage geodesic areas. Needs GPU (refine.align).

set -euo pipefail
REPO=/home/infres/dnguyen-25/HERALD
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
D_MAX=${D_MAX:-8}

echo "=================================================================="
echo "job ${SLURM_JOB_ID:-manual} on $(hostname) $(date -Is)  d_max=$D_MAX"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="
cd "$REPO"
"$REPO/.venv/bin/python" experiments/area-cluster-02/multi.py --d-max "$D_MAX" --device cuda
echo "finished: $(date -Is)"
