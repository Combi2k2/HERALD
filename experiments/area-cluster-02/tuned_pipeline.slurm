#!/bin/bash
#SBATCH --job-name=herald-tunedpipe
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Tuned end-to-end (after P0001/P0002 recon at frustum5): object-merge -> route-merge -> cluster.

set -euo pipefail
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
D_MAX=${D_MAX:-8}
REC=$REPO/data/tartanground/Hospital/recon
MERGED=$REC/merged_frustum5.npz
ART=$REPO/experiments/area-cluster-02/out/frustum5

echo "=================================================================="
echo "job ${SLURM_JOB_ID:-manual} on $(hostname) $(date -Is)  d_max=$D_MAX"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="
cd "$REPO"
mkdir -p "$ART"

echo "----- STAGE 0: object merge (3 tuned dumps -> merged_frustum5.npz) -----"
"$PY" scripts/merge_multi.py --no-embed \
    --sessions "$REC/session_P0000_frustum5.npz" "$REC/session_P0001_frustum5.npz" "$REC/session_P0002_frustum5.npz" \
    --out "$MERGED"

echo "----- STAGE 1: route-graph merge (persist fused route map) -----"
"$PY" experiments/area-cluster-02/merge_routes.py --merged "$MERGED" --dump-tag frustum5 --out-dir "$ART" --device cuda

echo "----- STAGE 2: area clustering (consume artifacts) -----"
"$PY" experiments/area-cluster-02/cluster_areas.py --merged "$MERGED" --art-dir "$ART" --d-max "$D_MAX" \
    --rrd "$ART/area_cluster_02_frustum5.rrd"
echo "finished: $(date -Is)"
