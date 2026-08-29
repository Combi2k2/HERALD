#!/bin/bash
#SBATCH --job-name=herald-precanalysis
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Precision-tuned ANALYSIS (CPU half) -- no 3090 needed. Overlap-dedup is now folded INTO the
# object-merge run (merge_multi --overlap-ios), so there is no separate dedup stage:
#   object-merge (+in-merge dedup) -> route-merge (align on CPU, node-sets) -> area cluster.
# Submit with --dependency=afterok:<recon_jobid>.

set -euo pipefail
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_VISIBLE_DEVICES=""            # force CPU for torch (align)

ENV_NAME=${ENV_NAME:-Hospital}
TAG=${TAG:-prec5b}
OVERLAP_IOS=${OVERLAP_IOS:-0.5}
SPACING=${SPACING:-3.0}
RSNAP=${RSNAP:-1.0}
D_MAX=${D_MAX:-5}
LINKAGE=${LINKAGE:-average}
REC=$REPO/data/tartanground/$ENV_NAME/recon
MERGED=$REC/merged_${TAG}.npz
ART=$REPO/experiments/area-cluster-02/out/$TAG

echo "=================================================================="
echo "job ${SLURM_JOB_ID:-manual} on $(hostname) $(date -Is)  [CPU]"
echo "tag=$TAG overlap_ios=$OVERLAP_IOS spacing=${SPACING}m r_snap=${RSNAP}m d_max=$D_MAX linkage=$LINKAGE"
echo "=================================================================="
cd "$REPO"
mkdir -p "$ART"

echo "----- STAGE 0: object merge + in-merge overlap-dedup (3 dumps -> $MERGED) -----"
"$PY" scripts/merge_multi.py --no-embed --device cpu --no-align --overlap-ios "$OVERLAP_IOS" \
    --sessions "$REC/session_P0000_${TAG}.npz" "$REC/session_P0001_${TAG}.npz" "$REC/session_P0002_${TAG}.npz" \
    --out "$MERGED"

echo "----- STAGE 1: route-graph merge (align on CPU, node-sets; spacing=$SPACING r_snap=$RSNAP) -----"
"$PY" experiments/area-cluster-02/merge_routes.py --merged "$MERGED" --dump-tag "$TAG" --no-align \
    --spacing "$SPACING" --r-snap "$RSNAP" --out-dir "$ART" --device cpu

echo "----- STAGE 2: area clustering (consume merged + route artifacts) -----"
"$PY" experiments/area-cluster-02/cluster_areas.py --merged "$MERGED" --art-dir "$ART" \
    --d-max "$D_MAX" --linkage "$LINKAGE" \
    --rrd "$ART/area_cluster_02_${TAG}_${LINKAGE}_d${D_MAX}.rrd"
echo "finished: $(date -Is)"
