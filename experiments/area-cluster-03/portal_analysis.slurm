#!/bin/bash
#SBATCH --job-name=herald-portalanalysis
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# area-cluster-03 ANALYSIS (CPU): consume the portal5 recon dumps ->
#   object-merge (+in-merge dedup) -> route-merge (node-sets) -> PORTAL-bounded area cluster.
# Submit with --dependency=afterok:<portal5_recon_jobid>.

set -euo pipefail
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_VISIBLE_DEVICES=""

ENV_NAME=${ENV_NAME:-Hospital}
TAG=${TAG:-portal5}
OVERLAP_IOS=${OVERLAP_IOS:-0.5}
SPACING=${SPACING:-3.0}
RSNAP=${RSNAP:-1.0}
D_MAX=${D_MAX:-5}
LINKAGE=${LINKAGE:-average}
GATE_MARGIN=${GATE_MARGIN:-0.3}
MIRROR_ADJ=${MIRROR_ADJ:-1.5}
REC=$REPO/data/tartanground/$ENV_NAME/recon
MERGED=$REC/merged_${TAG}.npz
ART=$REPO/experiments/area-cluster-03/out/$TAG

echo "=================================================================="
echo "job ${SLURM_JOB_ID:-manual} on $(hostname) $(date -Is)  [CPU]"
echo "tag=$TAG spacing=${SPACING}m r_snap=${RSNAP}m d_max=$D_MAX linkage=$LINKAGE gate_margin=$GATE_MARGIN"
echo "=================================================================="
cd "$REPO"
mkdir -p "$ART"

echo "----- STAGE 0: object merge + in-merge dedup -----"
"$PY" scripts/merge_multi.py --no-embed --device cpu --no-align --overlap-ios "$OVERLAP_IOS" \
    --sessions "$REC/session_P0000_${TAG}.npz" "$REC/session_P0001_${TAG}.npz" "$REC/session_P0002_${TAG}.npz" \
    --out "$MERGED"

echo "----- STAGE 1: route-graph merge (node-sets) -----"
"$PY" experiments/area-cluster-02/merge_routes.py --merged "$MERGED" --dump-tag "$TAG" --no-align \
    --spacing "$SPACING" --r-snap "$RSNAP" --out-dir "$ART" --device cpu

echo "----- STAGE 2: PORTAL-bounded area cluster -----"
"$PY" experiments/area-cluster-03/portal_cluster.py --merged "$MERGED" --art-dir "$ART" \
    --d-max "$D_MAX" --linkage "$LINKAGE" --gate-margin "$GATE_MARGIN" --mirror-adj "$MIRROR_ADJ" \
    --rrd "$ART/portal_cluster_${TAG}_d${D_MAX}.rrd"
echo "finished: $(date -Is)"
