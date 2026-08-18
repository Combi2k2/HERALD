#!/bin/bash
#SBATCH --job-name=herald-embed
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Backfill text-label embeddings onto a saved SceneMap (scripts/embed_map.py):
# vote-weighted mean of label vectors via the text-only TextEncoder, in place.
# Light job -- one small model download + ~30 short strings.

set -euo pipefail

REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
MAP=${MAP:-runs/merge_multi3/merged.npz}
EMBED_MODEL=${EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}

unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=================================================================="
echo "job:   ${SLURM_JOB_ID:-manual} on $(hostname)   $(date -Is)"
echo "map:   $MAP   model: $EMBED_MODEL"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "no nvidia-smi"
echo "=================================================================="

cd "$REPO"
"$PY" scripts/embed_map.py --map "$MAP" --embed-model "$EMBED_MODEL" --device cuda

echo "------------------------------------------------------------------"
echo "finished: $(date -Is)"
echo "=================================================================="
