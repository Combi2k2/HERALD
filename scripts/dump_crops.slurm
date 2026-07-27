#!/bin/bash
#SBATCH --job-name=herald-crops
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4    # 3090 partition allows max 4 CPUs per GPU
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/dnguyen-25/HERALD/runs/%x-%j.out
#SBATCH --error=/home/infres/dnguyen-25/HERALD/runs/%x-%j.err
# Dumps SAM2 mask crops (bbox + bg_fade, exactly as the Fuser embeds them) from
# the first few frames as PNGs, to eyeball SigLIP csim later. SAM2 only needs any
# CUDA GPU; the 3090 partition is just to avoid the P100 (cu130 torch crashes on it).

set -euo pipefail

# ------------------------------------------------------------------ config ---
REPO=/home/infres/dnguyen-25/HERALD
PY=$REPO/.venv/bin/python
ENV_NAME=${ENV_NAME:-Hospital}
TRAJ=${TRAJ:-P0001}
CAMERA=${CAMERA:-lcam_front}
N_FRAMES=${N_FRAMES:-4}          # number of frames to process
STRIDE=${STRIDE:-4}              # gap between processed frames
SAM2_MODEL=facebook/sam2.1-hiera-base-plus
SIGLIP_MODEL=google/siglip2-base-patch16-224
POINTS_PER_SIDE=${POINTS_PER_SIDE:-8}
MIN_AREA=${MIN_AREA:-800}
PAD_FRAC=${PAD_FRAC:-0.5}         # extra pad as frac of bbox size per side (0.5 = +H/2,+W/2 context)
BG_FADE=${BG_FADE:-0.8}           # fade non-mask crop pixels toward black (0=raw, 1=solid black)
MAX_PER_FRAME=${MAX_PER_FRAME:-12}
EMBED=${EMBED:-1}                # 1 = also run SigLIP -> embeddings.npz + cosine matrix
OUT=${OUT:-$REPO/runs/${SLURM_JOB_ID:-manual}/crops}

# clear the leaked system PYTHONPATH so the uv venv's packages win
# (system huggingface_hub otherwise shadows the venv's and breaks transformers)
unset PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# ------------------------------------------------------------- diagnostics ---
echo "=================================================================="
echo "job:        ${SLURM_JOB_ID:-manual}  on host $(hostname)"
echo "started:    $(date -Is)"
echo "git commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "target:     $ENV_NAME/$TRAJ/$CAMERA  n_frames=$N_FRAMES stride=$STRIDE  pad_frac=$PAD_FRAC bg_fade=$BG_FADE"
echo "out:        $OUT"
echo "------------------------------------------------------------------"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || echo "nvidia-smi unavailable"
echo "=================================================================="

# ------------------------------------------------------------------- run -----
cd "$REPO"
EMBED_ARG=""
[ "$EMBED" = "1" ] && EMBED_ARG="--embed"

"$PY" scripts/dump_crops.py \
    --root "$REPO/data/tartanground" \
    --env "$ENV_NAME" --traj "$TRAJ" --camera "$CAMERA" \
    --n-frames "$N_FRAMES" --stride "$STRIDE" \
    --sam2-model "$SAM2_MODEL" \
    --siglip-model "$SIGLIP_MODEL" \
    --points-per-side "$POINTS_PER_SIDE" \
    --min-area "$MIN_AREA" \
    --pad-frac "$PAD_FRAC" \
    --bg-fade "$BG_FADE" \
    --max-per-frame "$MAX_PER_FRAME" \
    $EMBED_ARG \
    --device cuda \
    --out "$OUT"

echo "------------------------------------------------------------------"
echo "finished:   $(date -Is)"
echo "crops in:   $OUT  ($(ls "$OUT"/*.png 2>/dev/null | wc -l) pngs)"
echo "=================================================================="
