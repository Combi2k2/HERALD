#!/bin/bash
# Full HERALD Phase-2 pipeline on Jean Zay: per trajectory in $TRAJS, recon (GPU) -> align ->
# track_obj -> merge_route (CPU); then once: area_cluster -> area_classify (VLM) -> render.
# Promoted from the ad hoc repro_pipeline.slurm this was validated against.
#
# Usage: sbatch --export=ALL,ENV_NAME=Hospital,TRAJS="P0000 P0001 P0002" scripts/scene_pipeline.slurm
#SBATCH --job-name=herald-pipeline
#SBATCH --account=hsl@a100
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:45:00
#SBATCH --output=runs/%j/slurm.out
#SBATCH --error=runs/%j/slurm.err

set -uo pipefail
unset PYTHONPATH   # avoid system site-packages shadowing the uv venv (CLAUDE.md)

cd "${SLURM_SUBMIT_DIR}"
export UV_CACHE_DIR="$SCRATCH/uv_cache"
# No python3.11-devel (no Python.h) on this cluster's login or GPU compute nodes, so Triton
# can't JIT-compile the CUDA launcher stub torch.compile() needs. services/boxer.py's
# OwlWrapper unconditionally calls torch.compile() for device="cuda", which otherwise
# hard-crashes SessionRecon.__init__ before any frame is processed. Env-level workaround, no
# repo code changed -- runs vision_detector eagerly instead of compiled.
export TORCHDYNAMO_DISABLE=1

ENV_NAME="${ENV_NAME:-Hospital}"
TRAJS="${TRAJS:-P0000 P0001}"
REC="data/tartanground/${ENV_NAME}/recon"
S="data/tartanground/${ENV_NAME}/scene"
mkdir -p "$REC" "$S"

RUN="runs/${SLURM_JOB_ID}"
mkdir -p "$RUN"
TIMING="$RUN/timing.csv"
echo "stage,traj,seconds,exit_code" > "$TIMING"

nvidia-smi -L || true
uv run --python .venv python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

run_stage () {
  local stage="$1"; local traj="$2"; shift 2
  echo "=== ${stage} ${traj} : $(date -Is) ==="
  local t0 t1 dt rc
  t0=$(date +%s.%N)
  uv run --python .venv python "$@" 2>&1 | tee "$RUN/${stage}_${traj}.log"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s.%N)
  dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  echo "${stage},${traj},${dt},${rc}" >> "$TIMING"
  echo "=== ${stage} ${traj} done in ${dt}s (exit ${rc}) ==="
  return "$rc"
}

first=1
for T in $TRAJS; do
  run_stage scene_recon "$T" scripts/scene_recon.py \
    --env "$ENV_NAME" --traj "$T" --out "${REC}/session_${T}.npz" || exit 1

  if [ "$first" = 1 ]; then
    # first trajectory seeds the map: identity Sim3 (no persistent map exists yet regardless
    # of --no-align). TartanGround poses are already globally co-registered, so later
    # trajectories' real align() call is a functionality check of registration, not a recovery
    # of a genuinely unregistered session.
    run_stage scene_align "$T" scripts/scene_align.py \
      --session "${REC}/session_${T}.npz" --persistent "${S}/persistent.npz" \
      --out "${S}/sim3_${T}.json" --no-align || exit 1
    first=0
  else
    run_stage scene_align "$T" scripts/scene_align.py \
      --session "${REC}/session_${T}.npz" --persistent "${S}/persistent.npz" \
      --out "${S}/sim3_${T}.json" --device cuda || exit 1
  fi
  run_stage scene_track_obj "$T" scripts/scene_track_obj.py \
    --session "${REC}/session_${T}.npz" --persistent "${S}/persistent.npz" \
    --sim3 "${S}/sim3_${T}.json" || exit 1
  run_stage scene_merge_route "$T" scripts/scene_merge_route.py \
    --objects "${S}/persistent.npz" --route "${S}/route.json" --attach "${S}/attach.json" \
    --session "$T" --sim3 "${S}/sim3_${T}.json" || exit 1
done

# after ALL sessions: portal-bounded area clustering, then VLM area naming (needs an Ollama
# server -- caller is expected to have started one, e.g. the pattern in
# experiments/*/run_*.slurm; skip --out overwrite of scene_graph.json if VLM isn't available)
run_stage scene_area_cluster ALL scripts/scene_area_cluster.py \
  --objects "${S}/persistent.npz" --route "${S}/route.json" --attach "${S}/attach.json" \
  --out "${S}/scene_graph.json" || exit 1

run_stage scene_extract_contour ALL scripts/scene_extract_contour.py \
  --persistent "${S}/persistent.npz" --out "${S}/contour.json" || exit 1

run_stage scene_render ALL scripts/scene_render.py \
  --objects "${S}/persistent.npz" --graph "${S}/scene_graph.json" --route "${S}/route.json" \
  --out "${S}/scene.rrd" || exit 1

echo "ALL STAGES DONE"
cat "$TIMING"
