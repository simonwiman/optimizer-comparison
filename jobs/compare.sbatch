#!/bin/bash
#SBATCH --job-name=nanochat-cmp
#SBATCH --partition=gpu-long
#SBATCH --time=01:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

# One optimizer variant per job, so the runs execute in parallel on identical settings.
# VARIANT selects the MuonEq / Muon^2 combination (all other Muon additions stay on):
#   eq     MuonEq on,  Muon^2 off   (= standard nanochat)
#   eq_m2  MuonEq on,  Muon^2 on
#   m2     MuonEq off, Muon^2 on
#   base   MuonEq off, Muon^2 off
#   for v in eq eq_m2 m2 base; do sbatch --export=ALL,VARIANT=$v compare.sbatch; done
#
# Optional overrides via --export=ALL,KEY=value,...
#   DEPTH (8)  NUM_ITERS (1500)  TOTAL_BATCH (262144)  EVAL_EVERY (250)
#   MATRIX_LR (unset = nanochat default)  MUON2_BETA2 (0.95)  MUON2_EPS (1e-8)
#   TAG (suffix for model tag / log, e.g. _lr0.03, for sweeps)
#
# Run length is fixed by NUM_ITERS * TOTAL_BATCH tokens (default ~393M tokens), NOT by wall-clock.
# If the first job's log shows a very different ETA than ~15 min, adjust NUM_ITERS for ALL variants.

set -euo pipefail

VARIANT="${VARIANT:?set VARIANT=eq|eq_m2|m2|base via --export=ALL,VARIANT=...}"
DEPTH="${DEPTH:-8}"
NUM_ITERS="${NUM_ITERS:-1500}"
TOTAL_BATCH="${TOTAL_BATCH:-262144}"
EVAL_EVERY="${EVAL_EVERY:-250}"
MUON2_BETA2="${MUON2_BETA2:-0.95}"
MUON2_EPS="${MUON2_EPS:-1e-8}"
TAG="${TAG:-}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-d${DEPTH}-${VARIANT}${TAG}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1

ARGS=(
    --depth="$DEPTH"
    --run=dummy
    --model-tag="cmp_d${DEPTH}_${VARIANT}${TAG}"
    --num-iterations="$NUM_ITERS"
    --total-batch-size="$TOTAL_BATCH"
    --eval-every="$EVAL_EVERY"
    --eval-tokens=5242880          # 10 x 524288 val tokens per eval (default is 80x, too slow here)
    --core-metric-every=-1         # CORE metric is mostly noise at this scale; disabled
    --sample-every=-1
)

if [[ -n "${MATRIX_LR:-}" ]]; then
    ARGS+=(--matrix-lr="$MATRIX_LR")
fi

M2_ARGS=(--muon2 --muon2-beta2="$MUON2_BETA2" --muon2-eps="$MUON2_EPS")
case "$VARIANT" in
    eq)    ARGS+=(--muoneq) ;;
    eq_m2) ARGS+=(--muoneq "${M2_ARGS[@]}") ;;
    m2)    ARGS+=(--no-muoneq "${M2_ARGS[@]}") ;;
    base)  ARGS+=(--no-muoneq) ;;
    *) echo "Unknown VARIANT='$VARIANT' (expected: eq | eq_m2 | m2 | base)" >&2; exit 1 ;;
esac

echo "Variant: $VARIANT | depth=$DEPTH iters=$NUM_ITERS batch=$TOTAL_BATCH"
python -m scripts.base_train "${ARGS[@]}"