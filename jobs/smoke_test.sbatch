#!/bin/bash
#SBATCH --job-name=nanochat-test
#SBATCH --partition=gpu-long
#SBATCH --time=00:30:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

# VARIANT selects the MuonEq / Muon^2 combination (all other Muon additions stay on):
#   eq     MuonEq on,  Muon^2 off   (= standard nanochat)
#   eq_m2  MuonEq on,  Muon^2 on
#   m2     MuonEq off, Muon^2 on
#   base   MuonEq off, Muon^2 off
#   all    all four, one after the other (default)
# Usage:
#   sbatch smoke_test.sbatch                                  # all four variants
#   sbatch --export=ALL,VARIANT=eq_m2 smoke_test.sbatch       # a single variant
# Optional overrides (same --export mechanism): NUM_ITERS (default 50), MUON2_BETA2, MUON2_EPS

set -euo pipefail

VARIANT="${VARIANT:-all}"
NUM_ITERS="${NUM_ITERS:-50}"
MUON2_BETA2="${MUON2_BETA2:-0.95}"
MUON2_EPS="${MUON2_EPS:-1e-8}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${VARIANT}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

# Use the uv-built venv in the repo (has torch + wandb); conda is only needed to provide the `uv` command itself
source "$REPO_ROOT/.venv/bin/activate"

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1

# Tiny model for a smoke test. Verify flag names against scripts/base_train.py.
# Short run, with the expensive periodic evals/sampling disabled (a final val-bpb eval still runs at the last step).
COMMON_ARGS=(
    --depth=4
    --device-batch-size=8
    --run=dummy
    --num-iterations="$NUM_ITERS"
    --eval-every=-1
    --eval-tokens=524288
    --core-metric-every=-1
    --sample-every=-1
)

run_variant() {
    local name="$1"; shift
    echo "=================================================================="
    echo "Smoke test: optimizer variant = $name"
    echo "=================================================================="
    # distinct --model-tag so the variants never overwrite each other's checkpoints
    python -m scripts.base_train "${COMMON_ARGS[@]}" --model-tag="smoke_${name}" "$@"
}

M2_ARGS=(--muon2 --muon2-beta2="$MUON2_BETA2" --muon2-eps="$MUON2_EPS")
run_one() {
    case "$1" in
        eq)    run_variant eq    --muoneq ;;
        eq_m2) run_variant eq_m2 --muoneq "${M2_ARGS[@]}" ;;
        m2)    run_variant m2    --no-muoneq "${M2_ARGS[@]}" ;;
        base)  run_variant base  --no-muoneq ;;
        *) echo "Unknown VARIANT='$1' (expected: eq | eq_m2 | m2 | base | all)" >&2; exit 1 ;;
    esac
}

if [[ "$VARIANT" == "all" ]]; then
    for v in eq eq_m2 m2 base; do run_one "$v"; done
else
    run_one "$VARIANT"
fi