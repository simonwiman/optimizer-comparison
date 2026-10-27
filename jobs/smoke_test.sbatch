#!/bin/bash
#SBATCH --job-name=nanochat-test
#SBATCH --partition=gpu-long
#SBATCH --time=00:30:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

# Usage (OPTIMIZER is one of: muon | muon2 | both; default: both):
#   sbatch smoke_test.sbatch                                   # both variants, one after the other
#   sbatch --export=ALL,OPTIMIZER=muon  smoke_test.sbatch      # current nanochat optimizer only
#   sbatch --export=ALL,OPTIMIZER=muon2 smoke_test.sbatch      # current optimizer + Muon^2 only
# Optional overrides (same --export mechanism): NUM_ITERS (default 50), MUON2_BETA2, MUON2_EPS

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-nanochat}"
OPTIMIZER="${OPTIMIZER:-both}"
NUM_ITERS="${NUM_ITERS:-50}"
MUON2_BETA2="${MUON2_BETA2:-0.95}"
MUON2_EPS="${MUON2_EPS:-1e-8}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${OPTIMIZER}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

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

case "$OPTIMIZER" in
    muon)
        run_variant muon
        ;;
    muon2)
        run_variant muon2 --muon2 --muon2-beta2="$MUON2_BETA2" --muon2-eps="$MUON2_EPS"
        ;;
    both)
        run_variant muon
        run_variant muon2 --muon2 --muon2-beta2="$MUON2_BETA2" --muon2-eps="$MUON2_EPS"
        ;;
    *)
        echo "Unknown OPTIMIZER='$OPTIMIZER' (expected: muon | muon2 | both)" >&2
        exit 1
        ;;
esac
