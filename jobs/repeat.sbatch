#!/bin/bash
#SBATCH --job-name=nanochat-repeat
#SBATCH --partition=gpu-long
#SBATCH --time=01:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

# One fixed configuration, repeated as independent jobs (REP = repetition number) to check how
# reproducible nanochat's wall-clock timing is. Nothing varies between repetitions except REP,
# which only appears in the tag / log names: same seed, same data order, same everything.
# Normally submitted by launch_repeat.sh, which overrides --time / --gpus / --job-name.
#
# Fixed: DEPTH=8, MATRIX_LR=0.025, same GPU type / DEV_BS as the d8 jobs of the old sweep.
# Default config = the old sweep.sbatch config: MuonEq, Muon+ and NorMuon ON (nanochat defaults).
#
# Required (--export=ALL,KEY=value,...):
#   REP        repetition number (1..8); only used to name the run, NOT passed to training
#
# Optional (defaults in brackets):
#   VARIANT (eq)       eq | eq_m2 | m2 | base
#   POSTORTH (on)      on | off   (off = --no-muonplus --no-normuon)
#   DEPTH (8)  MATRIX_LR (0.025)  DEV_BS (32)
#   EVAL_EVERY / EVAL_TOKENS / CORE_EVERY   only passed if set; otherwise nanochat defaults
#   MUON2_BETA2 (0.95)  MUON2_EPS (1e-8)
#
# Isolation from the old sweeps: tags start with "repeat_", logs go to logs/slurm/repeat/,
# .done markers to logs/repeat/, and --model-tag = tag (own checkpoint dir). Nothing under
# logs/slurm/sweep/ or logs/sweep/ is read or written, so the sweep notebooks never see these.
#
# Re-submitting a finished run is a no-op (skipped via logs/repeat/<tag>.done).

set -euo pipefail

REP="${REP:?set REP}"
VARIANT="${VARIANT:-eq}"
POSTORTH="${POSTORTH:-on}"
DEPTH="${DEPTH:-8}"
MATRIX_LR="${MATRIX_LR:-0.025}"
DEV_BS="${DEV_BS:-32}"
MUON2_BETA2="${MUON2_BETA2:-0.95}"
MUON2_EPS="${MUON2_EPS:-1e-8}"

TAG="repeat_d${DEPTH}_${VARIANT}_po${POSTORTH}_lr${MATRIX_LR}_rep${REP}"

# Repo root = git toplevel of the submit dir (so the scripts can live in e.g. jobs/).
# Override with --export=ALL,REPO_ROOT=/path/to/repo if the checkout isn't a git repo.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SUBMIT_DIR")}"
LOG_DIR="$REPO_ROOT/logs/slurm/repeat"
DONE_DIR="$REPO_ROOT/logs/repeat"
mkdir -p "$LOG_DIR" "$DONE_DIR"

if [[ -f "$DONE_DIR/$TAG.done" ]]; then
    echo "[$TAG] already finished, skipping"
    exit 0
fi

exec > "$LOG_DIR/${TAG}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1

ARGS=(
    --depth="$DEPTH"
    --run=dummy
    --model-tag="$TAG"
    --matrix-lr="$MATRIX_LR"
    --device-batch-size="$DEV_BS"
    --sample-every=-1              # text samples are not needed
)

[[ -n "${EVAL_EVERY:-}"  ]] && ARGS+=(--eval-every="$EVAL_EVERY")
[[ -n "${EVAL_TOKENS:-}" ]] && ARGS+=(--eval-tokens="$EVAL_TOKENS")
[[ -n "${CORE_EVERY:-}"  ]] && ARGS+=(--core-metric-every="$CORE_EVERY")

case "$POSTORTH" in
    on)  ;;                                          # Muon+ and NorMuon stay at their defaults (ON)
    off) ARGS+=(--no-muonplus --no-normuon) ;;
    *) echo "Unknown POSTORTH='$POSTORTH' (expected: on | off)" >&2; exit 1 ;;
esac

M2_ARGS=(--muon2 --muon2-beta2="$MUON2_BETA2" --muon2-eps="$MUON2_EPS")
case "$VARIANT" in
    eq)    ARGS+=(--muoneq) ;;
    eq_m2) ARGS+=(--muoneq "${M2_ARGS[@]}") ;;
    m2)    ARGS+=(--no-muoneq "${M2_ARGS[@]}") ;;
    base)  ARGS+=(--no-muoneq) ;;
    *) echo "Unknown VARIANT='$VARIANT' (expected: eq | eq_m2 | m2 | base)" >&2; exit 1 ;;
esac

echo "[$TAG] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | head -1)"   # node/GPU identity, to relate timing differences to hardware
echo "[$TAG] python -m scripts.base_train ${ARGS[*]}"
python -m scripts.base_train "${ARGS[@]}"

# Only reached if training exited cleanly (set -e)
date -Is > "$DONE_DIR/$TAG.done"
