# Combining Muon variants

Code for the report *Combining MUON variants: An investigation of how different MUON variants affect optimizer performance*.

We pre-train GPT models (built on a modified version of [NanoChat](https://github.com/karpathy/nanochat)) at three depths (8, 12, 16; roughly 125M, 286M and 536M parameters) with four optimizer variants and sweep the matrix learning rate for each:

- **Base**: Muon with Polar Express and Nesterov-style momentum
- **+MuonEq**: Base with MuonEq (row equilibration)
- **+Muon²**: Base with Muon² (second-moment scaling)
- **+MuonEq+Muon²**: Base with both

The repository reproduces the three sets of experiments in the report:

| Experiment | Description | Launch script | Analysis notebook |
|---|---|---|---|
| Set 1 | Variants with the post-orthogonalization extensions (Muon+ and NorMuon) ON, depths 8, 12, 16 | `launch_sweep.sh` | `full_optimizer_comparison_postorth.ipynb` |
| Set 2 | Variants with the post-orthogonalization extensions OFF (plain Muon base), depths 8 and 12 | `launch_sweep_reduced_variants.sh` | `full_optimizer_comparison_plain.ipynb` |
| Comparison | Post-orthogonalization ON vs OFF, between sets 1 and 2 (depths 8 and 12) | (uses the logs of sets 1 and 2) | `compare_postorth_vs_plain.ipynb` |
| Time error analysis | Eight identical runs to test run-to-run timing variability | `launch_repeat.sh` | `timing_summary_repeat.ipynb` |

The workflow has two parts:

1. **Training** (requires GPUs and a Slurm cluster, roughly 5 days in total). Produces raw output under `logs/`.
2. **Analysis** (no GPU required, quick to run). Reads `logs/` and writes all figures and tables of the report to `results/`.

If you only want to check the numbers and figures in the report, you can skip straight to [Part 2](#part-2-analysis-no-gpu-needed), provided `logs/` is present.

---

## Changes compared to the default NanoChat repo

This repository is a modified version of [NanoChat](https://github.com/karpathy/nanochat). The files that were mainly changed are:

- `nanochat/optim.py`
- `scripts/base_train.py`

The changes were made so that the optimizer can be configured to run with any of the extensions (MuonEq, Muon², Muon+, NorMuon) turned on or off.

---

## Part 1: Training (GPU required, ~5 days)

### Environment setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. Pick the extra that matches your hardware:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

Training the models in the report requires the GPU setup. The CPU/MPS option is only suitable for checking that the code runs.

### Data and tokenizer

The dataset is NVIDIA ClimbMix, and the tokenizer is trained with NanoChat's tokenizer training script. Run the following once, from the repository root with the environment activated, before launching any experiment:

```bash
python -m nanochat.dataset -n 170
python -m scripts.tok_train
```

The first command downloads the dataset shards (170 in this setup) and the second trains the tokenizer. All experiments use the same tokenizer.

### Launching the experiments

The launch scripts submit jobs to a cluster that uses **Slurm**. You may need to adapt partition names, GPU types and other cluster-specific settings in the scripts to match your own cluster.

**Experiment set 1** (post-orthogonalization extensions ON; depths 8, 12, 16):

```bash
bash launch_sweep.sh
```

**Experiment set 2** (post-orthogonalization extensions OFF; depths 8 and 12):

```bash
bash launch_sweep_reduced_variants.sh
```

**Time error analysis** (eight identical runs):

```bash
bash launch_repeat.sh
```

### Setup used in the report

- Depths 8 and 12 were trained on a single H100 (47 GB); depth 16 on a single H100 (96 GB), on the NUS School of Computing cluster. There was no guarantee that the same physical GPU was used across runs.
- Only the matrix learning rate was swept. Set 1 sweeps 0.005, 0.01, 0.02, 0.04 and 0.08; set 2 sweeps 0.01, 0.02 and 0.04. All other hyperparameters were held fixed (Table 2 of the report: device batch size 32, embedding LR 0.3, unembedding LR 0.008, scalar LR 0.5, weight decay 0.28, warmup 40 steps, warmdown ratio 0.65, final LR fraction 0.05, Muon² β₂ 0.95 and ε 1e-8).
- Training horizon follows NanoChat's default ratio of 12 tokens per parameter. Batch size, total tokens and iterations per depth are in Table 1 of the report (depth 8: 1,920 iterations; depth 12: 2,520; depth 16: 5,376).
- Five Polar Express iterations are used for every variant.
- The time error analysis uses eight identical runs at depth 8 with MuonEq, post-orthogonalization enabled, and matrix learning rate 0.025.

### Output

All jobs write their output under **`logs/`**. These logs are what the notebooks in Part 2 read, so keep the directory structure intact.

> **Note on timing:** wall-clock times vary between physical GPUs, even on the same node (Section 2.4 of the report). Timing results (ms/step, time-to-target) are therefore unreliable and are not used for conclusions. Final validation bits-per-byte (bpb) is unaffected and is the metric the report relies on.

---

## Part 2: Analysis (no GPU needed)

The analysis notebooks read the logs produced in Part 1 and save every figure and table used in the report to **`results/`** (figures in `results/figures/`, tables as CSV in `results/tables/`).

### Required libraries

The notebooks require:

- `numpy`
- `pandas`
- `matplotlib`

You also need Jupyter (or another environment that can open `.ipynb` files):

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook
```

### Notebooks

Run each notebook top to bottom (**Kernel → Restart & Run All**).

| Notebook | Report content |
|---|---|
| `full_optimizer_comparison_postorth.ipynb` (set 1) | Section 2.1: Tables 3 and 8, Figures 1–4 and 11 (and the time-to-target data behind Table 4) |
| `full_optimizer_comparison_plain.ipynb` (set 2) | Section 2.2: Tables 5, 6 and 9, Figures 5–7 and 12 |
| `compare_postorth_vs_plain.ipynb` (sets 1 vs 2) | Section 2.3: Figures 8 and 9, and the ON vs OFF comparison summarized in Table 7 |
| `timing_summary_repeat.ipynb` (time error analysis) | Section 2.4: Figure 10 |

`compare_postorth_vs_plain.ipynb` needs the logs of both experiment set 1 and set 2.

### Files written to `results/`

Figures (`results/figures/`):

| File | Report figure |
|---|---|
| `sweep_postorth_d8_loss_val_vs_wallclock.png` | Figure 1 |
| `sweep_postorth_d12_loss_val_vs_wallclock.png` | Figure 2 |
| `sweep_postorth_d16_loss_val_vs_wallclock.png` | Figure 3 |
| `sweep_postorth_val_bpb_vs_lr.png` | Figure 4 |
| `plain_d8_loss_val_vs_wallclock.png` | Figure 5 |
| `plain_d12_loss_val_vs_wallclock.png` | Figure 6 |
| `plain_val_bpb_vs_lr.png` | Figure 7 |
| `compare_postorth_d8_val_bpb_vs_wallclock.png` | Figure 8 |
| `compare_postorth_d12_val_bpb_vs_wallclock.png` | Figure 9 |
| `timing_repeat_d8_eq_poon_lr0.025_per_run.png` | Figure 10 |
| `sweep_postorth_val_bpb_vs_steps.png` | Figure 11 (appendix) |
| `plain_val_bpb_vs_steps.png` | Figure 12 (appendix) |

Tables (`results/tables/`):

| File | Report table |
|---|---|
| `sweep_postorth_selected_runs.csv` | Table 3 |
| `plain_selected_runs.csv` | Table 5 |
| `plain_time_to_target.csv` | Table 6 |
| `sweep_postorth_val_bpb_vs_lr.csv` | Table 8 (appendix) |
| `plain_val_bpb_vs_lr.csv` | Table 9 (appendix) |

The LaTeX source of the report reads these files directly. Tables 1, 2, 4 and 7 are typed into the LaTeX source; their values come from the training configuration and from the tables above (Table 7 combines the final validation bpb of Tables 3 and 5, and Table 9 for the Base LR 0.04 entry at depth 12).

### Interpreting the results

- **Val bpb**: final validation bits-per-byte. Lower is better. This is the main metric in the report.
- **Loss**: final (EMA-smoothed) training loss.
- **Min / ms/step**: total training time in minutes and median time per step. Treat these as unreliable, since they depend on which physical GPU ran the job.
- **Target / time-to-target**: in set 1 the target is the highest final validation bpb of any variant at that depth; in set 2 it is the final validation bpb of Base. The tables give the steps and wall-clock time each variant needs to reach it, relative to Base. Positive Δ means faster. Time columns are affected by the GPU issue above.
- **Learning rate sweeps**: each variant uses the learning rate with the lowest final validation bpb. Check whether the best value sits at the edge of the sweep, since a better value might then lie outside it. This applies to Base at depth 12 in set 2 (best LR 0.04, marked † in Table 7).
- **ON − OFF (Table 7, Figures 8–9)**: a negative Δ means the post-orthogonalization extensions (Muon+ and NorMuon) improve validation bpb. Only depths 8 and 12 are compared, since depth 16 was not run with the extensions OFF.
- **Time error analysis (Figure 10)**: the eight identical runs differ in final validation bpb by only about 2 × 10⁻⁴, while total training time ranges from about 14.3 to 16.5 minutes depending on the physical GPU. Differences in final validation bpb larger than this are meaningful; timing differences are not.

---

## References

Muon (Jordan et al., 2024), Polar Express (Amsel et al., 2026), Muon+ (Zhang et al., 2026), NorMuon (Li et al., 2025), MuonEq (Chang et al., 2026), Muon² (Liu et al., 2026), NanoChat (Karpathy, 2025), ClimbMix (Diao et al., 2025). Full references are in the report.

Author: Simon Axel Wiman (National University of Singapore)
