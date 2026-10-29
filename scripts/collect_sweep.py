#!/usr/bin/env python3
"""Collect sweep results from logs/slurm/sweep/*.out into logs/sweep/results.csv.

Usage (from repo root):  python collect_sweep.py
"""
import csv
import glob
import os
import re

FILE_RE = re.compile(r"sweep_d(\d+)_(eq_m2|eq|m2|base)_lr([0-9.eE+-]+?)-\d+\.out$")
VAL_RE = re.compile(r"Validation bpb: ([0-9.]+)")
MIN_RE = re.compile(r"Minimum validation bpb: ([0-9.]+)")
TIME_RE = re.compile(r"Total training time: ([0-9.]+)m")
CORE_RE = re.compile(r"CORE metric: ([0-9.\-]+)")
ITERS_RE = re.compile(r"number of iterations from target data:param ratio: ([0-9,]+)")
NAN_RE = re.compile(r"\bnan\b", re.I)

rows = []
for path in sorted(glob.glob("logs/slurm/sweep/*.out")):
    m = FILE_RE.search(os.path.basename(path))
    if not m:
        continue
    depth, variant, lr = int(m.group(1)), m.group(2), m.group(3)
    text = open(path, errors="replace").read()
    vals, mins = VAL_RE.findall(text), MIN_RE.findall(text)
    times, cores, iters = TIME_RE.findall(text), CORE_RE.findall(text), ITERS_RE.findall(text)
    finished = bool(mins)
    rows.append(dict(
        depth=depth, variant=variant, matrix_lr=lr,
        status="done" if finished else ("diverged?" if NAN_RE.search(text) else "incomplete"),
        iters=iters[0].replace(",", "") if iters else "",
        final_val_bpb=vals[-1] if vals else "",
        min_val_bpb=mins[-1] if mins else "",
        final_core=cores[-1] if cores else "",
        train_min=times[-1] if times else "",
        log=path,
    ))

rows.sort(key=lambda r: (r["depth"], r["variant"], float(r["matrix_lr"])))
os.makedirs("logs/sweep", exist_ok=True)
out = "logs/sweep/results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["depth"])
    w.writeheader()
    w.writerows(rows)

print(f"{len(rows)} logs -> {out}")
for d in sorted({r["depth"] for r in rows}):
    print(f"\ndepth {d}: final val bpb (rows = variant, cols = matrix_lr)")
    lrs = sorted({float(r["matrix_lr"]) for r in rows if r["depth"] == d})
    print("variant  " + "".join(f"{lr:>10g}" for lr in lrs))
    for v in ("eq", "eq_m2", "m2", "base"):
        cells = []
        for lr in lrs:
            r = next((x for x in rows if x["depth"] == d and x["variant"] == v
                      and float(x["matrix_lr"]) == lr), None)
            cells.append(f"{(r['final_val_bpb'] or r['status']) if r else '-':>10}")
        print(f"{v:<9}" + "".join(cells))
