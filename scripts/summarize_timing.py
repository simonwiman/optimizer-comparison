#!/usr/bin/env python3
"""
Summarize nanochat timing runs (see timing.sbatch) and extrapolate full-run wall-clock time.

Usage: python summarize_timing.py <run_dir> [--eval-every 250] [--core-every 2000] [--warmup-steps 20]

<run_dir> contains, per depth D:  d{D}.log  (stdout of the short run)  and  d{D}.meta  (key=value: wall_s, rc, device_bs, status)

Model of the full run:
    total = fixed_overhead + N * step_time + n_eval * eval_time + n_core * core_time
  step_time      mean dt over steady-state steps (step > warmup-steps; the first steps contain torch.compile)
  N              the num_iterations that the *real* run would use (printed by the timing run, which keeps it unchanged)
  eval/core time min over the measured evals (the first eval at step 0 includes compile / cold-start cost)
  fixed_overhead measured wall-clock of the short run minus (steps * step_time + all measured evals);
                 i.e. process startup, data/tokenizer setup and the torch.compile warmup
Not included: periodic sampling and the final checkpoint save (both small).
"""
import argparse, csv, re, sys
from pathlib import Path

def num(s):
    return float(s.replace(",", ""))

def parse_meta(p):
    d = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    return d

def parse_log(text):
    r = {}
    m = re.search(r"Calculated number of iterations from target data:param ratio: ([\d,]+)", text)
    r["N"] = int(num(m.group(1))) if m else None
    m = re.search(r"Total batch size ([\d,]+) => gradient accumulation steps: (\d+)", text)
    r["batch"], r["accum"] = (int(num(m.group(1))), int(m.group(2))) if m else (None, None)
    m = re.search(r"Total number of training tokens: ([\d,]+)", text)
    r["tokens"] = int(num(m.group(1))) if m else None
    m = re.search(r"^total\s*:\s*([\d,]+)\s*$", text, re.M)
    r["params"] = int(num(m.group(1))) if m else None
    m = re.search(r"Peak memory usage: ([\d.]+)MiB", text)
    r["peak_mib"] = float(m.group(1)) if m else None
    steps = []
    for m in re.finditer(r"step (\d+)/(\d+) \(.*?dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+)", text):
        steps.append((int(m.group(1)), float(m.group(3)), num(m.group(4)), float(m.group(5))))
    r["steps"] = steps
    r["evals"] = [float(x) for x in re.findall(r"Validation bpb: [\d.]+ \| eval time: ([\d.]+)s", text)]
    r["cores"] = [float(x) for x in re.findall(r"CORE metric: [-\d.]+ \| eval time: ([\d.]+)s", text)]
    return r

def count_events(N, every):
    """Steps at which the loop evaluates: multiples of `every` in [0, N] plus the final step N."""
    if every <= 0:
        return 0
    return len(set(range(0, N + 1, every)) | {N})

def count_core(N, every):
    if every <= 0:
        return 0
    return len(set(range(every, N + 1, every)) | {N})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--eval-every", type=int, default=250, help="eval cadence of the REAL run (base_train default 250)")
    ap.add_argument("--core-every", type=int, default=2000, help="CORE cadence of the REAL run (base_train default 2000; -1 = off)")
    ap.add_argument("--warmup-steps", type=int, default=20, help="steps ignored when averaging step time")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    logs = sorted(run_dir.glob("d*.log"), key=lambda p: int(re.match(r"d(\d+)", p.name).group(1)))
    if not logs:
        sys.exit(f"no d*.log files in {run_dir}")

    rows, csv_rows = [], []
    for lp in logs:
        depth = int(re.match(r"d(\d+)", lp.name).group(1))
        meta = parse_meta(lp.with_suffix(".meta"))
        status = meta.get("status", "?")
        dbs = meta.get("device_bs", "?")
        if status != "ok":
            rows.append([str(depth), dbs, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", status.upper()])
            continue
        r = parse_log(lp.read_text(errors="replace"))
        steady = [s for s in r["steps"] if s[0] > args.warmup_steps]
        if not steady or r["N"] is None:
            rows.append([str(depth), dbs, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "PARSE-ERROR"])
            continue
        dt = sum(s[1] for s in steady) / len(steady) / 1000.0          # seconds, mean
        toks = sum(s[2] for s in steady) / len(steady)
        mfu = sum(s[3] for s in steady) / len(steady)
        N = r["N"]
        eval_t = min(r["evals"]) if r["evals"] else 0.0
        core_t = min(r["cores"]) if r["cores"] else 0.0
        n_eval = count_events(N, args.eval_every) if r["evals"] else 0
        n_core = count_core(N, args.core_every) if r["cores"] else 0
        wall = float(meta.get("wall_s", "nan"))
        run_steps = len(r["steps"])
        overhead = wall - run_steps * dt - sum(r["evals"]) - sum(r["cores"])
        overhead = max(overhead, 0.0)
        train_s = N * dt
        total_s = overhead + train_s + n_eval * eval_t + n_core * core_t
        rows.append([
            str(depth), dbs, f"{(r['params'] or 0)/1e6:.0f}M",
            f"{(r['batch'] or 0)/1e3:.0f}K", f"{N:,}", f"{(r['tokens'] or 0)/1e9:.2f}B",
            f"{dt*1000:.0f}", f"{toks/1e3:.0f}K", f"{mfu:.1f}",
            f"{(r['peak_mib'] or 0)/1024:.1f}",
            f"{train_s/3600:.2f}", f"{(n_eval*eval_t + n_core*core_t)/3600:.2f}", f"{total_s/3600:.2f}",
        ])
        csv_rows.append(dict(depth=depth, device_bs=dbs, params=r["params"], batch_tokens=r["batch"], accum=r["accum"],
                             num_iterations=N, tokens=r["tokens"], step_ms=dt*1000, tok_per_s=toks, mfu=mfu,
                             peak_mem_gib=(r["peak_mib"] or 0)/1024, train_h=train_s/3600,
                             eval_h=(n_eval*eval_t)/3600, core_h=(n_core*core_t)/3600,
                             overhead_min=overhead/60, total_h=total_s/3600,
                             n_timed_steps=len(steady), eval_s=eval_t, core_s=core_t))

    header = ["depth", "dev_bs", "params", "batch", "iters", "tokens", "step_ms", "tok/s", "MFU%", "mem_GiB",
              "train_h", "evals_h", "TOTAL_h"]
    # rows for failures have an extra column; normalize widths
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    def fmt(r):
        out = "  ".join(r[i].rjust(widths[i]) for i in range(len(header)))
        return out + (f"  <- {r[-1]}" if len(r) > len(header) and r[-1] not in ("", None) else "")
    print("\nEstimated FULL-run wall-clock on ONE GPU of the type used for this timing job")
    print("(train_h = N * step time; evals_h = periodic val-bpb + CORE; TOTAL_h also includes startup/compile overhead)\n")
    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))
    print("\nMultiply TOTAL_h by the number of runs (variants x seeds x sweeps) for the total GPU-hours of your study.")

    out_csv = run_dir / "timing_summary.csv"
    if csv_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader(); w.writerows(csv_rows)
        print(f"CSV written to {out_csv}")

if __name__ == "__main__":
    main()
