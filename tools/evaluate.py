"""Multi-seed evaluation: honest performance numbers with spread.

A single run can be lucky or unlucky, so every configuration is run for many
seeds (each seed = different noise, turbulence, and for random scenes a
different beacon path, decoy set, etc.). Results are appended to a CSV after
EVERY run, so an interrupted evaluation resumes where it stopped.

Suites
  random     random scenes (the application's default mode)
  scenarios  the six fixed test scenarios, each with varied seeds

Outputs (runs/evaluation/)
  results.csv     one row per run
  summary.csv     one row per (scenario, method) with statistics
  summary.md      report-ready tables, incl. random-scene breakdowns
  *.png           box plots of time-locked and acquisition time

Examples
    python tools/evaluate.py --suite random --seeds 1-40 --methods hybrid none
    python tools/evaluate.py --suite scenarios --seeds 1-10
    python tools/evaluate.py --report-only
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsoc.config import load_config          # noqa: E402
from fsoc.simulation import Simulation       # noqa: E402

FIELDS = ["suite", "scenario", "method", "seed", "duration_s",
          "acquired", "acquisition_time_s", "time_locked_pct", "lock_retention_pct",
          "lock_retention_unoccluded_pct", "lock_losses", "false_lock_frames",
          "candidates_rejected", "locks_dropped_by_identity", "beacon_occluded_pct",
          "tracking_error_mean_px", "handover_ready_pct_of_locked", "mean_processing_ms",
          "beacon_path", "beacon_speed_dps", "decoys", "blinking_decoys",
          "turbulence_strength", "vibration_strength", "clouds", "initial_pointing_error_deg"]


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_one(suite, scenario, method, seed, duration, model=None, label=None):
    cfg = load_config(None if scenario == "random" else ROOT / "scenarios" / f"{scenario}.yaml")
    cfg["identification"]["method"] = method
    if model:
        cfg["identification"]["model_path"] = str(model)
    cfg["simulation"].update(seed=seed, duration=duration)
    sim = Simulation(cfg)
    while not sim.finished:
        sim.step()
    s = sim.metrics.summary()
    row = {"suite": suite, "scenario": scenario, "method": label or method, "seed": seed, "duration_s": duration}
    for k in FIELDS[5:18]:
        row[k] = s.get(k)
    for k, v in (sim.scene_info or {}).items():
        if k in FIELDS:
            row[k] = v
    return row


# ---------------------------------------------------------------- statistics
def _num(rows, key):
    vals = [r[key] for r in rows if r.get(key) not in (None, "", "None")]
    return np.array([float(v) for v in vals])


def wilson(k, n, z=1.96):
    """95 % confidence interval of a success rate (Wilson score)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(c - h, 0), 100 * min(c + h, 1)


def aggregate(rows):
    out = {"n": len(rows)}
    acq = [str(r["acquired"]) == "True" for r in rows]
    k = sum(acq)
    out["acquired_pct"] = 100 * k / len(rows)
    out["acquired_ci"] = wilson(k, len(rows))
    t = _num(rows, "acquisition_time_s")
    out["acq_median"] = float(np.median(t)) if len(t) else None
    out["acq_iqr"] = (float(np.percentile(t, 25)), float(np.percentile(t, 75))) if len(t) else None
    locked = _num(rows, "time_locked_pct")
    out["locked_mean"] = float(locked.mean())
    out["locked_ci"] = 1.96 * float(locked.std(ddof=1)) / math.sqrt(len(locked)) if len(locked) > 1 else 0.0
    out["locked_median"] = float(np.median(locked))
    out["locked_ge80"] = 100 * float((locked >= 80).mean())
    ret = _num([r for r, a in zip(rows, acq) if a], "lock_retention_pct")
    out["retention_median"] = float(np.median(ret)) if len(ret) else None
    out["false_lock_mean"] = float(_num(rows, "false_lock_frames").mean())
    out["false_lock_runs"] = 100 * float((_num(rows, "false_lock_frames") > 30).mean())
    err = _num(rows, "tracking_error_mean_px")
    out["err_median"] = float(np.median(err)) if len(err) else None
    out["proc_mean"] = float(_num(rows, "mean_processing_ms").mean())
    return out


def fmt(v, spec=".1f", none="-"):
    return none if v is None else format(v, spec)


def build_report(results_path, out):
    rows = list(csv.DictReader(open(results_path, encoding="utf-8")))
    if not rows:
        print("No results yet.")
        return
    groups = {}
    for r in rows:
        groups.setdefault((r["suite"], r["scenario"], r["method"]), []).append(r)

    summary_rows, lines = [], [
        "# Multi-seed evaluation", "",
        f"{len(rows)} runs. Each run = one seed; random scenes also differ in beacon path, decoys, "
        "turbulence, vibration, clouds and initial pointing error.", "",
        "**Time locked** = share of the whole run with a confirmed lock on the true beacon "
        "(penalises slow / failed acquisition). **Retention** = locked share after the first lock "
        "(median over runs that acquired). **False-lock runs** = runs with > 1 s locked on a wrong object.",
        "", "| Suite | Scenario | Method | Runs | Acquired % [95% CI] | Acq. time median [IQR] (s) "
        "| Time locked mean ± 95% CI (%) | Runs ≥ 80% locked | Retention median (%) | False-lock runs (%) "
        "| Error median (px) | Proc. (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (suite, scen, method), g in sorted(groups.items()):
        a = aggregate(g)
        iqr = a["acq_iqr"]
        lines.append(
            f"| {suite} | {scen} | {method} | {a['n']} | {a['acquired_pct']:.0f} "
            f"[{a['acquired_ci'][0]:.0f}-{a['acquired_ci'][1]:.0f}] | "
            f"{fmt(a['acq_median'])} [{fmt(iqr and iqr[0])}-{fmt(iqr and iqr[1])}] | "
            f"{a['locked_mean']:.1f} ± {a['locked_ci']:.1f} | {a['locked_ge80']:.0f} % | "
            f"{fmt(a['retention_median'])} | {a['false_lock_runs']:.0f} | {fmt(a['err_median'], '.2f')} | "
            f"{a['proc_mean']:.1f} |")
        summary_rows.append({"suite": suite, "scenario": scen, "method": method, **{
            k: (v if not isinstance(v, tuple) else f"{v[0]:.2f}-{v[1]:.2f}") for k, v in a.items()}})

    # Random-scene breakdowns: where does it struggle?
    rnd = [r for r in rows if r["suite"] == "random"]
    for method in sorted({r["method"] for r in rnd}):
        rr = [r for r in rnd if r["method"] == method]
        lines += ["", f"## Random scenes, method `{method}`: time locked (%) by scene property", "",
                  "| Property | Bin | Runs | Time locked mean (%) | Acquired % |", "|---|---|---|---|---|"]
        bins = [("Turbulence", "turbulence_strength", [(0.0, 1.0), (1.0, 1.5), (1.5, 9.0)]),
                ("Decoys", "decoys", [(0, 0.5), (0.5, 2.5), (2.5, 9)]),
                ("Beacon speed (°/s)", "beacon_speed_dps", [(0, 2), (2, 3.5), (3.5, 99)])]
        for name, key, edges in bins:
            for lo, hi in edges:
                sub = [r for r in rr if r.get(key) not in (None, "") and lo <= float(r[key]) < hi]
                if sub:
                    a = aggregate(sub)
                    label = f"{lo:g}-{hi:g}" if hi < 9 else f"≥ {lo:g}"
                    lines.append(f"| {name} | {label} | {len(sub)} | {a['locked_mean']:.1f} | {a['acquired_pct']:.0f} |")
        for val in ("True", "False"):
            sub = [r for r in rr if r.get("clouds") == val]
            if sub:
                a = aggregate(sub)
                lines.append(f"| Clouds | {'yes' if val == 'True' else 'no'} | {len(sub)} | "
                             f"{a['locked_mean']:.1f} | {a['acquired_pct']:.0f} |")
        for path in sorted({r["beacon_path"] for r in rr if r.get("beacon_path")}):
            sub = [r for r in rr if r.get("beacon_path") == path]
            a = aggregate(sub)
            lines.append(f"| Beacon path | {path} | {len(sub)} | {a['locked_mean']:.1f} | {a['acquired_pct']:.0f} |")

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    _plots(groups, out)
    print("\n".join(lines))
    print(f"\nSaved {out / 'summary.md'}, summary.csv and plots")


def _plots(groups, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(groups)
    labels = [f"{s}\n{m}" if s != "random" else f"random\n{m}" for _, s, m in keys]
    for metric, title, fname in (("time_locked_pct", "Time locked on the beacon (% of run)", "box_time_locked.png"),
                                 ("acquisition_time_s", "Acquisition time (s), runs that acquired",
                                  "box_acquisition_time.png")):
        data = [_num(groups[k], metric) for k in keys]
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(keys) + 2), 4.2))
        ax.boxplot([d if len(d) else [np.nan] for d in data], showmeans=True)
        ax.set_xticks(range(1, len(keys) + 1), labels, fontsize=8)
        for i, d in enumerate(data, start=1):
            ax.scatter(np.random.normal(i, 0.05, len(d)), d, s=8, alpha=0.45, color="#1f77b4")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=["random", "scenarios"], default="random")
    p.add_argument("--scenarios", nargs="+", default=["01_clear_sky", "02_strong_turbulence", "03_occlusion",
                                                       "04_decoys", "05_fast_target", "06_worst_case"])
    p.add_argument("--seeds", default="1-40")
    p.add_argument("--methods", nargs="+", default=["hybrid"])
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--out", default=str(ROOT / "runs" / "evaluation"))
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--model", help="CNN model to use instead of models/beacon_cnn.onnx (e.g. a candidate)")
    p.add_argument("--label", help="name for this method in the results (e.g. hybrid-v2)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
    if not args.report_only:
        done = set()
        if results.exists():
            done = {(r["suite"], r["scenario"], r["method"], int(r["seed"]))
                    for r in csv.DictReader(open(results, encoding="utf-8"))}
        scenarios = ["random"] if args.suite == "random" else args.scenarios
        name = lambda m: args.label or m
        jobs = [(args.suite, sc, m, sd) for sc in scenarios for m in args.methods for sd in parse_seeds(args.seeds)
                if (args.suite, sc, name(m), sd) not in done]
        print(f"{len(jobs)} runs to do ({len(done)} already in {results})", flush=True)
        t0 = time.time()
        for i, (suite, sc, m, sd) in enumerate(jobs, 1):
            row = run_one(suite, sc, m, sd, args.duration, args.model, args.label)
            new = not results.exists()
            with open(results, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)
            eta = (time.time() - t0) / i * (len(jobs) - i)
            print(f"[{i}/{len(jobs)}] {sc:22s} {row['method']:9s} seed {sd:3d}  acquired={row['acquired']!s:5s} "
                  f"locked={row['time_locked_pct']:5.1f}%  (ETA {eta / 60:.0f} min)", flush=True)
    build_report(results, out)


if __name__ == "__main__":
    main()
