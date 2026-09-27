"""Compare identification methods across scenarios (closed-loop, headless).

For every (scenario, method) pair a full simulation is run and its
performance summary collected. The result is printed as a table and saved
to runs/comparison/comparison.csv and comparison.md (ready for the report).

    python tools/compare_methods.py
    python tools/compare_methods.py --scenarios scenarios/04_decoys.yaml --methods none hybrid
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsoc.config import load_config          # noqa: E402
from fsoc.simulation import Simulation       # noqa: E402

COLUMNS = [
    ("acquisition_time_s", "Acq. time (s)"),
    ("lock_retention_pct", "Lock retention (%)"),
    ("lock_retention_unoccluded_pct", "Retention, visible (%)"),
    ("false_lock_frames", "False-lock frames"),
    ("lock_losses", "Lock losses"),
    ("candidates_rejected", "Candidates rejected"),
    ("tracking_error_mean_px", "Mean error (px)"),
    ("mean_processing_ms", "Processing (ms)"),
]


def run(scenario, method, duration, seed):
    cfg = load_config(scenario)
    cfg["identification"]["method"] = method
    cfg["simulation"]["duration"] = duration
    if seed is not None:
        cfg["simulation"]["seed"] = seed
    sim = Simulation(cfg)
    while not sim.finished:
        sim.step()
    return sim.metrics.summary()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+",
                   default=["default", "scenarios/04_decoys.yaml", "scenarios/06_worst_case.yaml"])
    p.add_argument("--methods", nargs="+", default=["none", "blink", "cnn", "hybrid"])
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "runs" / "comparison"))
    args = p.parse_args()

    rows = []
    for scenario in args.scenarios:
        path = None if scenario == "default" else scenario
        name = "default" if path is None else Path(scenario).stem
        for method in args.methods:
            t0 = time.time()
            s = run(path, method, args.duration, args.seed)
            row = {"scenario": name, "method": method}
            row.update({k: s.get(k) for k, _ in COLUMNS})
            rows.append(row)
            print(f"{name:22s} {method:7s} " +
                  "  ".join(f"{label.split(' (')[0]}={row[k]}" for k, label in COLUMNS[:5]) +
                  f"   [{time.time() - t0:.0f} s]", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "method"] + [k for k, _ in COLUMNS]
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    lines = ["| Scenario | Method | " + " | ".join(label for _, label in COLUMNS) + " |",
             "|" + "---|" * (len(COLUMNS) + 2)]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['method']} | " +
                     " | ".join(fmt(r[k]) for k, _ in COLUMNS) + " |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
