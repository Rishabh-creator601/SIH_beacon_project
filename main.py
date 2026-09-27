"""FSOC Virtual Camera Tracking Simulator - entry point.

Examples
--------
    python main.py                            # live viewer, default scenario
    python main.py --config scenarios/x.yaml  # custom scenario
    python main.py --headless --duration 120  # fast batch run, report only
"""

import argparse
import time

import cv2

from fsoc.config import load_config
from fsoc.metrics import format_summary
from fsoc.simulation import Simulation
from fsoc.visualization import Overlay

WINDOW = "FSOC Coarse Tracking Simulator"


def parse_args():
    p = argparse.ArgumentParser(description="FSOC virtual camera coarse-tracking simulator")
    p.add_argument("--config", help="scenario YAML merged over config/default.yaml")
    p.add_argument("--headless", action="store_true", help="no window, run as fast as possible")
    p.add_argument("--duration", type=float, help="override simulated duration (s)")
    p.add_argument("--seed", type=int, help="override random seed")
    p.add_argument("--name", help="name of the output run folder")
    return p.parse_args()


def run_headless(sim):
    while not sim.finished:
        sim.step()


def run_viewer(sim, cfg):
    overlay = Overlay(sim)
    realtime = cfg["simulation"].get("realtime", True)
    paused = False
    res = sim.step()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    next_tick = time.perf_counter()
    while True:
        next_tick += sim.dt
        if not paused:
            if sim.finished:
                break
            res = sim.step()
        cv2.imshow(WINDOW, overlay.draw(res, paused))

        key = cv2.waitKey(1) & 0xFF
        if realtime:
            # Deadline-based pacing: sleep off whatever is left of this frame.
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.perf_counter()   # running late: don't try to catch up
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("t"):
            overlay.show_truth = not overlay.show_truth
        elif key == ord("r"):
            sim = Simulation(cfg)
            overlay = Overlay(sim)
            res = sim.step()
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return sim


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.duration is not None:
        cfg["simulation"]["duration"] = args.duration
    if args.seed is not None:
        cfg["simulation"]["seed"] = args.seed

    sim = Simulation(cfg)
    if args.headless:
        if sim.duration <= 0:
            raise SystemExit("--headless needs a positive duration")
        run_headless(sim)
    else:
        sim = run_viewer(sim, cfg)

    folder = sim.metrics.save(args.name)
    print(format_summary(sim.metrics.summary()))
    print(f"Performance log written to: {folder}")


if __name__ == "__main__":
    main()
