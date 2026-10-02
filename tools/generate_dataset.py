"""Generate a labelled training set for the beacon-identification CNN.

The simulator knows the true position of every light source, so labels are
free. Each episode randomises the scene (beacon blink parameters, 2-6
decoys with various blink behaviours, trajectories, disturbance strengths)
and moves the camera around the beacon, running the SAME rendering,
detection and candidate-tracking code as the application. Every few frames
each candidate tracklet with a full history becomes one sample:

    X     (T, 24, 24) uint8   the candidate's last T image patches
    y     1 = beacon, 0 = anything else
    kind  0 beacon, 1 steady decoy, 2 wrong-rate blinking decoy,
          3 near-rate blinking decoy, 4 glint, 5 other (star / hot pixel / noise),
          6 harmonic decoy (blinks at f0/2, f0/3 or f0/4)

v2 (default output data/beacon_dataset_v2.npz) adds harmonic decoys,
turbulence up to 3.0, the current beacon waveform (off level 15-45 %) and
the application's region-of-interest detection around the followed source.

Episodes are split into train / validation BY EPISODE, so validation
scenes are never seen during training.

    python tools/generate_dataset.py                 # default: 48 episodes
    python tools/generate_dataset.py --episodes 12   # quick test
"""

import argparse
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsoc.camera import VirtualCamera                      # noqa: E402
from fsoc.config import deep_merge, load_config             # noqa: E402
from fsoc.detector import BlobDetector                      # noqa: E402
from fsoc.disturbances import DisturbanceModel, GaussMarkov  # noqa: E402
from fsoc.identification import CandidateTracker            # noqa: E402
from fsoc.scene import Scene, render_frame, to_uint8                   # noqa: E402

KIND_NAMES = ["beacon", "steady_decoy", "wrong_rate_decoy", "near_rate_decoy", "glint", "other",
              "harmonic_decoy"]
MATCH_DEG = 0.12        # a tracklet "is" a source if it stays this close to it
SAMPLE_EVERY = 4        # frames between samples of the same tracklet


def random_trajectory(rng, start):
    kind = rng.choice(["lissajous", "random_walk", "linear", "circular"], p=[0.3, 0.35, 0.2, 0.15])
    if kind == "lissajous":
        return {"type": "lissajous", "center": list(start),
                "amplitude": [float(rng.uniform(1, 8)), float(rng.uniform(1, 6))],
                "frequency": [float(rng.uniform(0.02, 0.1)), float(rng.uniform(0.02, 0.1))],
                "phase": [float(rng.uniform(0, 6.28)), float(rng.uniform(0, 6.28))]}
    if kind == "random_walk":
        return {"type": "random_walk", "start": list(start), "speed": float(rng.uniform(0.3, 7)),
                "turn_rate_std": float(rng.uniform(0.2, 1.0))}
    if kind == "linear":
        return {"type": "linear", "start": list(start),
                "velocity": [float(rng.uniform(-4, 4)), float(rng.uniform(-3, 3))]}
    return {"type": "circular", "center": list(start), "radius": float(rng.uniform(1, 6)),
            "period": float(rng.uniform(15, 60)), "phase": float(rng.uniform(0, 6.28))}


def random_blink(rng, low, high):
    return {"frequency_hz": float(rng.uniform(low, high)), "duty": float(rng.uniform(0.3, 0.8)),
            "off_level": float(rng.uniform(0.0, 0.6)), "phase": float(rng.uniform(0, 1))}


def random_config(base, rng, beacon_hz):
    """One randomised episode configuration."""
    start = np.array([rng.uniform(-15, 15), rng.uniform(-10, 10)])
    # v2: the beacon as currently designed (dims to ~30 % when "off"),
    # with some spread so the model does not over-fit one exact waveform.
    targets = [{
        "name": "beacon", "is_beacon": True,
        "intensity": float(rng.uniform(110, 250)), "sigma_px": float(rng.uniform(1.5, 2.5)),
        "blink": {"frequency_hz": beacon_hz, "duty": float(rng.uniform(0.6, 0.8)),
                  "off_level": float(rng.uniform(0.15, 0.45)), "phase": float(rng.uniform(0, 1))},
        "trajectory": random_trajectory(rng, start),
    }]
    decoy_kinds = []
    for i in range(int(rng.integers(2, 7))):
        r = rng.random()
        t = {"name": f"decoy_{i}", "intensity": float(rng.uniform(110, 250)),
             "sigma_px": float(rng.uniform(1.4, 2.6)),
             "trajectory": random_trajectory(rng, start + rng.uniform(-6, 6, 2))}
        if r < 0.30:
            kind = 1                                            # steady light
        elif r < 0.65:
            kind = 2                                            # clearly different rate
            lo, hi = (0.4, 2.5) if rng.random() < 0.5 else (5.5, 12.0)
            t["blink"] = random_blink(rng, lo, hi)
        elif r < 0.80:
            kind = 3                                            # close to the beacon rate
            lo, hi = (2.5, 3.0) if rng.random() < 0.5 else (5.0, 5.5)
            t["blink"] = random_blink(rng, lo, hi)
        else:
            # Harmonic decoy: blinks at f0/2, f0/3 or f0/4, so its square-wave
            # harmonic lands ON the beacon frequency (fooled the v1 model).
            kind = 6
            f = beacon_hz / int(rng.choice([2, 3, 4]))
            t["blink"] = random_blink(rng, f - 0.1, f + 0.1)
        targets.append(t)
        decoy_kinds.append(kind)

    u = rng.uniform
    dist = {
        "enabled": True,
        "vibration": {"enabled": True, "strength": float(u(0, 2.5))},
        "turbulence": {"enabled": True, "strength": float(u(0, 3.0))},   # v2: up to 3
        "sensor": {"enabled": True, "strength": float(u(0.5, 2.0))},
        "clouds": {"enabled": bool(rng.random() < 0.5), "count": int(rng.integers(2, 9))},
        "glints": {"enabled": True, "rate_hz": float(u(0.2, 3.0))},
        "glare": {"enabled": bool(rng.random() < 0.2),
                  "position_deg": [float(start[0] + u(-8, 8)), float(start[1] + u(-8, 8))]},
        "dropouts": {"enabled": bool(rng.random() < 0.3), "start_after_s": 1.0},
    }
    cfg = deep_merge(base, {"targets": targets, "disturbances": dist})
    return cfg, decoy_kinds


def run_episode(args):
    ep, frames, seed, beacon_hz = args
    rng = np.random.default_rng(seed)
    base = load_config()
    cfg, decoy_kinds = random_config(base, rng, beacon_hz)
    fps = float(cfg["simulation"]["fps"])
    dt = 1.0 / fps
    T = int(cfg["identification"]["window_frames"])

    scene = Scene(cfg, rng)
    cam_cfg = dict(cfg["camera"])
    cam_cfg["initial_pointing"] = [scene.beacon.az, scene.beacon.el]
    camera = VirtualCamera(cam_cfg, scene.bounds)
    disturb = DisturbanceModel(cfg, scene, camera, rng)
    detector = BlobDetector(cfg["detector"])
    cands = CandidateTracker(T, cfg["identification"].get("association_gate_deg", 0.25))

    # The camera follows a "focus point" wandering around the beacon or a
    # decoy; its wander range and speed vary per episode so that slow
    # tracking as well as fast slewing (like during search) are represented.
    offset = GaussMarkov(rng.uniform(0.3, 2.0), rng.uniform(0.5, 4.0), rng, 2)
    decoys = [t for t in scene.targets if not t.is_beacon]
    kinds = {id(t): k for t, k in zip(decoys, decoy_kinds)}
    near = {}                         # tid -> deque of per-frame source labels
    X, y, kind_out = [], [], []

    focus, switch_at = scene.beacon, rng.exponential(4.0)
    t = 0.0
    for _ in range(frames):
        scene.update(t, dt)
        disturb.update(t, dt, camera)
        pose = camera.pose
        los = disturb.line_of_sight(pose)
        frame = to_uint8(render_frame(scene, camera, los, disturb))
        fx_px, fy_px = camera.world_to_pixel(focus.az, focus.el, pose)
        roi = (float(fx_px), float(fy_px)) if camera.in_fov(focus.az, focus.el, pose) else None
        dets = detector.detect(frame, roi)
        for d in dets:
            d.az, d.el = camera.pixel_to_world(d.x, d.y, pose)
        cands.update(dets, frame, camera, pose, dt)

        # Ground truth: which true source is each tracklet sitting on now?
        sources = [(t_.az, t_.el, 0 if t_.is_beacon else kinds[id(t_)]) for t_ in scene.targets]
        sources += [(g[0], g[1], 4) for g in (disturb.glints.sources() if disturb.glints else [])]
        for tr in cands.tracklets.values():
            label = 5
            best = MATCH_DEG
            for saz, sel, k in sources:
                dist = math.hypot(tr.az - saz, tr.el - sel)
                if dist < best:
                    best, label = dist, k
            hist = near.setdefault(tr.id, [])
            hist.append(label)
            if tr.ready and tr.age % SAMPLE_EVERY == 0:
                window = hist[-T:]
                top = max(set(window), key=window.count)
                if window.count(top) >= 0.8 * len(window):     # skip ambiguous tracklets
                    X.append(cands.stack(tr))
                    y.append(1 if top == 0 else 0)
                    kind_out.append(top)

        # Camera: P-control towards the focus point + feed-forward of the
        # focused source. Focus switches between the beacon (half the time)
        # and random decoys, so decoys are well represented too.
        if t >= switch_at:
            focus = scene.beacon if (rng.random() < 0.5 or not decoys) else decoys[rng.integers(len(decoys))]
            switch_at = t + rng.exponential(4.0)
        fx = focus.az + offset.step(dt)[0]
        fy = focus.el + offset.x[1]
        camera.command_rate(4.0 * (fx - camera.az) + focus.vaz,
                            4.0 * (fy - camera.el) + focus.vel, dt)
        t += dt

    if not X:
        return ep, None
    return ep, (np.stack(X), np.array(y, np.uint8), np.array(kind_out, np.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--frames", type=int, default=450, help="frames per episode (15 s at 30 fps)")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=str(ROOT / "data" / "beacon_dataset_v2.npz"))
    args = p.parse_args()

    beacon_hz = float(load_config()["identification"]["beacon_blink_hz"])
    jobs = [(ep, args.frames, args.seed + ep, beacon_hz) for ep in range(args.episodes)]
    results = {}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for ep, res in pool.imap_unordered(run_episode, jobs):
            results[ep] = res
            n = 0 if res is None else len(res[1])
            pos = 0 if res is None else int(res[1].sum())
            print(f"episode {ep:3d}: {n:5d} samples ({pos} beacon)   "
                  f"[{len(results)}/{args.episodes}, {time.time() - t0:.0f} s]", flush=True)

    n_val = max(1, int(round(args.episodes * args.val_fraction)))
    Xs, ys, ks, splits = [], [], [], []
    for ep in range(args.episodes):
        res = results.get(ep)
        if res is None:
            continue
        Xs.append(res[0]); ys.append(res[1]); ks.append(res[2])
        splits.append(np.full(len(res[1]), 1 if ep >= args.episodes - n_val else 0, np.uint8))
    X, y, kind, split = map(np.concatenate, (Xs, ys, ks, splits))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, kind=kind, split=split, beacon_hz=beacon_hz)
    print(f"\nSaved {len(y)} samples to {out}")
    for s, name in ((0, "train"), (1, "validation")):
        m = split == s
        counts = ", ".join(f"{KIND_NAMES[k]}={int((kind[m] == k).sum())}" for k in range(len(KIND_NAMES)))
        print(f"  {name:10s}: {int(m.sum()):6d}  ({counts})")


if __name__ == "__main__":
    main()
