"""Performance monitoring and the automatic performance report.

Definitions used throughout the report:

* Pointing error   - angle between the TRUE beacon position and the camera's
                     actual line of sight (commanded pose + vibration) at
                     capture time (deg / mrad / pixels).
* Locked           - tracker is in TRACKING state AND its position estimate
                     is within ``lock_tolerance_px`` of the true beacon (so
                     following a decoy does not count as lock).
* False lock       - TRACKING, but the estimate is NOT on the beacon.
* Acquisition time - simulated time until the first lock.
* Time locked      - locked frames / ALL frames of the run.
* Lock retention   - locked frames / all frames after the first lock.
  (also reported excluding frames where the beacon was occluded, because
  no tracker can see through a cloud)
* Handover ready   - locked AND pointing error < ``handover_radius_px``,
                     i.e. close enough to boresight for fine pointing.
"""

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import OUTPUT_ROOT
from .tracker import TrackState

CSV_FIELDS = [
    "frame", "t", "state", "locked", "false_lock", "beacon_in_fov", "beacon_occluded",
    "beacon_az", "beacon_el", "cam_az", "cam_el", "los_az", "los_el", "est_az", "est_el",
    "err_deg", "err_px", "est_err_px", "num_detections",
    "proc_ms", "render_ms",
]


class PerformanceMonitor:
    def __init__(self, cfg, camera, sim_fps):
        self.camera = camera
        self.sim_fps = sim_fps
        self.lock_tol = float(cfg.get("lock_tolerance_px", 8.0))
        self.handover_px = float(cfg.get("handover_radius_px", 10.0))
        out = Path(cfg.get("output_dir", "runs"))
        self.output_dir = out if out.is_absolute() else OUTPUT_ROOT / out
        self.rows = []
        self.acq_time = None
        self.prev_locked = False
        self.lock_losses = 0
        self.frames_since_acq = 0      # running counters for the live display
        self.locked_since_acq = 0
        self.wall_start = None
        self.wall_last = None
        self.extra = None              # optional callable -> extra summary fields

    def record(self, frame_idx, t, track, beacon, pose, los, detections,
               proc_ms, render_ms, occluded=False):
        now = time.perf_counter()
        if self.wall_start is None:
            self.wall_start = now
        self.wall_last = now

        cam = self.camera
        in_fov = cam.in_fov(beacon.az, beacon.el, los)
        d_az, d_el = beacon.az - los[0], beacon.el - los[1]
        err_deg = math.hypot(d_az, d_el)
        err_px = math.hypot(d_az * cam.ppd_x, d_el * cam.ppd_y)

        # How far the tracker's belief is from the truth (pixels).
        est_err = float("nan")
        if track.estimate is not None:
            est_err = math.hypot((track.estimate[0] - beacon.az) * cam.ppd_x,
                                 (track.estimate[1] - beacon.el) * cam.ppd_y)
        tracking = track.state == TrackState.TRACKING
        on_beacon = est_err <= self.lock_tol          # False for NaN
        locked = tracking and in_fov and on_beacon
        false_lock = tracking and not on_beacon

        if locked and self.acq_time is None:
            self.acq_time = t
        if self.acq_time is not None:
            self.frames_since_acq += 1
            self.locked_since_acq += int(locked)
        if self.prev_locked and not locked:
            self.lock_losses += 1
        self.prev_locked = locked

        self.rows.append({
            "frame": frame_idx, "t": t, "state": track.state.value,
            "locked": int(locked), "false_lock": int(false_lock),
            "beacon_in_fov": int(in_fov), "beacon_occluded": int(occluded),
            "beacon_az": beacon.az, "beacon_el": beacon.el,
            "cam_az": pose[0], "cam_el": pose[1], "los_az": los[0], "los_el": los[1],
            "est_az": track.estimate[0] if track.estimate is not None else float("nan"),
            "est_el": track.estimate[1] if track.estimate is not None else float("nan"),
            "err_deg": err_deg, "err_px": err_px, "est_err_px": est_err,
            "num_detections": len(detections), "proc_ms": proc_ms, "render_ms": render_ms,
        })
        return self.rows[-1]

    # ------------------------------------------------------------------
    def live(self):
        """Cheap subset of statistics for the on-screen display."""
        if not self.rows:
            return {}
        r = self.rows[-1]
        n = len(self.rows)
        wall = (self.wall_last - self.wall_start) if n > 1 else 0.0
        recent = self.rows[-30:]
        stats = {
            "err_px": r["err_px"],
            "err_mrad": math.radians(r["err_deg"]) * 1e3,
            "fps": (n - 1) / wall if wall > 0 else 0.0,
            "proc_ms": float(np.mean([x["proc_ms"] for x in recent])),
            "acq_time": self.acq_time,
            "lock_losses": self.lock_losses,
            "occluded": bool(r["beacon_occluded"]),
            "retention": None,
        }
        if self.frames_since_acq:
            stats["retention"] = 100.0 * self.locked_since_acq / self.frames_since_acq
        return stats

    def summary(self):
        if not self.rows:
            return {}
        rows = self.rows
        col = lambda k, dtype=float: np.array([x[k] for x in rows], dtype)
        locked, false_lock = col("locked", bool), col("false_lock", bool)
        occluded, in_fov = col("beacon_occluded", bool), col("beacon_in_fov", bool)
        err_deg, err_px = col("err_deg"), col("err_px")
        proc, render, t = col("proc_ms"), col("render_ms"), col("t")
        wall = (self.wall_last - self.wall_start) if len(rows) > 1 else 0.0

        s = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "frames": len(rows),
            "simulation_duration_s": round(float(t[-1] + 1.0 / self.sim_fps), 3),
            "wall_clock_duration_s": round(wall, 3),
            "simulation_fps": self.sim_fps,
            "achieved_fps": round((len(rows) - 1) / wall, 2) if wall > 0 else None,
            "mean_processing_ms": round(float(proc.mean()), 3),
            "max_processing_ms": round(float(proc.max()), 3),
            "processing_throughput_fps": round(1000.0 / max(float(proc.mean()), 1e-6), 1),
            "mean_render_ms": round(float(render.mean()), 3),
            "acquisition_time_s": None if self.acq_time is None else round(self.acq_time, 3),
            "acquired": self.acq_time is not None,
            "lock_losses": self.lock_losses,
            "false_lock_frames": int(false_lock.sum()),
            "beacon_in_fov_pct": round(100.0 * in_fov.mean(), 2),
            "beacon_occluded_pct": round(100.0 * occluded.mean(), 2),
            # Share of the WHOLE run spent locked on the beacon: unlike the
            # retention below it also penalises slow or failed acquisition.
            "time_locked_pct": round(100.0 * locked.mean(), 2),
            "lock_retention_pct": 0.0,
            "lock_retention_unoccluded_pct": 0.0,
        }
        if self.acq_time is not None:
            after = t >= self.acq_time
            s["lock_retention_pct"] = round(100.0 * locked[after].mean(), 2)
            visible = after & ~occluded
            if visible.any():
                s["lock_retention_unoccluded_pct"] = round(100.0 * locked[visible].mean(), 2)

        if locked.any():
            e_deg, e_px = err_deg[locked], err_px[locked]
            s.update({
                "tracking_error_mean_px": round(float(e_px.mean()), 3),
                "tracking_error_rms_px": round(float(np.sqrt((e_px ** 2).mean())), 3),
                "tracking_error_max_px": round(float(e_px.max()), 3),
                "tracking_error_mean_mrad": round(float(np.radians(e_deg.mean()) * 1e3), 4),
                "tracking_error_max_mrad": round(float(np.radians(e_deg.max()) * 1e3), 4),
                "handover_ready_pct_of_locked": round(100.0 * (e_px <= self.handover_px).mean(), 2),
            })
        if self.extra is not None:
            s.update(self.extra())
        return s

    def save(self, run_name=None):
        """Write frames.csv, summary.json/.txt, report.html and report.pdf; return the folder."""
        run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        folder = self.output_dir / run_name
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "frames.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        summary = self.summary()
        with open(folder / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(folder / "summary.txt", "w", encoding="utf-8") as f:
            f.write(format_summary(summary))
        # Formatted reports with 3D views (HTML + PDF). A failure here must
        # never lose the raw log above, so it only produces a warning.
        try:
            from .report import generate
            cam = self.camera
            generate(folder, self.rows, summary, (cam.az_min, cam.az_max, cam.el_min, cam.el_max))
        except Exception as err:            # pragma: no cover - reporting is best-effort
            print(f"WARNING: HTML/PDF report not generated: {err}")
        return folder


def format_summary(s):
    lines = ["FSOC Coarse Tracking - Performance Report", "=" * 44]
    width = max(len(k) for k in s) if s else 0
    for key, value in s.items():
        lines.append(f"{key:<{width}} : {value}")
    return "\n".join(lines) + "\n"
