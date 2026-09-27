"""Simulation core: wires every module into one closed loop.

One call to ``Simulation.step()`` = one camera frame:

    scene.update ─► render view ─► disturbances ─► detect ─► identify ─► track ─► control ─► move camera
                                                                                  └──────► metrics

The class has no GUI code, so the same loop drives the OpenCV viewer,
headless batch runs and (later) the Qt application.
"""

import time
from dataclasses import dataclass

import numpy as np

from .camera import VirtualCamera
from .controller import PointingController
from .detector import BlobDetector
from .disturbances import DisturbanceModel
from .identification import BeaconIdentifier
from .metrics import PerformanceMonitor
from .randomizer import randomize_config
from .scene import Scene, to_uint8
from .tracker import BeaconTracker, TrackOutput


@dataclass
class FrameResult:
    frame_idx: int
    t: float
    frame: np.ndarray        # 8-bit grayscale camera image
    pose: tuple              # commanded camera (az, el) when the frame was captured
    los: tuple               # actual line of sight (pose + vibration)
    detections: list
    track: TrackOutput
    stats: dict              # the metrics row for this frame


class Simulation:
    def __init__(self, cfg):
        sim = cfg["simulation"]
        # A null seed means "new random run": draw one and remember it, so the
        # exact run can be replayed later with --seed.
        seed = sim.get("seed")
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % 2 ** 31)
        cfg, self.scene_info = randomize_config(cfg, np.random.default_rng([self.seed, 1]))
        self.cfg = cfg
        self.fps = float(sim.get("fps", 30))
        self.dt = 1.0 / self.fps
        self.duration = float(sim.get("duration", 0))
        self.rng = np.random.default_rng(self.seed)

        self.scene = Scene(cfg, self.rng)
        self.camera = VirtualCamera(cfg["camera"], self.scene.bounds)
        self.disturb = DisturbanceModel(cfg, self.scene, self.camera, self.rng)
        self.detector = BlobDetector(cfg["detector"])
        id_cfg = dict(cfg.get("identification") or {"method": "none"})
        try:
            self.identifier = BeaconIdentifier(id_cfg, self.fps)
        except FileNotFoundError as err:
            # No trained model yet: fall back to the classical blink detector.
            print(f"WARNING: {err}\nFalling back to identification method 'blink'.")
            id_cfg["method"] = "blink"
            self.identifier = BeaconIdentifier(id_cfg, self.fps)
        self.id_method = id_cfg["method"]
        self.tracker = BeaconTracker(cfg["tracker"], id_cfg)
        self.controller = PointingController(cfg["controller"], self.camera)
        self.metrics = PerformanceMonitor(cfg["metrics"], self.camera, self.fps)
        self.metrics.extra = lambda: {
            "seed": self.seed,
            "random_scene": self.scene_info or "off (fixed scenario)",
            "identification_method": self.id_method,
            "candidates_rejected": self.tracker.rejections,
            "locks_dropped_by_identity": self.tracker.dropped_locks,
        }
        self.t = 0.0
        self.frame_idx = 0

    @property
    def finished(self):
        return self.duration > 0 and self.t >= self.duration - 1e-9

    def step(self):
        dt = self.dt
        # 1. World and disturbances evolve.
        self.scene.update(self.t, dt)
        self.disturb.update(self.t, dt, self.camera)

        # 2. Camera captures a frame. Vibration makes the actual line of sight
        #    (los) differ from the commanded pose, and the tracker only knows
        #    the commanded pose - so jitter appears as apparent target motion.
        t0 = time.perf_counter()
        pose = self.camera.pose
        los = self.disturb.line_of_sight(pose)
        img = self.scene.render(self.camera, los, self.disturb)
        frame = to_uint8(self.disturb.process_image(img, self.camera, los))
        t1 = time.perf_counter()

        # 3. Detection -> world coordinates -> identification (P(beacon) per
        #    candidate) -> tracking -> control.
        roi_center = None
        est = self.tracker.predicted_position(dt)
        if est is not None:
            x, y = self.camera.world_to_pixel(est[0], est[1], pose)
            if 0 <= x < self.camera.width and 0 <= y < self.camera.height:
                roi_center = (float(x), float(y))
        detections = self.detector.detect(frame, roi_center)
        for d in detections:
            d.az, d.el = self.camera.pixel_to_world(d.x, d.y, pose)
        self.identifier.process(frame, detections, self.camera, pose, dt)
        track = self.tracker.step(detections, dt)
        rate_az, rate_el = self.controller.compute(track, self.camera, dt)
        t2 = time.perf_counter()

        # 4. Gimbal moves during the next frame interval.
        self.camera.command_rate(rate_az, rate_el, dt)

        # 5. Bookkeeping.
        beacon = self.scene.beacon
        stats = self.metrics.record(self.frame_idx, self.t, track, beacon, pose, los,
                                    detections, (t2 - t1) * 1e3, (t1 - t0) * 1e3,
                                    occluded=self.disturb.beacon_occluded(beacon))
        result = FrameResult(self.frame_idx, self.t, frame, pose, los, detections, track, stats)
        self.t += dt
        self.frame_idx += 1
        return result
