"""Beacon tracking: Kalman filter + acquisition state machine.

Tracking is done in *world* angular coordinates, not image pixels.
Because each detection is converted using the camera pose at capture
time, the camera's own motion does not look like target motion.

State machine (the PAT acquisition sequence):

    SEARCH ──detection──► ACQUIRING ──N hits──► TRACKING ◄─ short gaps (blink,
       ▲                      │ misses              │        fades) are held
       │                      ▼                     ▼ longer gap
       └──────────────────────┴── too many ──── COASTING
                                  misses     (predict only; a hit
                                              returns to TRACKING)
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TrackState(str, Enum):
    SEARCH = "SEARCH"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    COASTING = "COASTING"


class KalmanFilterCV:
    """2-D constant-velocity Kalman filter. State = [az, el, v_az, v_el]."""

    def __init__(self, az, el, process_noise, measurement_noise):
        self.x = np.array([az, el, 0.0, 0.0])
        self.P = np.diag([measurement_noise ** 2] * 2 + [25.0] * 2)
        self.q = process_noise
        self.R = np.eye(2) * measurement_noise ** 2
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # Discrete white-noise acceleration model.
        g = np.array([0.5 * dt * dt, dt])
        q1 = np.outer(g, g) * self.q ** 2
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = q1
        Q[np.ix_([1, 3], [1, 3])] = q1
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z):
        """Return (residual, innovation covariance) for measurement z."""
        y = np.asarray(z) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis2(self, z):
        y, S = self.innovation(z)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z):
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


@dataclass
class TrackOutput:
    state: TrackState
    estimate: tuple = None        # (az, el, v_az, v_el) or None
    selected: object = None       # Detection associated this frame
    last_known: tuple = None      # last confirmed (az, el), used by search
    last_velocity: tuple = (0.0, 0.0)   # velocity when the track was lost


class BeaconTracker:
    """Kalman tracker + acquisition state machine, optionally identity-aware.

    With an identification method enabled (``id_cfg.method`` != none) each
    detection carries ``score`` = P(beacon) and ``extra['tid']``. Then:
      * SEARCH picks the most beacon-like candidate, skipping blacklisted ones;
      * ACQUIRING also *verifies* identity: the candidate is confirmed only
        when its score reaches ``accept_score``, and rejected (blacklisted
        for ``blacklist_s``) if it scores below ``reject_score`` or is not
        confirmed within ``verify_max_frames``;
      * TRACKING drops a lock whose score stays below ``reject_score`` for
        ``drop_frames`` below ``drop_score`` (recovery from a false lock).
    """

    def __init__(self, cfg, id_cfg=None):
        self.acquire_frames = int(cfg.get("acquire_frames", 5))
        self.acquire_max_misses = int(cfg.get("acquire_max_misses", 4))
        self.hold_frames = int(cfg.get("hold_frames", 6))
        self.max_coast = int(cfg.get("max_coast_frames", 30))
        self.q = float(cfg.get("process_noise", 3.0))
        self.r = float(cfg.get("measurement_noise", 0.02))
        self.gate = float(cfg.get("gate_chi2", 11.6))
        self.gate_min = float(cfg.get("gate_min_deg", 0.15))
        self.gate_growth = float(cfg.get("gate_growth_per_miss", 0.5))
        self.gate_growth_max = float(cfg.get("gate_growth_max", 4.0))
        self.coast_max_sigma = float(cfg.get("coast_max_sigma_deg", 1.0))

        id_cfg = id_cfg or {}
        self.use_identity = id_cfg.get("method", "none") != "none"
        self.accept = float(id_cfg.get("accept_score", 0.7))
        self.reject = float(id_cfg.get("reject_score", 0.2))
        self.min_reject = int(id_cfg.get("min_reject_frames", 20))
        self.verify_max = int(id_cfg.get("verify_max_frames", 90))
        self.drop_frames = int(id_cfg.get("drop_frames", 30))
        self.drop_score = float(id_cfg.get("drop_score", 0.35))
        self.blacklist_s = float(id_cfg.get("blacklist_s", 4.0))
        self.rejections = 0            # candidates rejected during verification
        self.dropped_locks = 0         # locks dropped because identity failed
        self.reset()

    def predicted_position(self, dt):
        """Where the target is expected this frame (for ROI detection), or None."""
        if self.kf is None:
            return None
        return (float(self.kf.x[0] + self.kf.x[2] * dt), float(self.kf.x[1] + self.kf.x[3] * dt))

    def position_sigma(self):
        """1-sigma position uncertainty of the current estimate (deg)."""
        if self.kf is None:
            return float("inf")
        return float(np.sqrt(self.kf.P[0, 0] + self.kf.P[1, 1]))

    def reset(self):
        self.state = TrackState.SEARCH
        self.kf = None
        self.hits = 0
        self.misses = 0
        self.last_known = None
        self.last_velocity = (0.0, 0.0)
        self.time = 0.0
        self.blacklist = {}            # tracklet id -> time until which it is ignored
        self.locked_tid = None
        self.verify_count = 0
        self.low_count = 0

    def _start_track(self, det):
        self.kf = KalmanFilterCV(det.az, det.el, self.q, self.r)
        self.state = TrackState.ACQUIRING
        self.hits, self.misses = 1, 0
        self.locked_tid = det.extra.get("tid")
        self.verify_count = self.low_count = 0

    def _abandon(self, blacklist):
        """Give up the current (tentative or false) track and search again."""
        if blacklist and self.locked_tid is not None:
            self.blacklist[self.locked_tid] = self.time + self.blacklist_s
        self.state, self.kf, self.locked_tid = TrackState.SEARCH, None, None

    def _output(self, selected=None):
        est = tuple(self.kf.x) if self.kf is not None else None
        return TrackOutput(self.state, est, selected, self.last_known, self.last_velocity)

    def _usable(self, detections):
        if not self.use_identity:
            return detections
        self.blacklist = {k: v for k, v in self.blacklist.items() if v > self.time}
        out = []
        for d in detections:
            tid = d.extra.get("tid")
            if tid in self.blacklist:
                continue
            # Clearly not the beacon - unless it is the target we already hold
            # (a lock is dropped by the drop_frames rule, not by one bad frame).
            if self._clearly_not_beacon(d) and tid != self.locked_tid:
                continue
            out.append(d)
        return out

    def step(self, detections, dt):
        """Advance one frame. ``detections`` must have az/el filled in."""
        self.time += dt
        detections = self._usable(detections)

        if self.state == TrackState.SEARCH:
            if detections:
                # No prior: take the most beacon-like candidate (brightness
                # breaks ties, and is the only criterion without identification).
                best = max(detections, key=lambda d: (d.score, d.flux))
                self._start_track(best)
                return self._output(best)
            return self._output()

        # --- An active track exists: predict, then associate -------------
        # A detection is accepted if it is statistically consistent with the
        # prediction (Mahalanobis gate) OR simply very close to it (minimum
        # gate) - the latter keeps lock through sudden manoeuvres that the
        # constant-velocity model did not expect. The minimum gate widens with
        # every consecutive missed frame (up to 4x), because the prediction
        # grows less certain while the target is not seen. With
        # identification, the candidate we already follow wins over others.
        self.kf.predict(dt)
        gate_min = self.gate_min * min(1.0 + self.gate_growth * self.misses, self.gate_growth_max)
        best, best_key = None, None
        for det in detections:
            d2 = self.kf.mahalanobis2((det.az, det.el))
            near = np.hypot(det.az - self.kf.x[0], det.el - self.kf.x[1]) < gate_min
            if not (d2 < self.gate or near):
                continue
            same = self.use_identity and det.extra.get("tid") == self.locked_tid
            key = (not same, d2)
            if best_key is None or key < best_key:
                best, best_key = det, key

        if best is not None:
            self.kf.update((best.az, best.el))
            self.hits += 1
            self.misses = 0
            self.last_known = (best.az, best.el)
            if best.extra.get("tid") is not None:
                self.locked_tid = best.extra["tid"]
            if self.use_identity and not self._identity_ok(best):
                return self._output()
            if self.state == TrackState.ACQUIRING and self.hits >= self.acquire_frames:
                if not self.use_identity or (best.extra.get("ready") and best.score >= self.accept):
                    self.state = TrackState.TRACKING
            elif self.state == TrackState.COASTING:
                self.state = TrackState.TRACKING
            return self._output(best)

        # --- No detection inside the gate --------------------------------
        self.misses += 1
        if self.state == TrackState.ACQUIRING:
            self.verify_count += 1
            if self.misses > self.acquire_max_misses:   # tentative track faded away
                # The search resumes its spiral where it left off. (Tried and
                # rejected on multi-seed tests: re-centring the spiral on the
                # faded candidate - a decoy that blinks fully off traps the
                # search - and a longer fade tolerance during verification.)
                self._abandon(blacklist=False)
            elif self.use_identity and self.verify_count > self.verify_max:
                self.rejections += 1
                self._abandon(blacklist=True)
        elif self.state == TrackState.TRACKING:
            # Short gaps (blink-off, scintillation fades) keep the lock.
            if self.misses > self.hold_frames:
                self.state = TrackState.COASTING
        elif self.state == TrackState.COASTING and (
                self.misses > self.max_coast or self.position_sigma() > self.coast_max_sigma):
            # Give up coasting when either the frame budget runs out or the
            # prediction has become too uncertain to be worth following
            # (fast / manoeuvring targets reach this much sooner).
            self.last_known = tuple(self.kf.x[:2])
            self.last_velocity = tuple(self.kf.x[2:])
            self._abandon(blacklist=False)
        return self._output()

    def _clearly_not_beacon(self, det):
        """Rejecting needs more evidence than accepting: wrongly rejecting the
        real beacon (it is then ignored for blacklist_s) costs more than
        taking a little longer to decide."""
        return (det.extra.get("ready", False) and det.score < self.reject
                and det.extra.get("n_scored", 0) >= self.min_reject)

    def _identity_ok(self, det):
        """Identity checks after an associated detection. False = track abandoned."""
        ready = det.extra.get("ready", False)
        if self.state == TrackState.ACQUIRING:
            self.verify_count += 1
            if self._clearly_not_beacon(det) or self.verify_count > self.verify_max:
                self.rejections += 1
                self._abandon(blacklist=True)
                return False
        elif self.state in (TrackState.TRACKING, TrackState.COASTING):
            # Separate, stricter threshold for a lock: the score of a locked
            # target is a long (3 s) average, so it only sinks this low when
            # the evidence has really turned against it (a false lock).
            low = ready and det.score < self.drop_score
            self.low_count = self.low_count + 1 if low else 0
            if self.low_count >= self.drop_frames:
                self.dropped_locks += 1
                self._abandon(blacklist=True)
                return False
        return True
