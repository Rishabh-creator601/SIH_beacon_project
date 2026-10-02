"""Pointing controller: turns the tracker's estimate into gimbal rate commands.

* Target known (ACQUIRING / TRACKING / COASTING):
      rate = feedforward * v_target + Kp*e + Ki*∫e + Kd*de/dt
  where e is the angle between the Kalman-estimated target position and
  the camera boresight.

* Target unknown (SEARCH): follow an Archimedean spiral centred on the
  last known target position, with arm spacing smaller than the FOV so
  no part of the sky is skipped. The centre drifts along the target's
  last known velocity (predictive search), because a lost target keeps
  moving. After ``local_passes`` unsuccessful spirals, one raster scan of
  the whole sky follows (so every direction is eventually covered), then
  the local spiral resumes.
"""

import math

import numpy as np

from .tracker import TrackState


class PointingController:
    def __init__(self, cfg, camera):
        self.kp = float(cfg.get("kp", 5.0))
        self.ki = float(cfg.get("ki", 0.5))
        self.kd = float(cfg.get("kd", 0.05))
        self.ff = float(cfg.get("feedforward", 1.0))
        self.i_limit = float(cfg.get("integral_limit", 2.0))

        s = cfg.get("search", {})
        self.spacing = float(s.get("spacing_fov_frac", 0.8)) * min(camera.fov_x, camera.fov_y)
        self.scan_speed = float(s.get("scan_speed_dps", 12.0))
        self.max_radius = float(s.get("max_radius_deg", 25.0))
        self.search_kp = float(s.get("kp", 6.0))
        self.predict_tau = max(float(s.get("predict_tau_s", 3.0)), 1e-3)
        self.local_passes = max(int(s.get("local_passes", 4)), 1)
        # Where the first search is centred: the partner's reported position
        # (GPS / ephemeris prior) if known, else where the camera starts.
        prior = s.get("prior_deg")
        self.home = tuple(map(float, prior)) if prior is not None else (camera.az, camera.el)
        # Optional live position report of the partner (GPS / ephemeris stream):
        # a function returning (az, el). Set by the simulation when available.
        self.prior_fn = None
        self.bounds = (camera.az_min, camera.az_max, camera.el_min, camera.el_max)
        self.reset()

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = None
        self.search_center = None
        self.theta = 0.0
        self.raster_s = None
        self.search_point = None
        self.follow_prior = False

    # ------------------------------------------------------------------
    def compute(self, track, camera, dt):
        """Return commanded (rate_az, rate_el) in deg/s."""
        if track.state != TrackState.SEARCH and track.estimate is not None:
            # Only a confirmed track ends the search. During ACQUIRING the spiral
            # position is kept, so a false alarm (e.g. a glint) resumes the scan
            # where it left off instead of restarting it.
            if track.state in (TrackState.TRACKING, TrackState.COASTING):
                self.search_center = None
            return self._track(track.estimate, camera, dt)
        return self._search(track, camera, dt)

    def _track(self, est, camera, dt):
        az, el, vaz, vel = est
        # The feed-forward term already moves the camera with the target over
        # the next frame, so the feedback term only has to remove the current
        # offset (adding a one-frame lead here as well would double-count it).
        e = np.array([az - camera.az, el - camera.el])
        self.integral = np.clip(self.integral + e * dt, -self.i_limit, self.i_limit)
        de = (e - self.prev_error) / dt if self.prev_error is not None else np.zeros(2)
        self.prev_error = e
        cmd = self.ff * np.array([vaz, vel]) + self.kp * e + self.ki * self.integral + self.kd * de
        return float(cmd[0]), float(cmd[1])

    def _search(self, track, camera, dt):
        self.integral[:] = 0.0
        self.prev_error = None
        if self.search_center is None:
            # Start a new spiral from the last place the beacon was seen - or,
            # before the first sighting, around the partner's reported position.
            self.search_center = track.last_known or self.home
            self.search_velocity = track.last_velocity if track.last_known else (0.0, 0.0)
            self.follow_prior = track.last_known is None and self.prior_fn is not None
            self.search_time = 0.0
            self.theta = 0.0
            self.raster_s = None
            self.spiral_passes = 0
        self.search_time += dt

        if self.follow_prior:
            # The spiral follows the live position report (it moves with the
            # partner, offset by the report's error).
            cx, cy = self.prior_fn()
        else:
            # Predictive search: the spiral centre keeps drifting along the
            # beacon's last known velocity (fading out with time constant tau,
            # since the prediction becomes less trustworthy the longer it runs).
            drift = self.predict_tau * (1.0 - math.exp(-self.search_time / self.predict_tau))
            cx = self.search_center[0] + self.search_velocity[0] * drift
            cy = self.search_center[1] + self.search_velocity[1] * drift

        # Search policy: the local spiral around the best guess is flown
        # `local_passes` times (a hidden target most often reappears near where
        # it was lost), then ONE raster pass over the whole sky (so no
        # direction is missed for ever), then back to the local spiral.
        az_min, az_max, el_min, el_max = self.bounds
        ccx, ccy = min(max(cx, az_min), az_max), min(max(cy, el_min), el_max)
        if (self.raster_s is None and self.theta == 0.0 and self.spiral_passes == 0
                and math.hypot(ccx - camera.az, ccy - camera.el) > self.spacing / 2):
            # Far from the search centre (e.g. the camera starts parked away
            # from the reported position): slew there first, then spiral.
            px, py = ccx, ccy
        elif self.raster_s is None:
            # Local spiral r = spacing * theta / 2pi, at ~constant speed.
            r = self.spacing * self.theta / (2 * math.pi)
            self.theta += self.scan_speed * dt / max(r, self.spacing / 2)
            r = self.spacing * self.theta / (2 * math.pi)
            if r > self.max_radius:
                self.theta, r = 0.0, 0.0
                self.spiral_passes += 1
                if self.spiral_passes >= self.local_passes:
                    if self.prior_fn is not None and not self.follow_prior:
                        # Lost for long: fall back to the live position report
                        # (much faster than scanning the whole sky).
                        self.follow_prior, self.spiral_passes = True, 0
                    else:
                        self.raster_s = 0.0
            px = min(max(cx + r * math.cos(self.theta), az_min), az_max)
            py = min(max(cy + r * math.sin(self.theta), el_min), el_max)
        if self.raster_s is not None:
            # Raster ("lawnmower") scan of the whole sky, row by row.
            self.raster_s += self.scan_speed * dt
            px, py, done = self._raster_point(self.raster_s)
            if done:
                self.raster_s, self.spiral_passes = None, 0
                self.follow_prior = False
        self.search_point = (px, py)
        return (self.search_kp * (px - camera.az), self.search_kp * (py - camera.el))

    def _raster_point(self, s):
        """(x, y, pass_done) at path length ``s`` along a back-and-forth scan."""
        az_min, az_max, el_min, el_max = self.bounds
        half = self.spacing / 2
        rows = max(int(math.ceil((el_max - el_min - 2 * half) / self.spacing)) + 1, 1)
        width = az_max - az_min - 2 * half
        step = (el_max - el_min - 2 * half) / max(rows - 1, 1)
        if s >= rows * width + (rows - 1) * step:          # one full pass done
            return az_min + half, el_min + half, True
        for row in range(rows):
            y = el_max - half - row * step
            if s <= width:
                x = az_min + half + s if row % 2 == 0 else az_max - half - s
                return x, y, False
            s -= width
            if s <= step:
                return (az_max - half if row % 2 == 0 else az_min + half), y - s, False
            s -= step
        return az_min + half, el_min + half, True
