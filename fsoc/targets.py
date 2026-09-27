"""Moving targets and their trajectories.

A target lives in world angular coordinates (azimuth, elevation in
degrees). Each target owns a trajectory object that advances its
position every simulation step.
"""

import math

import numpy as np


class Trajectory:
    """Base class. ``step(t, dt)`` returns the (az, el) position at time t."""

    def __init__(self, bounds):
        self.az_min, self.az_max, self.el_min, self.el_max = bounds

    def step(self, t, dt):
        raise NotImplementedError


class StaticTrajectory(Trajectory):
    def __init__(self, cfg, bounds, rng):
        super().__init__(bounds)
        self.pos = tuple(cfg.get("start", [0.0, 0.0]))

    def step(self, t, dt):
        return self.pos


class LinearTrajectory(Trajectory):
    """Constant velocity; reflects off the edges of the world."""

    def __init__(self, cfg, bounds, rng):
        super().__init__(bounds)
        self.az, self.el = cfg.get("start", [0.0, 0.0])
        self.vaz, self.vel = cfg.get("velocity", [2.0, 1.0])

    def step(self, t, dt):
        if t > 0:
            self.az += self.vaz * dt
            self.el += self.vel * dt
        if not self.az_min <= self.az <= self.az_max:
            self.vaz = -self.vaz
            self.az = min(max(self.az, self.az_min), self.az_max)
        if not self.el_min <= self.el <= self.el_max:
            self.vel = -self.vel
            self.el = min(max(self.el, self.el_min), self.el_max)
        return self.az, self.el


class LissajousTrajectory(Trajectory):
    """Smooth figure-of-eight style motion: center + A * sin(2*pi*f*t + phi)."""

    def __init__(self, cfg, bounds, rng):
        super().__init__(bounds)
        self.center = cfg.get("center", [0.0, 0.0])
        self.amp = cfg.get("amplitude", [10.0, 6.0])
        self.freq = cfg.get("frequency", [0.03, 0.05])
        self.phase = cfg.get("phase", [0.0, 0.0])

    def step(self, t, dt):
        return tuple(
            self.center[i] + self.amp[i] * math.sin(2 * math.pi * self.freq[i] * t + self.phase[i])
            for i in range(2)
        )


class CircularTrajectory(Trajectory):
    """Circular orbit, e.g. a UAV loitering around a point."""

    def __init__(self, cfg, bounds, rng):
        super().__init__(bounds)
        self.center = cfg.get("center", [0.0, 0.0])
        self.radius = cfg.get("radius", 8.0)
        self.period = cfg.get("period", 30.0)
        self.phase = cfg.get("phase", 0.0)

    def step(self, t, dt):
        a = 2 * math.pi * t / self.period + self.phase
        return (self.center[0] + self.radius * math.cos(a),
                self.center[1] + self.radius * math.sin(a))


class RandomWalkTrajectory(Trajectory):
    """Unpredictable manoeuvring target with a physically bounded turn rate.

    The turn rate (not the heading) is a smooth random process, so the
    target banks into and out of turns like a real vehicle instead of
    jumping direction from frame to frame.
    """

    def __init__(self, cfg, bounds, rng):
        super().__init__(bounds)
        self.rng = rng
        self.az, self.el = cfg.get("start", [0.0, 0.0])
        self.speed = cfg.get("speed", 3.0)                     # deg/s
        self.turn_rate_std = cfg.get("turn_rate_std", 0.6)     # rad/s (1-sigma)
        self.turn_tau = max(cfg.get("turn_tau_s", 1.5), 1e-3)  # how long a turn lasts
        self.turn_rate = 0.0
        self.heading = rng.uniform(0, 2 * math.pi)
        # Near the world edge the target turns back smoothly (at most
        # max_turn_rate) instead of bouncing off it instantaneously.
        self.max_turn = cfg.get("max_turn_rate", 3.0)          # rad/s
        # The margin must exceed the room a U-turn needs. Heading diagonally
        # into a corner is the worst case: about 2 turning radii
        # (radius = speed / turn rate).
        turn_radius = self.speed / self.max_turn
        half_span = min(self.az_max - self.az_min, self.el_max - self.el_min) / 2
        self.margin = min(max(cfg.get("boundary_margin_deg", 4.0), 2.2 * turn_radius),
                          0.8 * half_span)
        self.avoid_dir = 0.0          # committed avoidance turn direction (+1 / -1)

    def _near_edge(self):
        return (self.az < self.az_min + self.margin or self.az > self.az_max - self.margin or
                self.el < self.el_min + self.margin or self.el > self.el_max - self.margin)

    def _nearest_wall_normal(self):
        """Unit vector pointing inward from the closest world edge."""
        gaps = [(self.az - self.az_min, (1.0, 0.0)), (self.az_max - self.az, (-1.0, 0.0)),
                (self.el - self.el_min, (0.0, 1.0)), (self.el_max - self.el, (0.0, -1.0))]
        return min(gaps)[1]

    def step(self, t, dt):
        if t > 0:
            a = math.exp(-dt / self.turn_tau)
            self.turn_rate = (a * self.turn_rate
                              + math.sqrt(1 - a * a) * self.turn_rate_std * self.rng.standard_normal())
            if self._near_edge():
                to_center = math.atan2((self.el_min + self.el_max) / 2 - self.el,
                                       (self.az_min + self.az_max) / 2 - self.az)
                err = (to_center - self.heading + math.pi) % (2 * math.pi) - math.pi
                if abs(err) > math.pi / 2:
                    # Large turn needed: turn AWAY from the nearest wall at the
                    # maximum rate, and keep that direction until the turn is
                    # done. (Re-deciding every frame makes the target dither
                    # into corners, where the "nearest wall" keeps switching.)
                    if self.avoid_dir == 0.0:
                        nx, ny = self._nearest_wall_normal()
                        cross = math.cos(self.heading) * ny - math.sin(self.heading) * nx
                        self.avoid_dir = 1.0 if cross >= 0 else -1.0
                    self.turn_rate = self.avoid_dir * self.max_turn
                else:
                    self.avoid_dir = 0.0
                    self.turn_rate = max(-self.max_turn, min(self.max_turn, 3.0 * err))
            else:
                self.avoid_dir = 0.0
            self.heading += self.turn_rate * dt
            self.az += self.speed * math.cos(self.heading) * dt
            self.el += self.speed * math.sin(self.heading) * dt
        # Reflect the heading at the world edges.
        if not self.az_min <= self.az <= self.az_max:
            self.heading = math.pi - self.heading
            self.az = min(max(self.az, self.az_min), self.az_max)
        if not self.el_min <= self.el <= self.el_max:
            self.heading = -self.heading
            self.el = min(max(self.el, self.el_min), self.el_max)
        return self.az, self.el


TRAJECTORIES = {
    "static": StaticTrajectory,
    "linear": LinearTrajectory,
    "lissajous": LissajousTrajectory,
    "circular": CircularTrajectory,
    "random_walk": RandomWalkTrajectory,
}


class Target:
    """A point-like light source moving through the virtual sky."""

    def __init__(self, cfg, bounds, rng):
        self.name = cfg.get("name", "target")
        self.is_beacon = bool(cfg.get("is_beacon", False))
        self.intensity = float(cfg.get("intensity", 200))
        self.sigma_px = float(cfg.get("sigma_px", 2.0))
        traj_cfg = cfg.get("trajectory", {"type": "static"})
        kind = traj_cfg.get("type", "static")
        if kind not in TRAJECTORIES:
            raise ValueError(f"Unknown trajectory type '{kind}' for target '{self.name}'")
        self.trajectory = TRAJECTORIES[kind](traj_cfg, bounds, rng)
        self.az, self.el = self.trajectory.step(0.0, 0.0)
        self.vaz = self.vel = 0.0

        # Optional on/off modulation. A beacon that blinks at a known rate can
        # be told apart from steady decoys and stars.
        blink = cfg.get("blink") or {}
        self.blink_hz = float(blink.get("frequency_hz", 0.0))
        self.blink_duty = float(blink.get("duty", 0.5))
        self.blink_off_level = float(blink.get("off_level", 0.0))
        self.blink_phase = float(blink.get("phase", 0.0))
        self.current_intensity = self.intensity

    def update(self, t, dt):
        if self.blink_hz > 0:
            cycle = (t * self.blink_hz + self.blink_phase) % 1.0
            on = cycle < self.blink_duty
            self.current_intensity = self.intensity * (1.0 if on else self.blink_off_level)
        az, el = self.trajectory.step(t, dt)
        if dt > 0:
            self.vaz = (az - self.az) / dt
            self.vel = (el - self.el) / dt
        self.az, self.el = az, el

    @property
    def speed(self):
        return float(np.hypot(self.vaz, self.vel))
