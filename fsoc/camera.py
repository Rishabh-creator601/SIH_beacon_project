"""Virtual pan-tilt camera.

The camera points at (az, el) in the world and sees a rectangular window
of size ``fov_deg``. It is driven by angular-rate commands, and like a
real gimbal it has limited slew rate and acceleration, so it cannot jump
instantly to a new direction.
"""

import numpy as np


class VirtualCamera:
    def __init__(self, cfg, bounds):
        self.width, self.height = cfg["resolution"]
        self.fov_x, self.fov_y = cfg["fov_deg"]
        # Pixels per degree (small-angle linear projection).
        self.ppd_x = self.width / self.fov_x
        self.ppd_y = self.height / self.fov_y
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0

        self.az, self.el = (float(v) for v in cfg.get("initial_pointing", [0.0, 0.0]))
        self.rate_az = self.rate_el = 0.0
        self.max_rate = float(cfg.get("max_rate_dps", 25.0))
        self.max_accel = float(cfg.get("max_accel_dps2", 80.0))
        # Gimbal travel limits = extent of the world.
        self.az_min, self.az_max, self.el_min, self.el_max = bounds

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def _limit_axis(self, rate, cmd, dt):
        """Apply acceleration then rate limits to one axis."""
        max_dv = self.max_accel * dt
        rate += float(np.clip(cmd - rate, -max_dv, max_dv))
        return float(np.clip(rate, -self.max_rate, self.max_rate))

    def command_rate(self, rate_az_cmd, rate_el_cmd, dt):
        """Move the camera for one time step using a commanded angular rate."""
        self.rate_az = self._limit_axis(self.rate_az, rate_az_cmd, dt)
        self.rate_el = self._limit_axis(self.rate_el, rate_el_cmd, dt)
        self.az += self.rate_az * dt
        self.el += self.rate_el * dt
        # Hard stops at the gimbal travel limits.
        if not self.az_min <= self.az <= self.az_max:
            self.az = float(np.clip(self.az, self.az_min, self.az_max))
            self.rate_az = 0.0
        if not self.el_min <= self.el <= self.el_max:
            self.el = float(np.clip(self.el, self.el_min, self.el_max))
            self.rate_el = 0.0

    # ------------------------------------------------------------------
    # Projection between world angles and image pixels
    # ------------------------------------------------------------------
    def world_to_pixel(self, az, el, pose=None):
        """World (az, el) -> image (x, y). Works on scalars or numpy arrays.

        ``pose`` optionally overrides the current pointing (az, el).
        """
        paz, pel = pose if pose is not None else (self.az, self.el)
        x = self.cx + (np.asarray(az) - paz) * self.ppd_x
        y = self.cy - (np.asarray(el) - pel) * self.ppd_y   # image y grows downward
        return x, y

    def pixel_to_world(self, x, y, pose=None):
        paz, pel = pose if pose is not None else (self.az, self.el)
        az = paz + (x - self.cx) / self.ppd_x
        el = pel - (y - self.cy) / self.ppd_y
        return az, el

    def in_fov(self, az, el, pose=None, margin_px=0.0):
        x, y = self.world_to_pixel(az, el, pose)
        return (margin_px <= x <= self.width - 1 - margin_px and
                margin_px <= y <= self.height - 1 - margin_px)

    @property
    def pose(self):
        return self.az, self.el
