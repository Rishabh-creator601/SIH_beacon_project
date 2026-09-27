"""Virtual environment: sky background, stars and moving targets.

Instead of cropping a giant pre-rendered image, the scene renders only
what the camera can currently see. Every light source is drawn as a 2-D
Gaussian spot at a sub-pixel position, which mimics the point-spread
function of a real optical system.
"""

import numpy as np

from .targets import Target


SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def add_gaussian(img, x, y, sigma, peak):
    """Add a Gaussian spot centred at sub-pixel (x, y) to a float image."""
    r = int(np.ceil(3.0 * sigma))
    ix, iy = int(np.floor(x)), int(np.floor(y))
    x0, x1 = max(ix - r, 0), min(ix + r + 2, img.shape[1])
    y0, y1 = max(iy - r, 0), min(iy + r + 2, img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    dx = np.arange(x0, x1) - x
    dy = np.arange(y0, y1) - y
    g = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / (2.0 * sigma * sigma))
    img[y0:y1, x0:x1] += peak * g


def add_streak(img, x, y, sigma, peak, ux, uy):
    """Add a motion-blurred spot: a Gaussian 'line' centred at (x, y).

    (ux, uy) is the full streak vector in pixels. Brightness falls off as a
    Gaussian of the distance to the segment, and the peak is lowered so the
    total flux equals that of the sharp spot.
    """
    length = float(np.hypot(ux, uy))
    if length < 0.5:
        add_gaussian(img, x, y, sigma, peak)
        return
    r = int(np.ceil(3.0 * sigma))
    xa, xb = x - ux / 2, x + ux / 2
    ya, yb = y - uy / 2, y + uy / 2
    x0 = max(int(np.floor(min(xa, xb))) - r, 0)
    x1 = min(int(np.floor(max(xa, xb))) + r + 2, img.shape[1])
    y0 = max(int(np.floor(min(ya, yb))) - r, 0)
    y1 = min(int(np.floor(max(ya, yb))) + r + 2, img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    px = np.arange(x0, x1)[None, :] - xa
    py = np.arange(y0, y1)[:, None] - ya
    # Projection of each pixel onto the segment, clamped to its ends.
    s = np.clip((px * ux + py * uy) / (length * length), 0.0, 1.0)
    d2 = (px - s * ux) ** 2 + (py - s * uy) ** 2
    # Flux of this shape = peak_eff * (2*pi*sigma^2 + sqrt(2*pi)*sigma*L).
    peak_eff = peak / (1.0 + length / (SQRT_2PI * sigma))
    img[y0:y1, x0:x1] += peak_eff * np.exp(-d2 / (2.0 * sigma * sigma))


class Scene:
    def __init__(self, cfg, rng):
        w = cfg["world"]
        self.az_range = tuple(w["az_range"])
        self.el_range = tuple(w["el_range"])
        self.bounds = (*self.az_range, *self.el_range)
        self.background_level = float(w.get("background_level", 18))
        self.star_sigma = float(w.get("star_sigma_px", 0.7))

        # Background stars: fixed in the world, so they move across the
        # image when the camera pans (realistic clutter).
        n = int(w.get("num_stars", 0))
        lo, hi = w.get("star_intensity", [15, 90])
        self.star_az = rng.uniform(*self.az_range, n)
        self.star_el = rng.uniform(*self.el_range, n)
        # Power-law brightness: many faint stars, few bright ones.
        self.star_peak = lo + (hi - lo) * rng.random(n) ** 3

        self.targets = [Target(t, self.bounds, rng) for t in cfg["targets"]]
        beacons = [t for t in self.targets if t.is_beacon]
        if len(beacons) != 1:
            raise ValueError("Exactly one target must have is_beacon: true")
        self.beacon = beacons[0]

    def update(self, t, dt):
        for target in self.targets:
            target.update(t, dt)

    def render(self, camera, pose=None, disturb=None):
        """Render the view seen from ``pose`` as a float32 image (not clipped).

        ``disturb`` (a DisturbanceModel) optionally adds turbulence effects to
        each light source, extra transient sources such as glints, and motion
        blur. Motion blur is computed per source from its motion RELATIVE to
        the camera: a target the camera is following stays sharp while the
        background stars streak, exactly as in a real tracking camera.
        """
        pose = pose or camera.pose
        img = np.full((camera.height, camera.width), self.background_level, np.float32)
        blur = disturb.psf_blur_px if disturb else 0.0
        exposure, cam_rate = disturb.motion_blur(camera) if disturb else (0.0, (0.0, 0.0))

        def spot(x, y, sigma, peak, src_rate=(0.0, 0.0)):
            if blur > 0:
                # Turbulence widens the spot; total flux is conserved.
                wide = float(np.hypot(sigma, blur))
                peak *= (sigma / wide) ** 2
                sigma = wide
            # Apparent image motion (px) of this source during the exposure.
            ux = (src_rate[0] - cam_rate[0]) * camera.ppd_x * exposure
            uy = -(src_rate[1] - cam_rate[1]) * camera.ppd_y * exposure
            add_streak(img, x, y, sigma, peak, ux, uy)

        xs, ys = camera.world_to_pixel(self.star_az, self.star_el, pose)
        m = 4
        visible = (xs > -m) & (xs < camera.width + m) & (ys > -m) & (ys < camera.height + m)
        for x, y, p in zip(xs[visible], ys[visible], self.star_peak[visible]):
            spot(x, y, self.star_sigma, p)

        for target in self.targets:
            dx, dy, gain = disturb.source_effect(target) if disturb else (0.0, 0.0, 1.0)
            peak = target.current_intensity * gain
            if peak <= 0:
                continue
            x, y = camera.world_to_pixel(target.az, target.el, pose)
            spot(float(x) + dx, float(y) + dy, target.sigma_px, peak, (target.vaz, target.vel))

        for az, el, peak, sigma in (disturb.extra_sources() if disturb else []):
            x, y = camera.world_to_pixel(az, el, pose)
            spot(float(x), float(y), sigma, peak)
        return img


def to_uint8(img):
    """Clip a float image into an 8-bit sensor frame (saturation included)."""
    return np.clip(img, 0, 255).astype(np.uint8)
