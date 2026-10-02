"""Disturbance models that degrade the virtual camera feed.

Each effect is a small class with its own ``enabled`` switch and a
``strength`` multiplier (so a GUI slider can scale it from 0 to N).

  Effect                 Physical cause                       Where it acts
  --------------------   ----------------------------------   -----------------------------
  PlatformVibration      engine / rotor / structural modes    line of sight of the camera
  Turbulence             atmospheric refractive-index cells   each light source + whole image
     - scintillation     intensity fluctuation (twinkle)
     - wander            angle-of-arrival jitter
     - blur              beam spreading (larger PSF)
     - warp              heat shimmer across the image
  Clouds                 moving cloud banks                   attenuate everything behind them
  BeaconDropouts         obstruction / beacon power dips     beacon disappears for a while
  Glints                 sun reflections, flares              short-lived false bright spots
  Glare                  sun / moon in or near the FOV        raises background, hides beacon
  SensorModel            camera electronics                   shot/read noise, hot pixels,
                                                              random bright hits; exposure
                                                              time sets motion blur, drawn
                                                              per source in Scene.render

Random processes are first-order Gauss-Markov (Ornstein-Uhlenbeck), i.e.
coloured noise with a correlation time ``tau``. This looks much more like
real turbulence or jitter than independent per-frame noise.
"""

import math

import cv2
import numpy as np


class GaussMarkov:
    """Coloured noise: x[k+1] = a*x[k] + sqrt(1-a^2)*sigma*n,  a = exp(-dt/tau)."""

    def __init__(self, sigma, tau, rng, shape=1):
        self.sigma = float(sigma)
        self.tau = max(float(tau), 1e-6)
        self.rng = rng
        self.x = rng.standard_normal(shape) * self.sigma

    def step(self, dt):
        a = math.exp(-dt / self.tau)
        self.x = a * self.x + math.sqrt(1.0 - a * a) * self.sigma * self.rng.standard_normal(self.x.shape)
        return self.x


def _world_grid(camera, pose, step_px=8):
    """Coarse grid of world (az, el) covering the camera view (for smooth fields)."""
    gw = max(camera.width // step_px, 2)
    gh = max(camera.height // step_px, 2)
    xs = np.linspace(0, camera.width - 1, gw)
    ys = np.linspace(0, camera.height - 1, gh)
    X, Y = np.meshgrid(xs, ys)
    return camera.pixel_to_world(X, Y, pose)


def _upsample(field, camera):
    return cv2.resize(field.astype(np.float32), (camera.width, camera.height),
                      interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------
class PlatformVibration:
    """Line-of-sight jitter: sinusoidal structural modes + broadband random jitter."""

    def __init__(self, cfg, rng):
        self.strength = float(cfg.get("strength", 1.0))
        self.tones = []
        for tone in cfg.get("tones", []):
            self.tones.append((float(tone["freq_hz"]), float(tone["amplitude_deg"]),
                               rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)))
        self.jitter = GaussMarkov(cfg.get("jitter_deg", 0.0), cfg.get("jitter_tau_s", 0.05), rng, 2)
        self.offset = np.zeros(2)
        self.rate = np.zeros(2)          # deg/s, used for motion blur

    def update(self, t, dt):
        off = self.jitter.step(dt).copy()
        for f, amp, pa, pe in self.tones:
            w = 2 * math.pi * f * t
            off += amp * np.array([math.sin(w + pa), math.sin(w + pe)])
        off *= self.strength
        self.rate = (off - self.offset) / dt if dt > 0 else np.zeros(2)
        self.offset = off


class Turbulence:
    def __init__(self, cfg, rng, camera):
        self.rng = rng
        self.base = {k: float(cfg.get(k, d)) for k, d in (
            ("scintillation_index", 0.3), ("wander_px", 1.0), ("blur_px", 0.5), ("warp_px", 0.0))}
        self.sci_tau = float(cfg.get("scintillation_tau_s", 0.03))
        self.wander_tau = float(cfg.get("wander_tau_s", 0.1))
        self._sources = {}               # per-target random processes

        # Heat-shimmer warp: smooth random displacement field on a coarse grid.
        self.warp = None
        if self.base["warp_px"] > 0:
            gw, gh = cfg.get("warp_grid", [16, 12])
            self.warp = GaussMarkov(self.base["warp_px"], cfg.get("warp_tau_s", 0.15), rng, (2, gh, gw))
            gx, gy = np.meshgrid(np.arange(camera.width, dtype=np.float32),
                                 np.arange(camera.height, dtype=np.float32))
            self._grid = (gx, gy)
        self.set_strength(float(cfg.get("strength", 1.0)))

    def set_strength(self, s):
        """Scale every turbulence effect; may be called while running."""
        self.strength = max(float(s), 0.0)
        self.sci_sigma = self.base["scintillation_index"] * self.strength
        self.wander_px = self.base["wander_px"] * self.strength
        self.psf_blur_px = self.base["blur_px"] * self.strength
        for sci, wander in self._sources.values():
            sci.sigma, wander.sigma = self.sci_sigma, self.wander_px
        if self.warp is not None:
            self.warp.sigma = self.base["warp_px"] * self.strength

    def _procs(self, target):
        key = id(target)
        if key not in self._sources:
            self._sources[key] = (GaussMarkov(self.sci_sigma, self.sci_tau, self.rng),
                                  GaussMarkov(self.wander_px, self.wander_tau, self.rng, 2))
        return self._sources[key]

    def update(self, dt, targets):
        for target in targets:
            sci, wander = self._procs(target)
            sci.step(dt)
            wander.step(dt)
        if self.warp is not None:
            self.warp.step(dt)

    def source_effect(self, target):
        """(dx_px, dy_px, intensity gain) for one light source this frame."""
        sci, wander = self._procs(target)
        # Log-normal intensity with unit mean.
        gain = math.exp(float(sci.x[0]) - 0.5 * self.sci_sigma ** 2)
        return float(wander.x[0]), float(wander.x[1]), gain

    def apply_warp(self, img):
        if self.warp is None:
            return img
        h, w = img.shape
        # Upsample both displacement components in one call (2-channel image).
        coarse = np.ascontiguousarray(np.moveaxis(self.warp.x, 0, -1), dtype=np.float32)
        flow = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        gx, gy = self._grid
        return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)


class Clouds:
    """Soft Gaussian cloud blobs drifting through the world with the wind."""

    def __init__(self, cfg, rng, bounds):
        s = float(cfg.get("strength", 1.0))
        self.bounds = bounds
        self.wind = np.array(cfg.get("wind_dps", [0.8, 0.2]), float)
        self.brightness = float(cfg.get("brightness", 35))
        az_min, az_max, el_min, el_max = bounds
        n = int(cfg.get("count", 4))
        self.az = rng.uniform(az_min, az_max, n)
        self.el = rng.uniform(el_min, el_max, n)
        self.size = rng.uniform(*cfg.get("size_deg", [1.5, 4.0]), n)
        self.depth = rng.uniform(*cfg.get("optical_depth", [1.0, 4.0]), n) * s
        # Relative distance of each cloud (0 = camera, 1 = stars). Drawn from a
        # generator seeded by the cloud positions so the shared random stream -
        # and with it every other part of the scene - is unchanged.
        own = np.random.default_rng(np.frombuffer(self.az.tobytes(), np.uint32))
        self.distance = own.uniform(*cfg.get("distance", [0.3, 0.7]), n)

    def update(self, dt):
        az_min, az_max, el_min, el_max = self.bounds
        self.az = (self.az + self.wind[0] * dt - az_min) % (az_max - az_min) + az_min
        self.el = (self.el + self.wind[1] * dt - el_min) % (el_max - el_min) + el_min

    def _depth(self, az, el, behind=None):
        """Optical depth along (az, el); with ``behind`` = a distance, only the
        clouds nearer than that distance count."""
        az, el = np.asarray(az, float), np.asarray(el, float)
        tau = np.zeros(np.broadcast(az, el).shape)
        for ca, ce, sz, d, r in zip(self.az, self.el, self.size, self.depth, self.distance):
            if behind is not None and r >= behind:
                continue
            tau += d * np.exp(-((az - ca) ** 2 + (el - ce) ** 2) / (2 * sz * sz))
        return tau

    def transmission_at(self, az, el, behind=None):
        return float(np.exp(-self._depth(az, el, behind)))

    def apply(self, img, camera, pose):
        gaz, gel = _world_grid(camera, pose)
        depth = self._depth(gaz, gel)
        if depth.max() < 0.01:           # no cloud in view: skip the full-frame work
            return img
        T = _upsample(np.exp(-depth), camera)
        # img*T + brightness*(1-T)  ==  (img - brightness)*T + brightness
        return cv2.add(cv2.multiply(cv2.subtract(img, self.brightness), T), self.brightness)


class Glare:
    """A very bright extended source (sun/moon) fixed in the world."""

    def __init__(self, cfg):
        self.pos = cfg.get("position_deg", [10.0, 12.0])
        self.sigma = float(cfg.get("size_deg", 3.0))
        self.level = float(cfg.get("level", 120)) * float(cfg.get("strength", 1.0))

    def apply(self, img, camera, pose):
        gaz, gel = _world_grid(camera, pose)
        d2 = (gaz - self.pos[0]) ** 2 + (gel - self.pos[1]) ** 2
        field = self.level * np.exp(-d2 / (2 * self.sigma ** 2))
        if field.max() < 0.5:            # glare source far outside the view
            return img
        return cv2.add(img, _upsample(field, camera))


class BeaconDropouts:
    """Random events where the beacon is fully blocked (e.g. structure in the path)."""

    def __init__(self, cfg, rng):
        self.rng = rng
        self.mean_interval = float(cfg.get("mean_interval_s", 15.0))
        self.duration = cfg.get("duration_s", [0.5, 2.0])
        self.start_after = float(cfg.get("start_after_s", 5.0))
        self.until = -1.0

    def update(self, t, dt):
        if t >= self.until and t > self.start_after and self.rng.random() < dt / self.mean_interval:
            self.until = t + self.rng.uniform(*self.duration)
        self.active = t < self.until


class Glints:
    """Short-lived bright flashes inside the camera view (false alarms)."""

    def __init__(self, cfg, rng):
        self.rng = rng
        self.rate = float(cfg.get("rate_hz", 0.3)) * float(cfg.get("strength", 1.0))
        self.duration = cfg.get("duration_s", [0.05, 0.4])
        self.intensity = cfg.get("intensity", [150, 255])
        self.sigma = float(cfg.get("sigma_px", 1.5))
        self.active = []                 # [az, el, peak, t_end]

    def update(self, t, dt, camera, pose):
        self.active = [g for g in self.active if g[3] > t]
        if self.rng.random() < self.rate * dt:
            az = pose[0] + self.rng.uniform(-0.45, 0.45) * camera.fov_x
            el = pose[1] + self.rng.uniform(-0.45, 0.45) * camera.fov_y
            self.active.append([az, el, self.rng.uniform(*self.intensity),
                                t + self.rng.uniform(*self.duration)])

    def sources(self):
        return [(g[0], g[1], g[2], self.sigma) for g in self.active]


class SensorModel:
    """Detector noise. (Its exposure time drives the per-source motion blur,
    which is drawn in Scene.render - see DisturbanceModel.motion_blur.)"""

    def __init__(self, cfg, rng, camera):
        self.rng = rng
        self.motion_blur = bool(cfg.get("motion_blur", True))
        self.exposure = float(cfg.get("exposure_s", 0.01))
        self.base = {k: float(cfg.get(k, d)) for k, d in (
            ("read_noise", 3.0), ("shot_noise", 0.5), ("salt_per_frame", 0.0))}
        self.set_strength(float(cfg.get("strength", 1.0)))
        self.salt_level = float(cfg.get("salt_level", 255))
        n_hot = int(cfg.get("hot_pixels", 0))
        self.hot_y = rng.integers(0, camera.height, n_hot)
        self.hot_x = rng.integers(0, camera.width, n_hot)
        self.hot_level = rng.uniform(*cfg.get("hot_pixel_level", [60, 200]), n_hot).astype(np.float32)
        # Unit Gaussian noise is drawn ONCE into a bank twice the frame size;
        # each frame takes a random crop of it. Visually identical to fresh
        # noise, but saves the most expensive per-frame operation.
        h, w = camera.height, camera.width
        self._bank = rng.standard_normal((2 * h, 2 * w), dtype=np.float32)

    def set_strength(self, s):
        """Scale all noise terms; may be called while running."""
        self.strength = max(float(s), 0.0)
        self.read_noise = self.base["read_noise"] * self.strength
        self.shot_noise = self.base["shot_noise"] * self.strength
        self.salt_per_frame = self.base["salt_per_frame"] * self.strength

    def _unit_noise(self, shape):
        h, w = shape
        y = int(self.rng.integers(0, self._bank.shape[0] - h + 1))
        x = int(self.rng.integers(0, self._bank.shape[1] - w + 1))
        return self._bank[y:y + h, x:x + w]

    def apply(self, img):
        # Shot noise ~ sqrt(signal), read noise constant: one Gaussian draw.
        # (OpenCV routines: ~5x faster than the equivalent NumPy on a full frame.)
        var = cv2.max(img, 0.0) * (self.shot_noise ** 2) + self.read_noise ** 2
        sigma = cv2.sqrt(var)
        img = cv2.add(img, cv2.multiply(sigma, self._unit_noise(img.shape)))
        if len(self.hot_x):
            img[self.hot_y, self.hot_x] += self.hot_level
        n_salt = self.rng.poisson(self.salt_per_frame) if self.salt_per_frame > 0 else 0
        if n_salt:
            img[self.rng.integers(0, img.shape[0], n_salt),
                self.rng.integers(0, img.shape[1], n_salt)] = self.salt_level
        return img


# ----------------------------------------------------------------------
class DisturbanceModel:
    """Owns every enabled disturbance and applies them in the right order."""

    def __init__(self, cfg, scene, camera, rng):
        d = cfg.get("disturbances", {}) or {}
        master = d.get("enabled", True)

        def section(name):
            c = d.get(name) or {}
            return c if master and c.get("enabled", False) else None

        c = section("vibration");  self.vibration = PlatformVibration(c, rng) if c else None
        c = section("turbulence"); self.turbulence = Turbulence(c, rng, camera) if c else None
        c = section("clouds");     self.clouds = Clouds(c, rng, scene.bounds) if c else None   # noqa: E702
        c = section("glare");      self.glare = Glare(c) if c else None
        c = section("dropouts");   self.dropouts = BeaconDropouts(c, rng) if c else None
        c = section("glints");     self.glints = Glints(c, rng) if c else None
        c = section("sensor");     self.sensor = SensorModel(c, rng, camera) if c else None
        self.scene = scene
        self._cfg, self._camera, self._rng = d, camera, rng

    # --- live adjustment (GUI sliders) ----------------------------------
    LIVE = ("turbulence", "vibration", "sensor")

    def strength(self, name):
        model = getattr(self, name)
        return 0.0 if model is None else float(model.strength)

    def set_strength(self, name, value):
        """Change a disturbance's strength while running. A disturbance that
        was switched off is created on demand when the value becomes > 0."""
        if name not in self.LIVE:
            raise ValueError(f"live strength only for {self.LIVE}")
        model = getattr(self, name)
        if model is None:
            if value <= 0:
                return
            c = dict(self._cfg.get(name) or {}, strength=value)
            model = {"turbulence": lambda: Turbulence(c, self._rng, self._camera),
                     "vibration": lambda: PlatformVibration(c, self._rng),
                     "sensor": lambda: SensorModel(c, self._rng, self._camera)}[name]()
            setattr(self, name, model)
        elif name == "vibration":
            model.strength = max(float(value), 0.0)
        else:
            model.set_strength(value)

    # --- time evolution ------------------------------------------------
    def update(self, t, dt, camera):
        if self.vibration:
            self.vibration.update(t, dt)
        if self.turbulence:
            self.turbulence.update(dt, self.scene.targets)
        if self.clouds:
            self.clouds.update(dt)
        if self.dropouts:
            self.dropouts.update(t, dt)
        if self.glints:
            self.glints.update(t, dt, camera, camera.pose)

    # --- geometry --------------------------------------------------------
    def line_of_sight(self, pose):
        """Actual pointing = commanded pose + platform vibration."""
        if self.vibration is None:
            return pose
        return pose[0] + float(self.vibration.offset[0]), pose[1] + float(self.vibration.offset[1])

    # --- per-source effects (used by Scene.render) ----------------------
    @property
    def psf_blur_px(self):
        return self.turbulence.psf_blur_px if self.turbulence else 0.0

    def source_effect(self, target):
        dx = dy = 0.0
        gain = 1.0
        if self.turbulence:
            dx, dy, gain = self.turbulence.source_effect(target)
        if self.dropouts and target.is_beacon and self.dropouts.active:
            gain = 0.0
        return dx, dy, gain

    def extra_sources(self):
        return self.glints.sources() if self.glints else []

    def motion_blur(self, camera):
        """(exposure_s, camera angular rate incl. vibration) for streak rendering."""
        if not (self.sensor and self.sensor.motion_blur):
            return 0.0, (0.0, 0.0)
        rate_az, rate_el = camera.rate_az, camera.rate_el
        if self.vibration:
            rate_az += float(self.vibration.rate[0])
            rate_el += float(self.vibration.rate[1])
        return self.sensor.exposure, (rate_az, rate_el)

    def beacon_occluded(self, beacon):
        if self.dropouts and self.dropouts.active:
            return True
        # Below ~50 % transmission the dimmed beacon falls under the detection
        # threshold, so the frame counts as occluded (not a tracker failure).
        return self.cloud_transmission(beacon) < 0.5

    def cloud_transmission(self, target):
        """Fraction of a target's light that passes the clouds in front of it."""
        if not self.clouds:
            return 1.0
        return self.clouds.transmission_at(target.az, target.el, behind=target.distance)

    # --- whole-image effects ---------------------------------------------
    def apply_clouds(self, img, camera, los):
        """Clouds dim and veil the sky and stars behind them (targets are added
        afterwards, each with its own cloud transmission)."""
        return self.clouds.apply(img, camera, los) if self.clouds else img

    def process_image(self, img, camera, los):
        if self.glare:
            img = self.glare.apply(img, camera, los)
        if self.turbulence:
            img = self.turbulence.apply_warp(img)
        if self.sensor:
            img = self.sensor.apply(img)
        return img
