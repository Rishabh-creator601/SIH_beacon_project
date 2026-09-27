"""Random scene generation.

With ``randomize.enabled: true`` every run gets a new scene:
  * the beacon's path: random type (lissajous / circular / random walk),
    random start point, size, speed and direction;
  * a random number of decoys, each steady or blinking at a wrong rate,
    with its own random path and brightness;
  * random atmospheric turbulence and platform vibration strength, and
    (with some probability) clouds;
  * a random initial pointing error: the camera starts aimed at the
    beacon's "reported" position (as from GPS / ephemeris data), off by a
    few degrees, and must search from there.

The scene is fully determined by the run's seed, which is written to the
performance report, so any interesting run can be reproduced exactly with
``--seed``. Ranges come from the ``randomize`` section of the config.
"""

import math

import numpy as np

from .config import deep_merge
from .targets import Target


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _random_point(rng, bounds, margin):
    az_min, az_max, el_min, el_max = bounds
    return [float(rng.uniform(az_min + margin, az_max - margin)),
            float(rng.uniform(el_min + margin, el_max - margin))]


def random_path(rng, bounds, kinds, speed):
    """A trajectory config of a random kind that stays inside ``bounds``.

    ``speed`` is the target's typical angular speed in deg/s; each kind's
    parameters are chosen so that it moves at about that speed.
    """
    az_min, az_max, el_min, el_max = bounds
    kind = str(rng.choice(kinds))
    if kind == "lissajous":
        room_az = (az_max - az_min) / 2 - 3.0
        room_el = (el_max - el_min) / 2 - 3.0
        amp = [float(rng.uniform(0.3, 0.9) * room_az), float(rng.uniform(0.3, 0.9) * room_el)]
        center = [float(rng.uniform(az_min + amp[0] + 2, az_max - amp[0] - 2)),
                  float(rng.uniform(el_min + amp[1] + 2, el_max - amp[1] - 2))]
        # Peak speed of A*sin(2*pi*f*t) is 2*pi*f*A: pick f to match `speed`.
        ratio = float(rng.uniform(0.5, 1.6))              # shape of the figure
        f_az = speed / (2 * math.pi * math.hypot(amp[0], amp[1] * ratio))
        return kind, {"type": kind, "center": center, "amplitude": amp,
                      "frequency": [f_az, f_az * ratio],
                      "phase": [float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0, 2 * math.pi))]}
    if kind == "circular":
        room = min(az_max - az_min, el_max - el_min) / 2 - 3.0
        radius = float(rng.uniform(0.25, 0.8) * room)
        center = [float(rng.uniform(az_min + radius + 2, az_max - radius - 2)),
                  float(rng.uniform(el_min + radius + 2, el_max - radius - 2))]
        period = 2 * math.pi * radius / max(speed, 0.1) * float(rng.choice([-1, 1]))
        return kind, {"type": kind, "center": center, "radius": radius, "period": period,
                      "phase": float(rng.uniform(0, 2 * math.pi))}
    # random_walk: smooth, unpredictable manoeuvres
    return "random_walk", {"type": "random_walk", "start": _random_point(rng, bounds, 5.0),
                           "speed": speed, "turn_rate_std": float(rng.uniform(0.2, 0.9)),
                           "turn_tau_s": float(rng.uniform(1.0, 3.0))}


def randomize_config(cfg, rng):
    """Return (new_cfg, info). ``info`` is None when randomisation is off."""
    r = cfg.get("randomize") or {}
    if not r.get("enabled", False):
        return cfg, None
    w = cfg["world"]
    bounds = (*w["az_range"], *w["el_range"])

    # --- Beacon: keep its identity (blink code) but give it a random path.
    beacon = dict(next(t for t in cfg["targets"] if t.get("is_beacon")))
    speed = _u(rng, r.get("beacon_speed_dps", [0.8, 5.0]))
    path, beacon["trajectory"] = random_path(rng, bounds, r.get("beacon_paths",
                                             ["lissajous", "circular", "random_walk"]), speed)
    beacon["intensity"] = _u(rng, r.get("beacon_intensity", [190, 250]))
    targets = [beacon]

    # --- Decoys: random count, behaviour, brightness and path.
    lo, hi = r.get("decoys", [2, 5])
    n_decoys = int(rng.integers(lo, hi + 1))
    beacon_hz = float((beacon.get("blink") or {}).get("frequency_hz", 4.0))
    blinking = 0
    for i in range(n_decoys):
        d = {"name": f"decoy_{i + 1}",
             "intensity": _u(rng, r.get("decoy_intensity", [150, 240])),
             "sigma_px": float(rng.uniform(1.6, 2.4))}
        _, d["trajectory"] = random_path(rng, bounds, ["lissajous", "circular", "random_walk"],
                                         _u(rng, r.get("decoy_speed_dps", [0.3, 4.0])))
        if rng.random() < r.get("decoy_blink_probability", 0.5):
            # Blinks, but clearly not at the beacon's rate.
            lo_band, hi_band = (0.5, beacon_hz - 1.5), (beacon_hz + 2.0, beacon_hz + 6.0)
            band = lo_band if rng.random() < 0.5 else hi_band
            d["blink"] = {"frequency_hz": float(rng.uniform(*band)), "duty": float(rng.uniform(0.3, 0.7)),
                          "off_level": float(rng.uniform(0.0, 0.5)), "phase": float(rng.uniform(0, 1))}
            blinking += 1
        targets.append(d)

    # --- Initial pointing: like a real terminal, the camera is first pointed
    # open-loop at the partner's reported position (GPS / ephemeris), which is
    # off by a random error; the camera must then search that uncertainty.
    start = Target(beacon, bounds, np.random.default_rng(0))
    err = _u(rng, r.get("initial_pointing_error_deg", [3.0, 10.0]))
    ang = float(rng.uniform(0, 2 * math.pi))
    az_min, az_max, el_min, el_max = bounds
    pointing = [float(np.clip(start.az + err * math.cos(ang), az_min, az_max)),
                float(np.clip(start.el + err * math.sin(ang), el_min, el_max))]

    # --- Atmosphere and platform.
    turb = _u(rng, r.get("turbulence_strength", [0.5, 2.0]))
    vib = _u(rng, r.get("vibration_strength", [0.3, 2.0]))
    clouds = bool(rng.random() < r.get("cloud_probability", 0.5))
    overrides = {
        "camera": {"initial_pointing": pointing},
        "targets": targets,
        "disturbances": {
            "turbulence": {"enabled": True, "strength": turb},
            "vibration": {"enabled": True, "strength": vib},
            "clouds": {"enabled": clouds, "count": int(rng.integers(2, 7))},
        },
    }
    info = {
        "beacon_path": path,
        "beacon_speed_dps": round(speed, 2),
        "decoys": n_decoys,
        "blinking_decoys": blinking,
        "initial_pointing_error_deg": round(err, 2),
        "turbulence_strength": round(turb, 2),
        "vibration_strength": round(vib, 2),
        "clouds": clouds,
    }
    return deep_merge(cfg, overrides), info
