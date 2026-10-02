"""3D situational view (pyqtgraph OpenGL).

The observer - the PAT terminal on its pan-tilt mount - stands at the origin.
Every direction (azimuth, elevation) is drawn on a sky dome of radius R.
The simulated sky patch (az +-30, el +-20 deg) is placed EL0 degrees above
the horizon, straight ahead (+Y), so the scene reads like a ground terminal
looking up at a UAV / satellite.

Shown live:
  * sky dome grid, horizon and the highlighted simulated sky patch
  * the pan-tilt terminal, whose camera head turns with the gimbal
  * the camera's field-of-view pyramid and its footprint on the sky,
    coloured by tracker state, and the history of where it has looked
    (the search spiral / raster becomes visible)
  * the beacon (pulsing with its blink) and its trail, decoys, clouds,
    background stars and the tracker's estimate
"""

import math
from collections import deque

import numpy as np
import pyqtgraph.opengl as gl
from OpenGL import GL
from PySide6 import QtGui

from . import theme

R = 100.0          # sky dome radius (scene units)
EL0 = 35.0         # elevation of the simulated sky patch's centre
MOUNT_H = 6.0      # height of the camera head above ground

# Depth layers (distance from the camera head). Every object keeps its exact
# direction, so the terminal's view is unchanged, but orbiting the view shows
# the layers: stars on the far dome, beacon and decoys in front of it, and
# clouds - as several puffs - closer still.
STAR_R = 1.0 * R
SKY_STARS_R = 1.35 * R     # decorative full-sky star field, beyond the dome
PUFFS = 8                  # spheres per cloud
VIB_EXAGGERATION = 60.0    # visual scale of the camera-head shaking


def depth_r(distance):
    """Simulation's relative distance (0 = camera, 1 = stars) -> 3D radius."""
    return R * (0.45 + 0.55 * np.asarray(distance, float))


# Star colours by temperature: blue-white, white, yellow-white, orange.
STAR_COLORS = np.array([[0.70, 0.80, 1.00], [1.00, 1.00, 1.00], [1.00, 0.95, 0.80], [1.00, 0.80, 0.60]])
STAR_P = [0.25, 0.40, 0.22, 0.13]


def to_xyz(az, el, radius=R):
    """World (az, el) in degrees -> 3D point(s) at ``radius`` from the camera
    head (which sits MOUNT_H above the ground). Works on arrays."""
    a = np.radians(np.asarray(az, float))
    e = np.radians(np.asarray(el, float) + EL0)
    return np.stack([radius * np.sin(a) * np.cos(e),
                     radius * np.cos(a) * np.cos(e),
                     MOUNT_H + radius * np.sin(e)], axis=-1)


def _box_mesh(sx, sy, sz, y0=0.0):
    """Box mesh centred on x/z, extending from y0 to y0+sy along +Y."""
    x, z = sx / 2, sz / 2
    v = np.array([[-x, y0, -z], [x, y0, -z], [x, y0 + sy, -z], [-x, y0 + sy, -z],
                  [-x, y0, z], [x, y0, z], [x, y0 + sy, z], [-x, y0 + sy, z]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return gl.MeshData(vertexes=v, faces=f)


class _CloudMesh(gl.GLMeshItem):
    """Mesh drawn without writing depth (so overlapping translucent puffs all
    blend); the depth mask is restored afterwards - a disabled mask would also
    stop the next frame from clearing the depth buffer."""

    def paint(self):
        try:
            super().paint()
        finally:
            GL.glDepthMask(GL.GL_TRUE)


class SceneView3D(gl.GLViewWidget):
    TRAIL = 240        # beacon trail length (updates)
    LOOK_TRAIL = 600   # line-of-sight history length

    PRESETS = {
        "Terminal": dict(distance=175, elevation=12, azimuth=-90),
        "Overview": dict(distance=260, elevation=28, azimuth=-135),
        "Side": dict(distance=240, elevation=8, azimuth=180),
        "Top": dict(distance=260, elevation=88, azimuth=-90),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(theme.BG)
        self.center = QtGui.QVector3D(*to_xyz(0.0, 0.0, R * 0.55))
        self.set_preset("Terminal")
        self.auto_orbit = False
        self.sim = None
        self._build_static()

    # ------------------------------------------------------------------
    def set_preset(self, name):
        p = self.PRESETS[name]
        self.opts["center"] = self.center
        self.setCameraPosition(distance=p["distance"], elevation=p["elevation"], azimuth=p["azimuth"])

    def _line(self, pts, color, width=1.0, mode="line_strip", antialias=True):
        item = gl.GLLinePlotItem(pos=np.asarray(pts, float), color=color, width=width,
                                 mode=mode, antialias=antialias)
        item.setGLOptions("translucent")
        self.addItem(item)
        return item

    def _build_static(self):
        # Ground grid and horizon ring.
        grid = gl.GLGridItem()
        grid.setSize(2.6 * R, 2.6 * R)
        grid.setSpacing(R / 8, R / 8)
        grid.setColor((40, 60, 90, 90))
        self.addItem(grid)
        t = np.linspace(0, 2 * np.pi, 181)
        self._line(np.c_[R * np.sin(t), R * np.cos(t), np.full_like(t, MOUNT_H)], theme.rgba(theme.ACCENT, 0.35), 1.5)

        # Sky dome: parallels (constant elevation) and meridians (constant azimuth).
        dome = theme.rgba("#4a6a90", 0.22)
        az = np.linspace(-180, 180, 181)
        for e_abs in range(15, 90, 15):
            self._line(to_xyz(az, np.full_like(az, e_abs - EL0)), dome)
        el = np.linspace(-EL0, 90 - EL0, 60)
        for a in range(-180, 180, 30):
            self._line(to_xyz(np.full_like(el, a), el), dome)

        # Compass labels on the horizon.
        for label, a in (("N", 0), ("E", 90), ("S", 180), ("W", -90)):
            p = to_xyz(a, -EL0 + 2, R * 1.06)
            self.addItem(gl.GLTextItem(pos=p, text=label, color=QtGui.QColor(theme.TEXT_DIM)))

        # Terminal mount: base plate + mast (static), camera head (moves).
        base = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=1, cols=24, radius=[5.0, 4.2], length=1.6),
                             color=theme.rgba("#3a4d66"), shader="shaded", smooth=True)
        mast = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=1, cols=16, radius=[1.1, 1.1], length=MOUNT_H),
                             color=theme.rgba("#5b7090"), shader="shaded", smooth=True)
        self.addItem(base)
        self.addItem(mast)
        self.head = gl.GLMeshItem(meshdata=_box_mesh(3.8, 9.5, 3.8, y0=-3.0),
                                  color=theme.rgba("#9fb7d6"), shader="shaded", smooth=False)
        self.lens = gl.GLMeshItem(meshdata=_box_mesh(2.4, 1.2, 2.4, y0=6.5),
                                  color=theme.rgba(theme.ACCENT), shader="shaded", smooth=False)
        self.addItem(self.head)
        self.addItem(self.lens)
        self.addItem(gl.GLTextItem(pos=(7.0, -6.0, 1.5), text="PAT terminal",
                                   color=QtGui.QColor(theme.TEXT_DIM)))

        # Dynamic items (data filled in on reset / update).
        self.patch_outline = self._line(np.zeros((2, 3)), theme.rgba(theme.ACCENT, 0.55), 2.0)
        self.patch_fill = gl.GLMeshItem(meshdata=gl.MeshData(), color=theme.rgba(theme.ACCENT, 0.05),
                                        smooth=False, drawEdges=False)
        self.patch_fill.setGLOptions("translucent")
        self.addItem(self.patch_fill)
        self.patch_labels = []
        # Stars: a faint decorative field over the whole sky (beyond the dome)
        # and the simulated stars of the sky patch, coloured by temperature.
        rng = np.random.default_rng(2024)
        n = 2600
        az = rng.uniform(-180, 180, n)
        el_abs = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(3)), 1, n)))   # uniform on the dome
        mag = rng.random(n) ** 4
        col = np.c_[STAR_COLORS[rng.choice(4, n, p=STAR_P)], 0.15 + 0.6 * mag]
        self.sky_stars = gl.GLScatterPlotItem(pos=to_xyz(az, el_abs - EL0, SKY_STARS_R), size=1.0 + 2.2 * mag,
                                              color=col)
        self.sky_stars.setGLOptions("additive")
        self.addItem(self.sky_stars)
        self.stars = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=2.0, color=(0.8, 0.85, 1.0, 0.35))
        self.stars.setGLOptions("additive")
        self.addItem(self.stars)
        # Clouds: ONE mesh holding every puff (fast), alpha-blended and drawn
        # last without writing depth: a beacon in front of a cloud hides the
        # cloud behind it, a beacon behind a cloud is veiled by it.
        sph = gl.MeshData.sphere(rows=8, cols=12, radius=1.0)
        self._sph_v, self._sph_f = sph.vertexes().astype(np.float32), sph.faces()
        self.clouds = _CloudMesh(meshdata=gl.MeshData(), smooth=True, shader="shaded")
        self.clouds.setGLOptions({GL.GL_DEPTH_TEST: True, GL.GL_BLEND: True, GL.GL_CULL_FACE: True,
                                  "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
                                  "glDepthMask": (GL.GL_FALSE,)})
        self.clouds.setDepthValue(10)
        self.addItem(self.clouds)

        self.look_trail = self._line(np.zeros((2, 3)), theme.rgba(theme.ACCENT, 0.25), 1.0)
        self.los = self._line(np.zeros((2, 3)), theme.rgba(theme.ACCENT, 0.9), 1.5)
        self.frustum = self._line(np.zeros((2, 3)), theme.rgba(theme.GOOD, 0.8), 1.2, mode="lines")
        self.footprint = gl.GLMeshItem(meshdata=gl.MeshData(), color=theme.rgba(theme.GOOD, 0.22),
                                       smooth=False, drawEdges=False)
        self.footprint.setGLOptions("translucent")
        self.addItem(self.footprint)

        self.beacon_trail = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=2.0, antialias=True)
        self.beacon_trail.setGLOptions("translucent")
        self.addItem(self.beacon_trail)
        self.decoys = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=10, color=theme.rgba(theme.DECOY, 0.0))
        self.decoys.setGLOptions("translucent")
        self.addItem(self.decoys)
        self.beacon_glow = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=34, color=theme.rgba(theme.BEACON, 0.25))
        self.beacon_glow.setGLOptions("additive")
        self.addItem(self.beacon_glow)
        self.beacon = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=14, color=theme.rgba(theme.BEACON))
        self.beacon.setGLOptions("translucent")
        self.addItem(self.beacon)
        self.estimate = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=9, color=theme.rgba(theme.ESTIMATE, 0.0))
        self.estimate.setGLOptions("translucent")
        self.addItem(self.estimate)
        self.beacon_label = gl.GLTextItem(pos=(0, 0, 0), text="B", color=QtGui.QColor(theme.BEACON))
        self.addItem(self.beacon_label)

        # Live disturbance cues (amount follows the current levels):
        #   turbulence    - shimmering air pockets along the line of sight
        #   sensor noise  - sparkles flickering inside the camera footprint
        #   vibration     - the camera head shakes (real jitter, exaggerated)
        self.turb_cells = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0, 0, 0, 0), pxMode=False)
        self.turb_cells.setGLOptions("additive")
        self.addItem(self.turb_cells)
        self.noise_sparks = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0, 0, 0, 0))
        self.noise_sparks.setGLOptions("additive")
        self.addItem(self.noise_sparks)
        self._cue_rng = np.random.default_rng(5)

    # ------------------------------------------------------------------
    def bind_sim(self, sim):
        """Bind to a (new) simulation: redraw everything that depends on it."""
        self.sim = sim
        az_min, az_max, el_min, el_max = sim.scene.bounds
        n = 40
        top = np.c_[np.linspace(az_min, az_max, n), np.full(n, el_max)]
        right = np.c_[np.full(n, az_max), np.linspace(el_max, el_min, n)]
        bottom = np.c_[np.linspace(az_max, az_min, n), np.full(n, el_min)]
        left = np.c_[np.full(n, az_min), np.linspace(el_min, el_max, n)]
        ring = np.vstack([top, right, bottom, left])
        self.patch_outline.setData(pos=to_xyz(ring[:, 0], ring[:, 1], R * 0.999))
        # Translucent surface of the patch.
        ga, ge = np.meshgrid(np.linspace(az_min, az_max, 16), np.linspace(el_min, el_max, 11))
        verts = to_xyz(ga.ravel(), ge.ravel(), R * 1.001)
        faces = []
        for i in range(10):
            for j in range(15):
                k = i * 16 + j
                faces += [[k, k + 1, k + 17], [k, k + 17, k + 16]]
        self.patch_fill.setMeshData(meshdata=gl.MeshData(vertexes=verts, faces=np.array(faces)))
        for item in self.patch_labels:
            self.removeItem(item)
        self.patch_labels = []
        for a in (az_min, 0.0, az_max):
            p = to_xyz(a, el_min - 2.5, R)
            item = gl.GLTextItem(pos=p, text=f"{a:+.0f}°", color=QtGui.QColor(theme.TEXT_DIM))
            self.addItem(item)
            self.patch_labels.append(item)
        rng = np.random.default_rng(99)
        b = (sim.scene.star_peak - sim.scene.star_peak.min()) / max(np.ptp(sim.scene.star_peak), 1)
        spos = to_xyz(sim.scene.star_az, sim.scene.star_el, STAR_R)
        scol = np.c_[STAR_COLORS[rng.choice(4, len(b), p=STAR_P)], 0.30 + 0.60 * b]
        self.stars.setData(pos=spos, size=1.0 + 2.4 * b, color=scol)
        self.trails = {"beacon": deque(maxlen=self.TRAIL), "look": deque(maxlen=self.LOOK_TRAIL)}
        # Cloud puff layouts for this scene (cumulus: puffs spread sideways,
        # a flat-ish base, bigger and brighter puffs on top). Each puff is a
        # core sphere plus a larger, fainter halo for a soft edge.
        rng = np.random.default_rng(12345)
        n_cl = len(sim.disturb.clouds.az) if sim.disturb.clouds is not None else 0
        x = rng.uniform(-0.85, 0.85, (n_cl, PUFFS))
        y = 0.45 * (1 - np.abs(x)) * rng.uniform(0.2, 1.0, (n_cl, PUFFS)) - 0.1
        self.puff_off = np.stack([x, y], -1)
        self.puff_dd = rng.normal(0, 0.03, (n_cl, PUFFS))
        self.puff_scale = (0.30 + 0.30 * (1 - np.abs(x))) * rng.uniform(0.8, 1.15, (n_cl, PUFFS))
        self.puff_shade = 0.80 + 0.20 * (y + 0.1) / 0.55          # darker base, bright tops
        nv = len(self._sph_v)
        n = n_cl * PUFFS * 2
        self._cloud_faces = (self._sph_f[None] + (np.arange(n) * nv)[:, None, None]).reshape(-1, 3)
        self.clouds.setMeshData(meshdata=gl.MeshData())
        self.clouds.setVisible(n_cl > 0)

    def _disturbance_cues(self, res):
        sim, rng = self.sim, self._cue_rng
        paz, pel = res.los
        origin = np.array([0.0, 0.0, MOUNT_H])
        # Turbulence: air pockets between the terminal and the beacon's distance,
        # jittering every update (shimmer); count and size grow with the level.
        s = sim.disturb.strength("turbulence")
        n = int(round(40 * s))
        if n > 0:
            reach = float(depth_r(sim.scene.beacon.distance))
            f = np.sqrt(rng.uniform(0.02, 1.0, n))                        # fraction of the way out
            az = paz + rng.normal(0, 3.0, n)
            el = pel + rng.normal(0, 3.0, n)
            pts = origin + (to_xyz(az, el, reach) - origin) * f[:, None]
            col = np.tile(theme.rgba("#7fc8ff"), (n, 1))
            col[:, 3] = rng.uniform(0.03, 0.09, n) * min(s, 2.0)
            size = reach * f * np.radians(rng.uniform(2.0, 5.0, n))        # world units: ~2-5 deg wide
            self.turb_cells.setData(pos=pts, size=size, color=col)
        else:
            self.turb_cells.setData(pos=np.zeros((1, 3)), color=(0, 0, 0, 0))
        # Sensor noise: random sparkles on the footprint, re-drawn every update.
        s = sim.disturb.strength("sensor")
        n = int(round(30 * s))
        if n > 0:
            cam = sim.camera
            az = paz + rng.uniform(-0.5, 0.5, n) * cam.fov_x
            el = pel + rng.uniform(-0.5, 0.5, n) * cam.fov_y
            col = np.tile((1.0, 1.0, 1.0, 0.0), (n, 1))
            col[:, 3] = rng.uniform(0.25, 0.8, n)
            self.noise_sparks.setData(pos=to_xyz(az, el, R * 0.985), size=rng.uniform(1.5, 3.5, n), color=col)
        else:
            self.noise_sparks.setData(pos=np.zeros((1, 3)), color=(0, 0, 0, 0))

    def update_scene(self, res):
        if self.sim is None:
            return
        sim, cam = self.sim, self.sim.camera
        state = res.track.state.value
        state_col = theme.STATE_COLORS[state]

        # Gimbal: camera head turns with the line of sight. Vibration (the
        # difference between the actual and the commanded direction, a few
        # hundredths of a degree) is exaggerated 60x so the shaking is visible.
        paz, pel = res.los
        shake = (np.subtract(res.los, res.pose) * VIB_EXAGGERATION) if sim.disturb.vibration else (0.0, 0.0)
        m = QtGui.QMatrix4x4()
        m.translate(0, 0, MOUNT_H)
        m.rotate(-(res.pose[0] + shake[0]), 0, 0, 1)
        m.rotate(res.pose[1] + shake[1] + EL0, 1, 0, 0)
        self.head.setTransform(m)
        self.lens.setTransform(m)
        self._disturbance_cues(res)

        # Line of sight, its history and the field-of-view pyramid + footprint.
        origin = np.array([0.0, 0.0, MOUNT_H])
        tip = to_xyz(paz, pel)
        self.los.setData(pos=np.vstack([origin, tip]))
        self.trails["look"].append(to_xyz(paz, pel, R * 0.995))
        if len(self.trails["look"]) > 1:
            self.look_trail.setData(pos=np.array(self.trails["look"]))
        hx, hy = cam.fov_x / 2, cam.fov_y / 2
        corners = to_xyz([paz - hx, paz + hx, paz + hx, paz - hx], [pel + hy, pel + hy, pel - hy, pel - hy], R * 0.99)
        segs = []
        for c in corners:
            segs += [origin, c]
        for i in range(4):
            segs += [corners[i], corners[(i + 1) % 4]]
        self.frustum.setData(pos=np.array(segs), color=theme.rgba(state_col, 0.85))
        self.footprint.setMeshData(meshdata=gl.MeshData(vertexes=corners, faces=np.array([[0, 1, 2], [0, 2, 3]])))
        self.footprint.setColor(theme.rgba(state_col, 0.28))

        # Beacon (size pulses with its blink) and its trail.
        b = sim.scene.beacon
        bpos = to_xyz(b.az, b.el, depth_r(b.distance))
        on = b.current_intensity / max(b.intensity, 1)
        self.beacon.setData(pos=bpos[None], size=8 + 10 * on)
        self.beacon_glow.setData(pos=bpos[None], size=20 + 26 * on, color=theme.rgba(theme.BEACON, 0.10 + 0.25 * on))
        self.beacon_label.setData(pos=bpos + np.array([1.5, 0.0, 2.0]))
        self.trails["beacon"].append(bpos)
        if len(self.trails["beacon"]) > 1:
            pts = np.array(self.trails["beacon"])
            alpha = np.linspace(0.0, 0.9, len(pts))
            col = np.tile(theme.rgba(theme.BEACON), (len(pts), 1))
            col[:, 3] = alpha
            self.beacon_trail.setData(pos=pts, color=col)

        decoys = [t for t in sim.scene.targets if not t.is_beacon]
        if decoys:
            dpos = to_xyz([t.az for t in decoys], [t.el for t in decoys], depth_r([t.distance for t in decoys]))
            on = np.array([t.current_intensity / max(t.intensity, 1) for t in decoys])
            col = np.tile(theme.rgba(theme.DECOY), (len(decoys), 1))
            col[:, 3] = 0.35 + 0.65 * on
            self.decoys.setData(pos=dpos, color=col, size=7 + 6 * on)
        else:
            self.decoys.setData(pos=np.zeros((1, 3)), color=(0, 0, 0, 0))

        # Tracker estimate.
        if res.track.estimate is not None:
            e = res.track.estimate
            self.estimate.setData(pos=to_xyz(e[0], e[1], depth_r(b.distance) * 0.99)[None],
                                  color=theme.rgba(theme.ESTIMATE, 0.95))
        else:
            self.estimate.setData(color=(0, 0, 0, 0))

        # Clouds at their simulated distance, covering their true angular size;
        # optically thicker clouds are drawn more opaque.
        clouds = sim.disturb.clouds
        if clouds is not None and len(self.puff_off):
            n = min(len(clouds.az), len(self.puff_off))
            k = 2.3 * clouds.size[:n, None]
            az = clouds.az[:n, None] + self.puff_off[:n, :, 0] * k
            el = clouds.el[:n, None] + self.puff_off[:n, :, 1] * k
            dist = depth_r(clouds.distance[:n, None] + self.puff_dd[:n])
            pos = to_xyz(az, el, dist).reshape(-1, 3)                       # (n*PUFFS, 3)
            rad = (dist * np.radians(k) * self.puff_scale[:n]).reshape(-1)
            alpha = np.repeat(np.clip(0.055 + 0.022 * clouds.depth[:n], 0.07, 0.15), PUFFS)
            shade = self.puff_shade[:n].reshape(-1)
            # core + halo for every puff
            pos2 = np.concatenate([pos, pos])
            rad2 = np.concatenate([rad, rad * 1.45])
            a2 = np.concatenate([alpha, alpha * 0.4])
            s2 = np.concatenate([shade, shade])
            v = self._sph_v[None] * np.array([1.0, 1.0, 0.8], np.float32) * rad2[:, None, None] + pos2[:, None, :]
            col = np.stack([s2, s2, np.minimum(s2 + 0.05, 1.0), a2], -1)
            vc = np.repeat(col[:, None, :], len(self._sph_v), axis=1)
            md = gl.MeshData(vertexes=v.reshape(-1, 3).astype(np.float32),
                             faces=self._cloud_faces[:len(pos2) * len(self._sph_f)],
                             vertexColors=vc.reshape(-1, 4).astype(np.float32))
            # Normals of a (slightly flattened) sphere: the unit-sphere directions.
            md._vertexNormals = np.tile(self._sph_v, (len(pos2), 1))
            self.clouds.setMeshData(meshdata=md)

        if self.auto_orbit:
            self.orbit(0.25, 0)
