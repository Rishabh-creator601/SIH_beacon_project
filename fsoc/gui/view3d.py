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
from PySide6 import QtGui

from . import theme

R = 100.0          # sky dome radius (scene units)
EL0 = 35.0         # elevation of the simulated sky patch's centre
MOUNT_H = 6.0      # height of the camera head above ground


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
        self.stars = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=2.0, color=(0.8, 0.85, 1.0, 0.35))
        self.stars.setGLOptions("translucent")
        self.addItem(self.stars)
        self.clouds = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=1.0, pxMode=False,
                                           color=(0.75, 0.8, 0.9, 0.0))
        self.clouds.setGLOptions("translucent")
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
        self.stars.setData(pos=to_xyz(sim.scene.star_az, sim.scene.star_el, R * 1.003),
                           size=1.5 + 3.0 * (sim.scene.star_peak / max(sim.scene.star_peak.max(), 1)))
        self.trails = {"beacon": deque(maxlen=self.TRAIL), "look": deque(maxlen=self.LOOK_TRAIL)}

    def update_scene(self, res):
        if self.sim is None:
            return
        sim, cam = self.sim, self.sim.camera
        state = res.track.state.value
        state_col = theme.STATE_COLORS[state]

        # Gimbal: camera head turns with the line of sight.
        paz, pel = res.los
        m = QtGui.QMatrix4x4()
        m.translate(0, 0, MOUNT_H)
        m.rotate(-paz, 0, 0, 1)
        m.rotate(pel + EL0, 1, 0, 0)
        self.head.setTransform(m)
        self.lens.setTransform(m)

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
        bpos = to_xyz(b.az, b.el, R * 0.985)
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
            dpos = to_xyz([t.az for t in decoys], [t.el for t in decoys], R * 0.985)
            on = np.array([t.current_intensity / max(t.intensity, 1) for t in decoys])
            col = np.tile(theme.rgba(theme.DECOY), (len(decoys), 1))
            col[:, 3] = 0.35 + 0.65 * on
            self.decoys.setData(pos=dpos, color=col, size=7 + 6 * on)
        else:
            self.decoys.setData(pos=np.zeros((1, 3)), color=(0, 0, 0, 0))

        # Tracker estimate.
        if res.track.estimate is not None:
            e = res.track.estimate
            self.estimate.setData(pos=to_xyz(e[0], e[1], R * 0.98)[None], color=theme.rgba(theme.ESTIMATE, 0.95))
        else:
            self.estimate.setData(color=(0, 0, 0, 0))

        # Clouds: soft translucent blobs sized to their angular size.
        clouds = sim.disturb.clouds
        if clouds is not None:
            cpos = to_xyz(clouds.az, clouds.el, R * 0.97)
            size = 2 * R * np.radians(clouds.size) * 1.6
            alpha = np.clip(0.08 + 0.05 * clouds.depth, 0.08, 0.35)
            col = np.tile((0.78, 0.84, 0.95, 0.0), (len(cpos), 1))
            col[:, 3] = alpha
            self.clouds.setData(pos=cpos, size=size, color=col)
        else:
            self.clouds.setData(pos=np.zeros((1, 3)), color=(0, 0, 0, 0))

        if self.auto_orbit:
            self.orbit(0.25, 0)
