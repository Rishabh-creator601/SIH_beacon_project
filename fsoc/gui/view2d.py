"""2D situational view: the simulated sky as a flat azimuth / elevation map.

Same content as the 3D view, seen face-on, which makes distances and the
search pattern easy to read:
  * the simulated sky patch, background stars and clouds
  * where the camera has looked (search spiral / raster) and its current
    field of view, coloured by tracker state
  * the beacon ("B", pulsing with its blink) and its trail, decoys and the
    tracker's estimate
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from . import theme

PUFFS = 8                               # puffs per cloud (as in the 3D view)
STAR_P = [0.25, 0.40, 0.22, 0.13]       # star colour mix: blue-white, white, yellow, orange


def _pen(color, alpha=1.0, width=1.0, style=QtCore.Qt.SolidLine):
    c = QtGui.QColor(color)
    c.setAlphaF(alpha)
    return pg.mkPen(c, width=width, style=style)


def _brush(color, alpha=1.0):
    c = QtGui.QColor(color)
    c.setAlphaF(alpha)
    return pg.mkBrush(c)


class SkyMap2D(pg.PlotWidget):
    TRAIL = 240          # beacon trail length (updates), as in the 3D view
    LOOK_TRAIL = 600     # line-of-sight history length

    def __init__(self, parent=None):
        super().__init__(parent, background=theme.BG)
        self.sim = None
        p = self.getPlotItem()
        p.showGrid(x=True, y=True, alpha=0.12)
        p.setLabel("bottom", "azimuth", "°")
        p.setLabel("left", "elevation", "°")
        p.setAspectLocked(True)
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons()
        p.setMenuEnabled(False)

        self.patch = pg.PlotCurveItem(pen=_pen(theme.ACCENT, 0.6, 1.5), fillLevel=None)
        self.stars = pg.ScatterPlotItem(pen=None, size=2)
        # Clouds: soft puffs (several overlapping translucent discs per cloud).
        # Drawn above the targets behind them and below the targets in front.
        self.clouds = pg.ScatterPlotItem(pen=None, pxMode=False)
        self.clouds.setZValue(5)
        self.look = pg.PlotCurveItem(pen=_pen(theme.ACCENT, 0.3, 1.0))
        self.fov = pg.PlotCurveItem(pen=_pen(theme.GOOD, 0.9, 1.6))
        self.los = pg.ScatterPlotItem(symbol="+", size=11, pen=_pen(theme.ACCENT, 0.9, 1.4), brush=None)
        self.trail = pg.PlotCurveItem(pen=_pen(theme.BEACON, 0.55, 1.6))
        self.decoys = pg.ScatterPlotItem(pen=None)
        self.beacon = pg.ScatterPlotItem(pen=_pen("#ffffff", 0.8, 1.0))
        self.estimate = pg.ScatterPlotItem(symbol="x", size=10, pen=_pen(theme.ESTIMATE, 1.0, 2.0), brush=None)
        self.label = pg.TextItem("B", color=theme.BEACON, anchor=(-0.3, 1.1))
        f = QtGui.QFont()
        f.setBold(True)
        self.label.setFont(f)
        for item in (self.patch, self.stars, self.clouds, self.look, self.fov, self.los, self.trail,
                     self.decoys, self.beacon, self.estimate, self.label):
            self.addItem(item)
        for item in (self.fov, self.los, self.estimate, self.label):
            item.setZValue(8)

    # ------------------------------------------------------------------
    def bind_sim(self, sim):
        """Bind to a (new) simulation: redraw everything that depends on it."""
        self.sim = sim
        a0, a1, e0, e1 = sim.scene.bounds
        self.patch.setData([a0, a1, a1, a0, a0], [e0, e0, e1, e1, e0])
        # Stars coloured by temperature (as in the 3D view), brighter = larger.
        rng = np.random.default_rng(99)
        b = (sim.scene.star_peak - sim.scene.star_peak.min()) / max(np.ptp(sim.scene.star_peak), 1)
        tint = np.array(["#b3ccff", "#ffffff", "#fff2cc", "#ffcc99"])[rng.choice(4, len(b), p=STAR_P)]
        self.stars.setData(sim.scene.star_az, sim.scene.star_el, size=1.0 + 2.2 * b,
                           brush=[_brush(c, 0.25 + 0.55 * v) for c, v in zip(tint, b)])
        # Puff layout per cloud (offsets and sizes in units of the cloud size).
        n_cl = len(sim.disturb.clouds.az) if sim.disturb.clouds is not None else 0
        x = rng.uniform(-0.85, 0.85, (n_cl, PUFFS))
        y = 0.45 * (1 - np.abs(x)) * rng.uniform(0.2, 1.0, (n_cl, PUFFS)) - 0.1
        self.puff = np.stack([x, y], -1)
        self.puff_size = (0.30 + 0.30 * (1 - np.abs(x))) * rng.uniform(0.8, 1.15, (n_cl, PUFFS))
        m = 2.0
        self.setRange(xRange=(a0 - m, a1 + m), yRange=(e0 - m, e1 + m), padding=0)
        self.trails = {"beacon": deque(maxlen=self.TRAIL), "look": deque(maxlen=self.LOOK_TRAIL)}
        self.clear_dynamic()

    def clear_dynamic(self):
        for item in (self.look, self.fov, self.trail):
            item.setData([], [])
        for item in (self.clouds, self.decoys, self.beacon, self.estimate, self.los):
            item.setData([], [])

    def update_scene(self, res, draw=True):
        """Record trails every frame; redraw only when ``draw`` (view visible)."""
        if self.sim is None:
            return
        sim = self.sim
        b = sim.scene.beacon
        self.trails["beacon"].append((b.az, b.el))
        self.trails["look"].append(res.los)
        if not draw:
            return

        paz, pel = res.los
        cam = sim.camera
        hx, hy = cam.fov_x / 2, cam.fov_y / 2
        col = theme.STATE_COLORS[res.track.state.value]
        self.fov.setData([paz - hx, paz + hx, paz + hx, paz - hx, paz - hx],
                         [pel + hy, pel + hy, pel - hy, pel - hy, pel + hy])
        self.fov.setPen(_pen(col, 0.95, 1.8))
        self.los.setData([paz], [pel])
        look = np.array(self.trails["look"])
        self.look.setData(look[:, 0], look[:, 1])

        tr = np.array(self.trails["beacon"])
        self.trail.setData(tr[:, 0], tr[:, 1])
        on = b.current_intensity / max(b.intensity, 1)
        self.beacon.setData([b.az], [b.el], size=7 + 6 * on, brush=_brush(theme.BEACON, 0.5 + 0.5 * on))
        self.label.setPos(b.az, b.el)
        # Depth: behind a cloud that covers it, the beacon is drawn under the
        # (translucent) cloud; otherwise on top of it.
        behind = sim.disturb.cloud_transmission(b) < 0.97
        for item in (self.beacon, self.trail):
            item.setZValue(2 if behind else 7)

        decoys = [t for t in sim.scene.targets if not t.is_beacon]
        self.decoys.setData([t.az for t in decoys], [t.el for t in decoys],
                            size=[5 + 4 * t.current_intensity / max(t.intensity, 1) for t in decoys],
                            brush=[_brush(theme.DECOY, 0.4 + 0.6 * t.current_intensity / max(t.intensity, 1))
                                   for t in decoys])

        e = res.track.estimate
        self.estimate.setData([] if e is None else [e[0]], [] if e is None else [e[1]])

        clouds = sim.disturb.clouds
        if clouds is not None and len(self.puff):
            n = min(len(clouds.az), len(self.puff))
            k = 2.3 * clouds.size[:n, None]
            az = (clouds.az[:n, None] + self.puff[:n, :, 0] * k).ravel()
            el = (clouds.el[:n, None] + self.puff[:n, :, 1] * k).ravel()
            size = (2 * k * self.puff_size[:n]).ravel()
            alpha = np.repeat(np.clip(0.05 + 0.02 * clouds.depth[:n], 0.06, 0.13), PUFFS)
            # core + soft halo per puff
            self.clouds.setData(np.r_[az, az], np.r_[el, el], size=np.r_[size, size * 1.45],
                                brush=[_brush("#dfe7f5", a) for a in np.r_[alpha, alpha * 0.45]])
        else:
            self.clouds.setData([], [])
