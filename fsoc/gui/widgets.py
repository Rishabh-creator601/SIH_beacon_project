"""Reusable dashboard widgets: panels, stat cards, camera feed, live plots."""

from collections import deque

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme


def panel(title=None):
    """A rounded dark panel with an optional small-caps title. Returns (frame, layout)."""
    frame = QtWidgets.QFrame()
    frame.setObjectName("panel")
    lay = QtWidgets.QVBoxLayout(frame)
    lay.setContentsMargins(12, 10, 12, 12)
    lay.setSpacing(8)
    if title:
        lab = QtWidgets.QLabel(title.upper())
        lab.setObjectName("section")
        lay.addWidget(lab)
    return frame, lay


class StatCard(QtWidgets.QFrame):
    """Label, big value and a small sub-line; value colour can change."""

    def __init__(self, label, value="--", sub=""):
        super().__init__()
        self.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(1)
        self.label = QtWidgets.QLabel(label.upper())
        self.label.setObjectName("cardLabel")
        self.value = QtWidgets.QLabel(value)
        self.value.setObjectName("cardValue")
        self.sub = QtWidgets.QLabel(sub)
        self.sub.setObjectName("cardSub")
        for w in (self.label, self.value, self.sub):
            lay.addWidget(w)

    def set(self, value, sub=None, color=None):
        self.value.setText(value)
        if sub is not None:
            self.sub.setText(sub)
        self.value.setStyleSheet(f"color: {color};" if color else "")


class StateBadge(QtWidgets.QLabel):
    def __init__(self):
        super().__init__("SEARCH")
        self.setObjectName("stateBadge")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.set_state("SEARCH")

    def set_state(self, state):
        col = theme.STATE_COLORS.get(state, theme.TEXT)
        self.setText(state)
        self.setStyleSheet(f"color: {col}; background: {col}22; border: 1px solid {col}88;")


class CameraView(QtWidgets.QLabel):
    """Shows a BGR numpy frame, scaled to fit while keeping its aspect ratio.

    Zoom (1x - 8x, mouse wheel or set_zoom) crops a window around ``center``
    - normally the tracked beacon - and shows it with crisp pixels, so the
    blink, noise and detection boxes of a few-pixel target become visible.
    """

    zoomChanged = QtCore.Signal(int)
    LEVELS = (1, 2, 4, 8)

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet(f"background: #05080c; border-radius: 6px;")
        self.setToolTip("Mouse wheel: zoom (centred on the tracked beacon)")
        self._pix = None
        self.zoom = 1
        self._last = None

    def set_zoom(self, level):
        self.zoom = min(max(int(level), 1), self.LEVELS[-1])
        self.zoomChanged.emit(self.zoom)
        if self._last is not None:
            self.show_frame(*self._last)

    def wheelEvent(self, e):
        i = self.LEVELS.index(self.zoom) if self.zoom in self.LEVELS else 0
        i = min(i + 1, len(self.LEVELS) - 1) if e.angleDelta().y() > 0 else max(i - 1, 0)
        self.set_zoom(self.LEVELS[i])

    def show_frame(self, bgr, center=None):
        self._last = (bgr, center)
        smooth = True
        if self.zoom > 1:
            h, w = bgr.shape[:2]
            cw, ch = w // self.zoom, h // self.zoom
            cx, cy = center if center is not None else (w / 2, h / 2)
            x0 = int(min(max(cx - cw / 2, 0), w - cw))
            y0 = int(min(max(cy - ch / 2, 0), h - ch))
            bgr = cv2.resize(bgr[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.putText(bgr, f"ZOOM {self.zoom}x", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 198, 56), 1, cv2.LINE_AA)
            smooth = False
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._pix = QtGui.QPixmap.fromImage(img)
        self._smooth = smooth
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            mode = QtCore.Qt.SmoothTransformation if getattr(self, "_smooth", True) else QtCore.Qt.FastTransformation
            self.setPixmap(self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, mode))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()


class LivePlots(QtWidgets.QWidget):
    """Rolling plots: pointing error and the beacon score of the followed candidate."""

    WINDOW_S = 30.0

    def __init__(self):
        super().__init__()
        pg.setConfigOptions(antialias=True, background=theme.PANEL, foreground=theme.TEXT_DIM)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.err_plot = self._plot("Pointing error", "px")
        self.err_curve = self.err_plot.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self.lock_curve = self.err_plot.plot(pen=None, fillLevel=0, brush=pg.mkBrush(53, 208, 127, 40))
        self.handover = pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.GOOD, width=1, style=QtCore.Qt.DashLine))
        self.err_plot.addItem(self.handover)
        self.err_plot.setLogMode(y=True)
        lay.addWidget(self.err_plot)

        self.score_plot = self._plot("Beacon score of followed candidate", "P(beacon)")
        self.score_plot.setYRange(0, 1.02)
        self.score_curve = self.score_plot.plot(pen=pg.mkPen(theme.GOOD, width=2))
        self.accept = pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.GOOD, width=1, style=QtCore.Qt.DashLine))
        self.reject = pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.BAD, width=1, style=QtCore.Qt.DashLine))
        self.score_plot.addItem(self.accept)
        self.score_plot.addItem(self.reject)
        lay.addWidget(self.score_plot)
        self.clear()

    def _plot(self, title, units):
        w = pg.PlotWidget()
        w.setTitle(f"<span style='color:{theme.TEXT_DIM};font-size:9pt'>{title}</span>")
        w.showGrid(x=True, y=True, alpha=0.15)
        w.setLabel("left", units)
        w.setLabel("bottom", "time", "s")
        w.setMouseEnabled(x=False, y=False)
        w.hideButtons()
        w.getPlotItem().getViewBox().setBackgroundColor(theme.PANEL)
        return w

    def clear(self):
        n = int(self.WINDOW_S * 30)
        self.t, self.err, self.lock, self.score = (deque(maxlen=n) for _ in range(4))

    def set_thresholds(self, handover_px, accept, reject):
        self.handover.setValue(np.log10(max(handover_px, 1e-3)))
        self.accept.setValue(accept)
        self.reject.setValue(reject)

    def add(self, t, err_px, locked, score):
        self.t.append(t)
        self.err.append(max(err_px, 0.1))
        self.lock.append(max(err_px, 0.1) if locked else np.nan)
        self.score.append(np.nan if score is None else score)

    def refresh(self):
        if not self.t:
            return
        t = np.array(self.t)
        self.err_curve.setData(t, np.array(self.err))
        self.lock_curve.setData(t, np.array(self.lock), connect="finite")
        self.score_curve.setData(t, np.array(self.score), connect="finite")
        for p in (self.err_plot, self.score_plot):
            p.setXRange(max(t[-1] - self.WINDOW_S, 0), max(t[-1], self.WINDOW_S), padding=0)


class Legend(QtWidgets.QFrame):
    """Framed key: one row per symbol - swatch, name and a short explanation."""

    def __init__(self, items, title="LEGEND"):
        super().__init__()
        self.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(5)
        head = QtWidgets.QLabel(title)
        head.setObjectName("section")
        lay.addWidget(head)
        for symbol, color, name, note in items:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(8)
            sw = QtWidgets.QLabel(symbol)
            sw.setFixedWidth(22)
            sw.setAlignment(QtCore.Qt.AlignCenter)
            sw.setStyleSheet(f"color: {color}; font-size: 13pt; font-weight: 700;")
            txt = QtWidgets.QLabel(f"<b>{name}</b><br><span style='color:{theme.TEXT_DIM};"
                                   f"font-size:8pt'>{note}</span>")
            txt.setWordWrap(True)
            row.addWidget(sw, 0, QtCore.Qt.AlignTop)
            row.addWidget(txt, 1)
            lay.addLayout(row)
        lay.addStretch(1)


class LabeledSlider(QtWidgets.QWidget):
    """Slider with name and live value; emits valueChanged(float)."""

    valueChanged = QtCore.Signal(float)

    def __init__(self, name, lo, hi, value, step=0.05, tooltip=""):
        super().__init__()
        self.scale = 1 / step
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setVerticalSpacing(2)
        self.name = QtWidgets.QLabel(name)
        self.val = QtWidgets.QLabel()
        self.val.setAlignment(QtCore.Qt.AlignRight)
        self.val.setStyleSheet(f"color: {theme.ACCENT}; font-family: Consolas;")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(int(lo * self.scale), int(hi * self.scale))
        self.slider.valueChanged.connect(self._changed)
        lay.addWidget(self.name, 0, 0)
        lay.addWidget(self.val, 0, 1)
        lay.addWidget(self.slider, 1, 0, 1, 2)
        if tooltip:
            self.setToolTip(tooltip)
        self.set_value(value, emit=False)

    def _changed(self, v):
        self.val.setText(f"{v / self.scale:.2f}")
        self.valueChanged.emit(v / self.scale)

    def set_value(self, v, emit=True):
        self.slider.blockSignals(not emit)
        self.slider.setValue(int(round(v * self.scale)))
        self.val.setText(f"{v:.2f}")
        self.slider.blockSignals(False)

    def value(self):
        return self.slider.value() / self.scale
