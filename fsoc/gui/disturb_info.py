"""Disturbance explainer: what each disturbance is and what it does to the
camera image, illustrated with pictures generated at the current levels.

The pictures use the same parameters as the simulator (config/default.yaml,
scaled by the slider level) on a small patch around the beacon, magnified
so single pixels are visible.
"""

import math

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme

PATCH = 40            # patch size (camera pixels)
ZOOM = 3              # display magnification of the patches
BG = 18.0             # sky background level
PEAK, SIGMA = 220.0, 2.0
PPD = 80.0            # camera pixels per degree


def _spot(cx=PATCH / 2, cy=PATCH / 2, peak=PEAK, sigma=SIGMA):
    y, x = np.mgrid[0:PATCH, 0:PATCH].astype(np.float32)
    return BG + peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))


def _to_pixmap(img, label=None):
    img = np.clip(img, 0, 255).astype(np.uint8)
    big = cv2.resize(img, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    if label:
        cv2.putText(big, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 255), 1, cv2.LINE_AA)
    return big


def _strip(tiles):
    sep = np.full((tiles[0].shape[0], 4, 3), 40, np.uint8)
    out = []
    for i, t in enumerate(tiles):
        out += [t] if i == 0 else [sep, t]
    return np.hstack(out)


def _qpix(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy())


# ---------------------------------------------------------------- pictures
def turbulence_strip(s, rng):
    """Clean beacon, then 5 frames 0.1 s apart: twinkle, wander and blur."""
    tiles = [_to_pixmap(_spot(), "clean")]
    sci, wander, blur = 0.3 * s, 0.8 * s, 0.6 * s
    for k in range(5):
        chi = rng.normal(0, sci) if sci > 0 else 0.0
        gain = math.exp(chi - sci * sci / 2)
        dx, dy = rng.normal(0, wander, 2) if wander > 0 else (0.0, 0.0)
        sig = math.hypot(SIGMA, blur)
        tiles.append(_to_pixmap(_spot(PATCH / 2 + dx, PATCH / 2 + dy, PEAK * gain * (SIGMA / sig) ** 2, sig),
                                f"x{gain:.2f}"))
    return _strip(tiles)


def vibration_trace(s, rng):
    """Line-of-sight jitter over 1 s (7 Hz + 23 Hz tones + random jitter), in pixels, magnified."""
    t = np.arange(0, 1.0, 0.001)
    az = 0.010 * s * np.sin(2 * np.pi * 7 * t) + 0.005 * s * np.sin(2 * np.pi * 23 * t + 0.7)
    el = 0.010 * s * np.sin(2 * np.pi * 7 * t + 1.3) + 0.005 * s * np.sin(2 * np.pi * 23 * t + 2.1)
    a = math.exp(-0.001 / 0.03)
    j = np.zeros((len(t), 2))
    for i in range(1, len(t)):
        j[i] = a * j[i - 1] + math.sqrt(1 - a * a) * 0.006 * s * rng.standard_normal(2)
    x, y = (az + j[:, 0]) * PPD, (el + j[:, 1]) * PPD          # pixels
    size = PATCH * ZOOM
    img = np.full((size, size, 3), 12, np.uint8)
    c = size // 2
    reach = max(float(np.abs(np.r_[x, y]).max()), 0.5)
    mag = (c - 10) / reach                                      # fit the trace in the box
    for r in range(1, int(reach) + 1):                          # rings every pixel
        cv2.circle(img, (c, c), int(r * mag), (60, 60, 60), 1, cv2.LINE_AA)
    pts = np.c_[c + x * mag, c - y * mag].astype(np.int32)
    cv2.polylines(img, [pts], False, (255, 180, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "ring = 1 px", (4, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1,
                cv2.LINE_AA)
    rms = float(np.sqrt(np.mean(x ** 2 + y ** 2)))
    return img, rms, mag


def sensor_strip(s, rng):
    """Clean dim beacon (blink 'off' phase) vs the same with sensor noise."""
    dim = _spot(peak=PEAK * 0.3)
    noisy = dim + rng.standard_normal(dim.shape) * np.sqrt(np.maximum(dim, 0) * (0.5 * s) ** 2 + (3.0 * s) ** 2)
    n_hits = rng.poisson(max(4 * s, 0) * PATCH * PATCH / (640 * 480) * 60)   # exaggerated for visibility
    for _ in range(n_hits):
        noisy[rng.integers(0, PATCH), rng.integers(0, PATCH)] = 255
    if s > 0:
        noisy[5, 31] += 120                                      # a hot pixel
    return _strip([_to_pixmap(_spot(), "beacon on"), _to_pixmap(dim, "off, clean"), _to_pixmap(noisy, "off, noisy")])


def cloud_strip():
    """A cloud drifting across the beacon: transmission exp(-depth)."""
    tiles = []
    for T in (1.0, 0.55, 0.15, 0.55, 1.0):
        img = (_spot() - 35.0) * T + 35.0                         # cloud glow 35
        tiles.append(_to_pixmap(img, f"T={T:.2f}"))
    return _strip(tiles)


# ---------------------------------------------------------------- dialog
class DisturbanceInfo(QtWidgets.QDialog):
    """Non-modal window; call ``refresh(levels)`` before showing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Disturbances - what they are and what they do")
        self.setStyleSheet(f"background: {theme.BG}; color: {theme.TEXT};")
        # Content in a scroll area so the window fits small (e.g. 1366 x 768) screens.
        top = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        top.addWidget(scroll, 1)
        outer = QtWidgets.QVBoxLayout(body)
        intro = QtWidgets.QLabel(
            "Each disturbance has a <b>level</b>: 0 = off, 1 = nominal, 2 = twice as strong, 3 = extreme. "
            "Pictures below are generated at the <b>current slider levels</b> (magnified patch of the camera "
            "image around the beacon, 1 square = 1 camera pixel = 0.22 mrad).")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        self.setMinimumWidth(780)
        self.rows = {}
        for key in ("turbulence", "vibration", "sensor", "clouds"):
            # Description on top (full width, wraps freely), picture below it.
            pic, txt = QtWidgets.QLabel(), QtWidgets.QLabel()
            txt.setWordWrap(True)
            txt.setTextFormat(QtCore.Qt.RichText)
            outer.addSpacing(6)
            outer.addWidget(txt)
            outer.addWidget(pic)
            self.rows[key] = (pic, txt)
        outer.addStretch(1)
        close = QtWidgets.QPushButton("Close  (Esc)")
        close.clicked.connect(self.close)
        top.addWidget(close, 0, QtCore.Qt.AlignRight)

    def refresh(self, levels):
        rng = np.random.default_rng(7)
        t, v, n = levels["turbulence"], levels["vibration"], levels["sensor"]
        head = lambda name, lvl: (f"<span style='color:{theme.ACCENT};font-size:11pt'><b>{name}</b></span>"
                                  + ("" if lvl is None else f" &nbsp; level <b>{lvl:.2f}</b>") + "<br>")

        pic, txt = self.rows["turbulence"]
        pic.setPixmap(_qpix(turbulence_strip(t, rng)))
        txt.setText(head("Atmospheric turbulence", t) +
                    "<i>Cause:</i> pockets of warmer and cooler air bend the light on its way to the camera.<br>"
                    f"<i>Effect:</i> the beacon <b>twinkles</b> (brightness varies by about ±{100 * 0.3 * t:.0f}%), "
                    f"<b>wanders</b> (±{0.8 * t:.1f} px) and gets <b>blurred</b>; the whole image shimmers. "
                    "Deep fades can make the beacon disappear for a few frames. (The number on each frame is its "
                    "brightness factor; frames are 0.1 s apart.)")

        pic, txt = self.rows["vibration"]
        img, rms, mag = vibration_trace(v, rng)
        pic.setPixmap(_qpix(img))
        txt.setText(head("Platform vibration", v) +
                    "<i>Cause:</i> engines, rotors and the structure shake the camera mount (7 Hz and 23 Hz "
                    "modes plus random jitter).<br>"
                    f"<i>Effect:</i> the line of sight jitters by about <b>{rms:.1f} px RMS</b> "
                    f"({rms * 0.218:.2f} mrad); the trace shows 1 s of pointing jitter, enlarged so that each ring is 1 camera pixel. The "
                    "tracker sees it as fake target motion, and the image gets motion-blurred.")

        pic, txt = self.rows["sensor"]
        pic.setPixmap(_qpix(sensor_strip(n, rng)))
        txt.setText(head("Sensor noise", n) +
                    "<i>Cause:</i> the camera electronics: photon (shot) noise, read noise, hot pixels and "
                    "random pixel hits (e.g. cosmic rays).<br>"
                    f"<i>Effect:</i> grainy background (read noise {3.0 * n:.1f} grey levels). Hardest when the "
                    "beacon is in its dim 'off' blink phase: the spot can sink into the noise, and single bright "
                    "pixels can look like tiny targets.")

        pic, txt = self.rows["clouds"]
        pic.setPixmap(_qpix(cloud_strip()))
        txt.setText(head("Clouds  (part of the scene, not a slider)", None) +
                    "<i>Cause:</i> drifting cloud banks between the camera and the beacon.<br>"
                    "<i>Effect:</i> the beacon is dimmed (transmission T) or hidden for seconds. The tracker then "
                    "<b>coasts</b> on its prediction and searches near where the beacon should reappear.")
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        body = self.findChild(QtWidgets.QScrollArea).widget()
        body.adjustSize()
        self.resize(max(self.minimumWidth(), body.sizeHint().width() + 40),
                    min(body.sizeHint().height() + 70, screen.height() - 60))
