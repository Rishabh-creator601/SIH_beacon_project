"""Main window of the FSOC tracking simulator.

Layout
------
  header   : title, tracker state badge, run controls
  left     : scenario / identification / speed, live disturbance sliders,
             display options, report export
  centre   : camera feed (with detection overlays) | 3D situational view
             live plots underneath
  right    : performance cards and the current scene description

The simulation runs in the GUI thread, paced by a QTimer: each tick advances
the simulation by as many frames as real time (x speed) requires, then
redraws. Views are refreshed at different rates to keep the loop light.
"""

import os
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..config import PROJECT_ROOT, load_config
from ..simulation import Simulation
from ..visualization import Overlay
from . import theme
from .disturb_info import DisturbanceInfo
from .view2d import SkyMap2D
from .view3d import SceneView3D
from .widgets import CameraView, LabeledSlider, Legend, LivePlots, StatCard, StateBadge, panel

SCENARIOS = [("Random scene", "random"), ("Default (fixed)", "default")] + [
    (p.stem.replace("_", " ").title(), str(p)) for p in sorted((PROJECT_ROOT / "scenarios").glob("*.yaml"))]
SPEEDS = [("0.5x", 0.5), ("1x  real time", 1.0), ("2x", 2.0), ("4x", 4.0), ("Max", 0.0)]
METHODS = [("Hybrid  (CNN + blink)", "hybrid"), ("CNN only", "cnn"), ("Blink test only", "blink"),
           ("None  (brightest dot)", "none")]
# Simulated sky: (label, (az_min, az_max, el_min, el_max)) in degrees.
SKY_AREAS = [("Standard  (az ±30°, el ±20°)", (-30.0, 30.0, -20.0, 20.0)),
             ("Wide  (az ±60°, el ±30°)", (-60.0, 60.0, -30.0, 30.0))]
VIEW_MODES = ("3D", "2D", "Both")
DISTURBANCES = ("turbulence", "vibration", "sensor")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSOC Coarse PAT Simulator")
        # Fit the screen: size everything from the usable screen area (taskbar
        # excluded) so the window works from 1366x768 laptops to large monitors.
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.compact = screen.width() < 1600 or screen.height() < 900
        self.side_w = 262 if self.compact else 290
        self.stats_w = 230 if self.compact else 270
        self.setMinimumSize(min(1100, screen.width()), min(640, screen.height()))
        self.setGeometry(screen)
        self.sim = None
        self.res = None
        self.running = False
        self.speed = 1.0
        self._tick_n = 0
        self._seed = None             # seed of the current scene (kept for restarts, not shown)
        self._dist_override = None    # disturbance levels the user applied, kept across restarts
        self._build_ui()
        self._shortcuts()
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.new_run(new_seed=True)
        self.setFocus()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 8)
        outer.setSpacing(10)
        outer.addLayout(self._header())

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._controls(), 0)
        body.addLayout(self._center(), 1)
        body.addWidget(self._stats(), 0)
        outer.addLayout(body, 1)
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    def _header(self):
        h = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        t = QtWidgets.QLabel("FSOC  ·  Coarse Pointing, Acquisition & Tracking")
        t.setObjectName("title")
        s = QtWidgets.QLabel("Virtual pan-tilt camera acquiring and tracking a blinking optical beacon "
                             "through turbulence, vibration, clouds and decoys")
        s.setObjectName("subtitle")
        titles.addWidget(t)
        titles.addWidget(s)
        h.addLayout(titles, 1)
        self.badge = StateBadge()
        self.badge.setMinimumWidth(170)
        h.addWidget(self.badge)
        h.addSpacing(16)
        self.btn_run = QtWidgets.QPushButton("▶  Start")
        self.btn_run.setObjectName("primary")
        self.btn_run.setMinimumWidth(110)
        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_restart = QtWidgets.QPushButton("⟲  Restart")
        self.btn_restart.setToolTip("Restart the same scene from the beginning  [R]")
        self.btn_restart.clicked.connect(lambda: self.new_run(new_seed=False))
        self.btn_new = QtWidgets.QPushButton("🎲  New random scene")
        self.btn_new.setToolTip("New random beacon path, decoys and turbulence  [N]")
        self.btn_new.clicked.connect(lambda: self.new_run(new_seed=True))
        for b in (self.btn_run, self.btn_restart, self.btn_new):
            h.addWidget(b)
        return h

    def _controls(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(self.side_w)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(10)

        f, l = panel("Scenario")
        self.cb_scenario = QtWidgets.QComboBox()
        for name, _ in SCENARIOS:
            self.cb_scenario.addItem(name)
        self.cb_scenario.setToolTip("Random scene: new beacon path, decoy count and turbulence each run.\n"
                                    "The numbered scenarios are fixed, reproducible test cases.")
        l.addWidget(self.cb_scenario)
        # A different scenario brings its own disturbance levels.
        self.cb_scenario.currentIndexChanged.connect(lambda _: setattr(self, "_dist_override", None))
        l.addWidget(QtWidgets.QLabel("Sky area"))
        self.cb_sky = QtWidgets.QComboBox()
        for name, _ in SKY_AREAS:
            self.cb_sky.addItem(name)
        self.cb_sky.setToolTip("Size of the simulated sky the beacon and decoys move in.\n"
                               "Applies on restart / new scene.")
        l.addWidget(self.cb_sky)
        l.addWidget(QtWidgets.QLabel("Beacon identification"))
        self.cb_method = QtWidgets.QComboBox()
        for name, _ in METHODS:
            self.cb_method.addItem(name)
        l.addWidget(self.cb_method)
        # Decoy count range (random scenes): each new scene draws a count in [from, to].
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Decoys"))
        self.sp_dec_lo, self.sp_dec_hi = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for i, (sp, v) in enumerate(((self.sp_dec_lo, 2), (self.sp_dec_hi, 5))):
            sp.setRange(2, 8)
            sp.setValue(v)
            sp.setToolTip("Number of decoy lights in random scenes: a random count in this range\n"
                          "(set both equal for a fixed count; at least 2). Each decoy gets a random path and\n"
                          "is steady or blinks at a wrong rate. Applies on restart / new scene.")
            if i:
                row.addWidget(QtWidgets.QLabel("to"))
            row.addWidget(sp, 1)
        self.sp_dec_lo.valueChanged.connect(lambda v: self.sp_dec_hi.setValue(max(v, self.sp_dec_hi.value())))
        self.sp_dec_hi.valueChanged.connect(lambda v: self.sp_dec_lo.setValue(min(v, self.sp_dec_lo.value())))
        l.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Run length"))
        self.sp_duration = QtWidgets.QSpinBox()
        self.sp_duration.setRange(0, 3600)
        self.sp_duration.setValue(120)
        self.sp_duration.setSuffix(" s")
        self.sp_duration.setSpecialValueText("endless")
        row.addWidget(self.sp_duration)
        l.addLayout(row)
        apply = QtWidgets.QPushButton("Apply && restart")
        apply.clicked.connect(lambda: self.new_run(new_seed=False))
        l.addWidget(apply)
        lay.addWidget(f)

        f, l = panel("Simulation speed")
        self.cb_speed = QtWidgets.QComboBox()
        for name, _ in SPEEDS:
            self.cb_speed.addItem(name)
        self.cb_speed.setCurrentIndex(1)
        self.cb_speed.currentIndexChanged.connect(lambda i: setattr(self, "speed", SPEEDS[i][1]))
        l.addWidget(self.cb_speed)
        lay.addWidget(f)

        f, l = panel()
        head = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel("DISTURBANCES")
        lab.setObjectName("section")
        head.addWidget(lab)
        head.addStretch(1)
        b = QtWidgets.QPushButton("ⓘ")
        b.setToolTip("What each disturbance is and what it does, pictured at the current levels  [F2]")
        b.setStyleSheet("padding: 0px 7px; font-size: 10pt; font-weight: 700;")
        b.clicked.connect(self._show_disturbance_info)
        head.addWidget(b)
        l.addLayout(head)
        self.sl_turb = LabeledSlider("Atmospheric turbulence", 0, 3, 1.0,
                                     tooltip="Scintillation (twinkle), beam wander, blur and heat shimmer")
        self.sl_vib = LabeledSlider("Platform vibration", 0, 3, 1.0, tooltip="Line-of-sight jitter of the gimbal")
        self.sl_noise = LabeledSlider("Sensor noise", 0, 3, 1.0, tooltip="Shot / read noise and pixel hits")
        self.dist_sliders = dict(zip(DISTURBANCES, (self.sl_turb, self.sl_vib, self.sl_noise)))
        for s in self.dist_sliders.values():
            s.valueChanged.connect(lambda _v: self._set_pending(True))
            l.addWidget(s)
        self.lbl_pending = QtWidgets.QLabel(f"<span style='color:{theme.WARN}'>●</span> changed - not applied yet")
        self.lbl_pending.setObjectName("subtitle")
        self.lbl_pending.setVisible(False)
        l.addWidget(self.lbl_pending)
        b = QtWidgets.QPushButton("▶  Apply && continue")
        b.setToolTip("Apply the new levels now; the run continues from the current moment  [C]")
        b.clicked.connect(lambda: self.apply_disturbances(restart=False))
        l.addWidget(b)
        b = QtWidgets.QPushButton("⟲  Apply && restart")
        b.setToolTip("Restart the same scene from the beginning with the new levels  [Shift+R]")
        b.clicked.connect(lambda: self.apply_disturbances(restart=True))
        l.addWidget(b)
        lay.addWidget(f)

        f, l = panel("Display")
        self.ck_scores = QtWidgets.QCheckBox("Show beacon scores")
        self.ck_scores.setChecked(True)
        self.ck_roi = QtWidgets.QCheckBox("Show tracking window (ROI)")
        self.ck_roi.setChecked(True)
        self.ck_orbit = QtWidgets.QCheckBox("Auto-rotate 3D view")
        for c in (self.ck_scores, self.ck_roi, self.ck_orbit):
            c.toggled.connect(self._display_changed)
            l.addWidget(c)
        lay.addWidget(f)

        f, l = panel("Performance report")
        b = QtWidgets.QPushButton("💾  Save report now")
        b.clicked.connect(self.save_report)
        l.addWidget(b)
        b = QtWidgets.QPushButton("📂  Open reports folder")
        b.clicked.connect(self.open_reports)
        l.addWidget(b)
        hint = QtWidgets.QLabel("A report is also saved automatically\nwhen a run finishes.")
        hint.setObjectName("subtitle")
        l.addWidget(hint)
        lay.addWidget(f)
        lay.addStretch(1)
        # Long list entries must not widen the fixed-width panel (they are elided instead).
        for cb in box.findChildren(QtWidgets.QComboBox):
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(6)
        for sp in box.findChildren(QtWidgets.QSpinBox):
            sp.setMinimumWidth(0)
        scroll.setWidget(box)
        return scroll

    def _center(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(10)
        views = QtWidgets.QHBoxLayout()
        views.setSpacing(10)

        f, l = panel()
        head = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel("CAMERA FEED")
        lab.setObjectName("viewTitle")
        head.addWidget(lab)
        head.addStretch(1)
        self.zoom_buttons = {}
        for level in CameraView.LEVELS:
            b = QtWidgets.QPushButton("Fit" if level == 1 else f"{level}×")
            b.setObjectName("toggle")
            b.setCheckable(True)
            b.setStyleSheet("padding: 3px 8px; font-size: 8pt;")
            b.setToolTip("Zoom the camera feed around the tracked beacon (or use the mouse wheel)")
            b.clicked.connect(lambda _=False, z=level: self.camera_view.set_zoom(z))
            head.addWidget(b)
            self.zoom_buttons[level] = b
        l.addLayout(head)
        self.lbl_cam = QtWidgets.QLabel("")
        self.lbl_cam.setObjectName("subtitle")
        l.addWidget(self.lbl_cam)
        self.camera_view = CameraView()
        self.camera_view.zoomChanged.connect(
            lambda z: [b.setChecked(k == z) for k, b in self.zoom_buttons.items()])
        self.zoom_buttons[1].setChecked(True)
        l.addWidget(self.camera_view, 1)
        legend = QtWidgets.QLabel(
            f"<span style='color:{theme.GOOD}'>■</span> beacon-like   "
            f"<span style='color:{theme.BAD}'>■</span> rejected   "
            f"<span style='color:{theme.WARN}'>■</span> undecided / new   "
            f"<span style='color:{theme.ESTIMATE}'>✚</span> track estimate   "
            f"<span style='color:#ffaa3c'>┅</span> tracking window")
        legend.setObjectName("subtitle")
        legend.setWordWrap(True)
        l.addWidget(legend)
        views.addWidget(f, 1)

        f, l = panel()
        head = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel("SITUATIONAL VIEW")
        lab.setObjectName("viewTitle")
        head.addWidget(lab)
        self.mode_buttons = {}
        for mode in VIEW_MODES:
            b = QtWidgets.QPushButton(mode)
            b.setObjectName("toggle")
            b.setCheckable(True)
            b.setStyleSheet("padding: 3px 9px; font-size: 8pt;")
            b.setToolTip("3D view, 2D sky map, or both  [M]")
            b.clicked.connect(lambda _=False, m=mode: self.set_view_mode(m))
            head.addWidget(b)
            self.mode_buttons[mode] = b
        head.addStretch(1)
        self.btn_info = QtWidgets.QPushButton("ⓘ")
        self.btn_info.setToolTip("What do the symbols in the views mean?  [I]")
        self.btn_info.setStyleSheet("padding: 2px 9px; font-size: 11pt; font-weight: 700;")
        self.btn_info.clicked.connect(self._show_legend)
        head.addWidget(self.btn_info)
        self.preset_buttons = []
        for name in SceneView3D.PRESETS:
            b = QtWidgets.QPushButton(name)
            b.setStyleSheet("padding: 3px 9px; font-size: 8pt;")
            b.setToolTip("3D camera preset  [V]")
            b.clicked.connect(lambda _=False, n=name: self.view3d.set_preset(n))
            head.addWidget(b)
            self.preset_buttons.append(b)
        l.addLayout(head)
        self.view3d = SceneView3D()
        self.view3d.setMinimumSize(300, 150)
        l.addWidget(self.view3d, 1)
        self.view2d = SkyMap2D()
        self.view2d.setMinimumSize(300, 150)
        l.addWidget(self.view2d, 1)
        self.view_hint = QtWidgets.QLabel()
        self.view_hint.setObjectName("subtitle")
        l.addWidget(self.view_hint)
        views.addWidget(f, 1)
        self.set_view_mode("3D")

        # Legend of the 3D view's symbols.
        states = "  ".join(f"<span style='color:{c}'>{s.title()}</span>" for s, c in theme.STATE_COLORS.items())
        self.legend = Legend([
            ("B", theme.BEACON, "Beacon", "true position of the target light; pulses with its 4 Hz blink, "
                                          "red trail = recent path"),
            ("●", theme.DECOY, "Decoy", "other lights the tracker must ignore (steady or wrong blink rate)"),
            ("▲", "#9fb7d6", "Camera (PAT terminal)", "pan-tilt camera on its mount; the head turns with the gimbal"),
            ("▭", theme.GOOD, "Camera field of view", "pyramid + footprint on the sky; colour = tracker state:<br>"
                                                     + states),
            ("━", theme.ACCENT, "Line of sight / search path", "where the camera points now, and where it has looked"),
            ("●", theme.ESTIMATE, "Tracker estimate", "where the Kalman filter believes the beacon is"),
            ("◯", "#c7d3e6", "Cloud", "dims or hides the beacon only when the beacon is BEHIND it; "
                                      "the beacon drifts in depth, in front of and behind clouds"),
            ("·", "#c8d4ff", "Stars", "background clutter, always farthest away (3D: on the far dome)"),
            ("○", "#7fc8ff", "Turbulence (3D)", "faint shimmering air pockets along the line of sight; "
                                                "more and brighter = stronger turbulence"),
            ("✦", "#ffffff", "Sensor noise (3D)", "sparkles flickering in the camera footprint; more = noisier"),
            ("▲", "#9fb7d6", "Vibration (3D)", "the camera head shakes with the real jitter, shown 60x larger"),
            ("▢", theme.ACCENT, "Simulated sky", "the region the beacon and decoys move in (see Sky area)"),
        ])
        # Shown as a popup from the ⓘ button (closes on any click outside it).
        self.legend.setParent(self, QtCore.Qt.Popup)
        self.legend.setFixedWidth(300)
        col.addLayout(views, 3)

        f, l = panel()
        self.plots = LivePlots()
        self.plots.setMinimumHeight(140 if self.compact else 190)
        l.addWidget(self.plots)
        col.addWidget(f, 1)
        return col

    def _stats(self):
        w = QtWidgets.QScrollArea()
        w.setWidgetResizable(True)
        w.setFixedWidth(self.stats_w)
        w.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        w.setWidget(inner)
        lay = QtWidgets.QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(6 if self.compact else 8)
        self.c_time = StatCard("Simulated time")
        self.c_acq = StatCard("Acquisition time")
        self.c_ret = StatCard("Lock retention")
        self.c_err = StatCard("Pointing error")
        self.c_fps = StatCard("Frame rate")
        self.c_id = StatCard("Identification")
        for c in (self.c_time, self.c_acq, self.c_ret, self.c_err, self.c_fps, self.c_id):
            lay.addWidget(c)
        f, l = panel("Current scene")
        self.lbl_scene = QtWidgets.QLabel()
        self.lbl_scene.setWordWrap(True)
        self.lbl_scene.setTextFormat(QtCore.Qt.RichText)
        l.addWidget(self.lbl_scene)
        lay.addWidget(f)
        lay.addStretch(1)
        return w

    SHORTCUTS = [
        ("Space", "Start / pause"), ("R", "Restart the same scene"), ("N", "New random scene"),
        ("C", "Disturbances: apply and continue"), ("Shift + R", "Disturbances: apply and restart"),
        ("+  /  -", "Camera feed: zoom in / out"), ("0", "Camera feed: fit (no zoom)"),
        ("M", "Situational view: 3D / 2D / both"), ("W  /  S", "3D view: tilt up / down"),
        ("A  /  D", "3D view: rotate left / right"), ("Q  /  E", "3D view: zoom in / out"),
        ("V", "3D view: next preset view"), ("I", "Legend of the views"),
        ("F2", "Disturbances explained (pictures at current levels)"), ("H  or  F1", "This help"),
        ("Tab", "Move between controls (arrows change a slider / list)"),
    ]

    def _shortcuts(self):
        """Everything can be driven from the keyboard (no mouse needed)."""
        cam = lambda step: self.camera_view.set_zoom(
            CameraView.LEVELS[min(max(CameraView.LEVELS.index(self.camera_view.zoom) + step, 0),
                                  len(CameraView.LEVELS) - 1)])
        keys = [
            ("Space", self.toggle_run), ("R", lambda: self.new_run(new_seed=False)),
            ("N", lambda: self.new_run(new_seed=True)),
            ("C", lambda: self.apply_disturbances(restart=False)),
            ("Shift+R", lambda: self.apply_disturbances(restart=True)),
            ("M", lambda: self.set_view_mode(VIEW_MODES[(VIEW_MODES.index(self.view_mode) + 1) % len(VIEW_MODES)])),
            ("+", lambda: cam(+1)), ("=", lambda: cam(+1)), ("-", lambda: cam(-1)),
            ("0", lambda: self.camera_view.set_zoom(1)),
            ("A", lambda: self.view3d.orbit(-8, 0)), ("D", lambda: self.view3d.orbit(8, 0)),
            ("W", lambda: self.view3d.orbit(0, 5)), ("S", lambda: self.view3d.orbit(0, -5)),
            ("Q", lambda: self._zoom3d(0.85)), ("E", lambda: self._zoom3d(1.18)),
            ("V", self._next_preset), ("I", self._show_legend), ("F2", self._show_disturbance_info),
            ("H", self._show_help), ("F1", self._show_help),
        ]
        for key, fn in keys:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)
            sc.setContext(QtCore.Qt.ApplicationShortcut)

    def _zoom3d(self, factor):
        self.view3d.opts["distance"] = min(max(self.view3d.opts["distance"] * factor, 40), 700)
        self.view3d.update()

    def _next_preset(self):
        names = list(SceneView3D.PRESETS)
        self._preset_i = (getattr(self, "_preset_i", 0) + 1) % len(names)
        self.view3d.set_preset(names[self._preset_i])
        self.status.showMessage(f"3D view: {names[self._preset_i]}", 2000)

    def set_view_mode(self, mode):
        """Situational view: the 3D scene, the 2D sky map, or both stacked."""
        self.view_mode = mode
        self.view3d.setVisible(mode in ("3D", "Both"))
        self.view2d.setVisible(mode in ("2D", "Both"))
        for m, b in self.mode_buttons.items():
            b.setChecked(m == mode)
        for b in self.preset_buttons:
            b.setVisible(mode != "2D")
        self.view_hint.setText(
            "2D sky map: azimuth / elevation, seen face-on  ·  M = switch view  ·  H = all shortcuts" if mode == "2D"
            else "Keys: A/D rotate · W/S tilt · Q/E zoom · V preset · M switch view  (mouse: drag / wheel)"
                 "  ·  H = all shortcuts")
        if self.res is not None:
            self.view2d.update_scene(self.res, draw=True) if mode != "3D" else None

    def _show_disturbance_info(self):
        """Explain the disturbances, pictured at the current slider levels."""
        if not hasattr(self, "dist_info"):
            self.dist_info = DisturbanceInfo(self)
        self.dist_info.refresh({name: s.value() for name, s in self.dist_sliders.items()})
        self.dist_info.show()
        self.dist_info.raise_()
        self.dist_info.activateWindow()

    def _show_help(self):
        rows = "".join(f"<tr><td style='color:{theme.ACCENT};font-family:Consolas;padding-right:18px'>{k}</td>"
                       f"<td>{v}</td></tr>" for k, v in self.SHORTCUTS)
        QtWidgets.QMessageBox.information(self, "Keyboard shortcuts", f"<table cellspacing=4>{rows}</table>")

    # ------------------------------------------------------------- runs
    def _build_config(self, new_seed):
        key = SCENARIOS[self.cb_scenario.currentIndex()][1]
        if key == "random":
            cfg = load_config()
        elif key == "default":
            cfg = load_config()
            cfg["randomize"]["enabled"] = False
            cfg["simulation"]["seed"] = 42
        else:
            cfg = load_config(key)
        # Random scene: a new seed for a new scene, the same seed for a restart
        # (the seed is kept internally and written to the reports, not shown).
        if key == "random":
            cfg["simulation"]["seed"] = None if (new_seed or self._seed is None) else self._seed
        cfg["identification"]["method"] = METHODS[self.cb_method.currentIndex()][1]
        if key == "random":
            cfg["randomize"]["decoys"] = [self.sp_dec_lo.value(), self.sp_dec_hi.value()]
        # Sky area; the star count scales with it so the star density stays the same.
        w = cfg["world"]
        old_area = (w["az_range"][1] - w["az_range"][0]) * (w["el_range"][1] - w["el_range"][0])
        a0, a1, e0, e1 = SKY_AREAS[self.cb_sky.currentIndex()][1]
        w["az_range"], w["el_range"] = [a0, a1], [e0, e1]
        w["num_stars"] = int(round(w.get("num_stars", 0) * (a1 - a0) * (e1 - e0) / old_area))
        cfg["simulation"]["duration"] = float(self.sp_duration.value())
        return cfg

    def new_run(self, new_seed):
        was_running = self.running or self.sim is None
        self.pause()
        if new_seed:
            self._dist_override = None          # a new scene brings its own disturbance levels
        cfg = self._build_config(new_seed)
        self.sim = Simulation(cfg)
        self._seed = self.sim.seed
        for name, v in (self._dist_override or {}).items():
            self.sim.disturb.set_strength(name, v)
        self.overlay = Overlay(self.sim)
        self.overlay.show_hud = self.overlay.show_minimap = False
        self._display_changed()
        for name, s in self.dist_sliders.items():
            s.set_value(self.sim.disturb.strength(name), emit=False)
        self._set_pending(False)
        self.view3d.bind_sim(self.sim)
        self.view2d.bind_sim(self.sim)
        self.plots.clear()
        tr = self.sim.tracker
        self.plots.set_thresholds(self.sim.metrics.handover_px, tr.accept, tr.reject)
        self._describe_scene()
        self.res = self.sim.step()
        self._draw(full=True)
        self.status.showMessage("New run.   Keyboard: Space start/pause · R restart · N new scene · "
                                "C / Shift+R apply disturbances · +/- zoom camera · M 2D/3D · H all shortcuts", 12000)
        if was_running:
            self.start()

    def _set_pending(self, pending):
        self.lbl_pending.setVisible(pending)

    def apply_disturbances(self, restart):
        """Apply the slider levels: from the current moment (continue) or from
        the start of the same scene (restart). The levels are kept for later
        restarts of this scene; a new scene brings its own."""
        self._dist_override = {name: s.value() for name, s in self.dist_sliders.items()}
        if restart:
            self.new_run(new_seed=False)
            self.status.showMessage("Disturbances applied - scene restarted from the beginning", 6000)
            return
        for name, v in self._dist_override.items():
            self.sim.disturb.set_strength(name, v)
        self._set_pending(False)
        self._describe_scene()
        self.status.showMessage(f"Disturbances applied at t = {self.res.t:.1f} s - the run continues", 6000)

    def start(self):
        if self.sim is None or self.sim.finished:
            return
        self.running = True
        self.btn_run.setText("⏸  Pause")
        self._t_wall = time.perf_counter()
        self._sim_debt = 0.0
        self.timer.start(5)

    def pause(self):
        self.running = False
        self.timer.stop()
        self.btn_run.setText("▶  Start")

    def toggle_run(self):
        self.pause() if self.running else self.start()

    def _tick(self):
        now = time.perf_counter()
        elapsed, self._t_wall = now - self._t_wall, now
        budget_end = now + 0.028                    # keep the UI responsive
        if self.speed > 0:
            self._sim_debt = min(self._sim_debt + elapsed * self.speed, 0.5)
            steps = int(self._sim_debt / self.sim.dt)
            self._sim_debt -= steps * self.sim.dt
        else:
            steps = 1000                            # "Max": as many as the budget allows
        done = 0
        while done < steps and not self.sim.finished:
            self.res = self.sim.step()
            self._record(self.res)
            done += 1
            if time.perf_counter() > budget_end:
                self._sim_debt = 0.0                # machine too slow: don't pile up
                break
        if done:
            self._draw()
        if self.sim.finished:
            self.pause()
            folder = self.sim.metrics.save()
            self.status.showMessage(f"Run finished - performance report saved to {folder}")
            self._draw(full=True)

    def _record(self, res):
        sel = res.track.selected
        score = sel.score if (sel is not None and sel.extra.get("ready")) else None
        self.plots.add(res.t, res.stats["err_px"], bool(res.stats["locked"]), score)

    # ------------------------------------------------------------- drawing
    def _draw(self, full=False):
        self._tick_n += 1
        res = self.res
        self.camera_view.show_frame(self.overlay.draw(res, paused=not self.running), self._zoom_center(res))
        self.badge.set_state(res.track.state.value)
        self.lbl_cam.setText(f"{self.sim.camera.width}×{self.sim.camera.height}  ·  "
                             f"FOV {self.sim.camera.fov_x:.0f}°×{self.sim.camera.fov_y:.0f}°  ·  "
                             f"az {res.pose[0]:+6.2f}°  el {res.pose[1]:+6.2f}°")
        if full or self._tick_n % 2 == 0:
            self.view3d.update_scene(res)
            self.view2d.update_scene(res, draw=self.view_mode != "3D")
        if full or self._tick_n % 3 == 0:
            self.plots.refresh()
        if full or self._tick_n % 5 == 0:
            self._update_cards()

    def _zoom_center(self, res):
        """Zoom centre: the tracker's estimate of the beacon, else the image centre."""
        if res.track.estimate is None:
            return None
        x, y = self.sim.camera.world_to_pixel(res.track.estimate[0], res.track.estimate[1], res.pose)
        return float(x), float(y)

    def _update_cards(self):
        live = self.sim.metrics.live()
        res = self.res
        dur = self.sim.duration
        self.c_time.set(f"{res.t:6.1f} s", f"of {dur:.0f} s" if dur > 0 else "endless run")
        acq = live.get("acq_time")
        self.c_acq.set("--" if acq is None else f"{acq:.1f} s",
                       "searching..." if acq is None else "first confirmed lock",
                       None if acq is None else theme.GOOD)
        ret = live.get("retention")
        col = None if ret is None else (theme.GOOD if ret >= 90 else theme.WARN if ret >= 60 else theme.BAD)
        self.c_ret.set("--" if ret is None else f"{ret:.1f} %", f"lock losses: {live.get('lock_losses', 0)}", col)
        err = live.get("err_px", 0.0)
        ho = self.sim.metrics.handover_px
        self.c_err.set(f"{err:.1f} px", f"{live.get('err_mrad', 0):.2f} mrad   (handover < {ho:.0f} px)",
                       theme.GOOD if err <= ho else None)
        self.c_fps.set(f"{live.get('fps', 0):.1f}", f"processing {live.get('proc_ms', 0):.1f} ms / frame")
        tr = self.sim.tracker
        self.c_id.set(self.sim.id_method.upper(),
                      f"rejected {tr.rejections}   ·   dropped {tr.dropped_locks}")

    def _show_legend(self):
        """Open the legend popup just below the ⓘ button, kept on screen."""
        self.legend.adjustSize()
        pos = self.btn_info.mapToGlobal(QtCore.QPoint(self.btn_info.width() - self.legend.width(),
                                                      self.btn_info.height() + 4))
        screen = QtGui.QGuiApplication.screenAt(pos) or QtGui.QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        pos.setX(max(area.left(), min(pos.x(), area.right() - self.legend.width())))
        pos.setY(max(area.top(), min(pos.y(), area.bottom() - self.legend.height())))
        self.legend.move(pos)
        self.legend.show()

    def _describe_scene(self):
        s = self.sim
        i = s.scene_info
        a0, a1, e0, e1 = s.scene.bounds
        rows = [("Sky area", f"az ±{a1:.0f}°, el ±{e1:.0f}°")]
        if i:
            rows += [("Beacon path", f"{i['beacon_path'].replace('_', ' ')}, {i['beacon_speed_dps']} °/s"),
                     ("Decoys", f"{i['decoys']}  ({i['blinking_decoys']} blinking)")]
        else:
            n = len(s.scene.targets) - 1
            rows += [("Scene", SCENARIOS[self.cb_scenario.currentIndex()][0]), ("Decoys", str(n))]
        rows += [("Turbulence", f"{s.disturb.strength('turbulence'):.2f}"),
                 ("Vibration", f"{s.disturb.strength('vibration'):.2f}"),
                 ("Sensor noise", f"{s.disturb.strength('sensor'):.2f}")]
        if i:
            rows += [("Clouds", "yes" if i["clouds"] else "no"),
                     ("GPS error", f"{i['initial_pointing_error_deg']:.1f}°  (reported vs true)")]
            if "camera_to_prior_deg" in i:
                rows.append(("Camera start", f"{i['camera_to_prior_deg']:.0f}° from reported position"))
        b = s.scene.beacon
        rows.append(("Beacon code", f"{b.blink_hz:.1f} Hz blink"))
        self.lbl_scene.setText("<table cellspacing=3>" + "".join(
            f"<tr><td style='color:{theme.TEXT_DIM}'>{k}</td><td>&nbsp;&nbsp;{v}</td></tr>" for k, v in rows)
            + "</table>")

    def _display_changed(self):
        if not hasattr(self, "overlay"):
            return
        self.overlay.show_truth = False
        self.overlay.show_scores = self.ck_scores.isChecked()
        self.overlay.show_roi = self.ck_roi.isChecked()
        self.view3d.auto_orbit = self.ck_orbit.isChecked()
        if self.res is not None and not self.running:
            self.camera_view.show_frame(self.overlay.draw(self.res, paused=True), self._zoom_center(self.res))

    # ------------------------------------------------------------- reports
    def save_report(self):
        folder = self.sim.metrics.save()
        self.status.showMessage(f"Performance report saved to {folder}", 8000)

    def open_reports(self):
        folder = Path(self.sim.metrics.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder)) if hasattr(os, "startfile") else None
