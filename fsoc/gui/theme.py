"""Colours and the Qt stylesheet: a dark "mission control" look.

Colour meaning is kept consistent across every view:
  cyan   - the camera / line of sight          red/pink - the beacon
  amber  - decoys                               magenta  - tracker estimate
  state colours as in the camera overlay (SEARCH red, ACQUIRING orange,
  TRACKING green, COASTING yellow).
"""

BG = "#0a0e14"
PANEL = "#111823"
PANEL_2 = "#162030"
BORDER = "#22314a"
TEXT = "#d7e3f4"
TEXT_DIM = "#7f93ad"
ACCENT = "#38c6ff"        # camera / line of sight
BEACON = "#ff4d6d"
DECOY = "#ffb020"
ESTIMATE = "#d65cff"
GOOD = "#35d07f"
WARN = "#ffcc33"
BAD = "#ff5c5c"

STATE_COLORS = {
    "SEARCH": "#ff5c5c",
    "ACQUIRING": "#ff9a2e",
    "TRACKING": "#35d07f",
    "COASTING": "#ffd54a",
}


def rgba(hex_color, alpha=1.0):
    """'#rrggbb' -> (r, g, b, a) floats for pyqtgraph.opengl items."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, alpha)


STYLESHEET = f"""
* {{ font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#title {{ font-size: 15pt; font-weight: 600; color: {TEXT}; }}
QLabel#subtitle {{ color: {TEXT_DIM}; font-size: 9pt; }}
QLabel#section {{ color: {TEXT_DIM}; font-size: 8pt; font-weight: 700; letter-spacing: 1.5px; }}
QLabel#cardValue {{ font-size: 16pt; font-weight: 600; font-family: 'Consolas', 'Cascadia Mono', monospace; }}
QLabel#cardLabel {{ color: {TEXT_DIM}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#cardSub {{ color: {TEXT_DIM}; font-size: 8pt; font-family: 'Consolas', monospace; }}
QLabel#stateBadge {{ font-size: 13pt; font-weight: 700; letter-spacing: 2px; border-radius: 6px; padding: 6px 14px; }}
QLabel#viewTitle {{ color: {TEXT_DIM}; font-size: 8pt; font-weight: 700; letter-spacing: 1.5px; }}
QPushButton {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 7px 12px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #0d1622; }}
QPushButton#primary {{ background: #0f5f86; border-color: #1a86ba; }}
QPushButton#primary:hover {{ background: #137aac; }}
QPushButton#toggle:checked {{ background: #0f5f86; border-color: {ACCENT}; }}
QComboBox, QLineEdit, QSpinBox {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 8px; min-height: 20px;
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; selection-background-color: #0f5f86; border: 1px solid {BORDER}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid {BORDER}; background: {PANEL_2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
QStatusBar {{ background: {PANEL}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: {BG}; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #33507a; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
"""
