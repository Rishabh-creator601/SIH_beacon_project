"""FSOC Coarse PAT Simulator - desktop application.

    python app.py                     # open the application
    python app.py --selftest 20       # no window: run 20 s, write a report, exit

(`python main.py` still offers the lightweight OpenCV viewer and headless
batch runs.)

The self-test is mainly for checking a packaged .exe: it exercises the
simulation, the CNN (OpenCV DNN) and the HTML/PDF report generation, and
writes the outcome to runs/selftest.txt next to the executable.
"""

import sys
import traceback


def selftest(seconds):
    from fsoc.config import OUTPUT_ROOT, load_config
    from fsoc.simulation import Simulation

    out = OUTPUT_ROOT / "runs"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "selftest.txt"
    try:
        cfg = load_config()
        cfg["simulation"].update(duration=float(seconds), seed=8)
        sim = Simulation(cfg)
        while not sim.finished:
            sim.step()
        folder = sim.metrics.save("selftest_run")
        s = sim.metrics.summary()
        files = sorted(p.name for p in folder.iterdir())
        ok = {"report.html", "report.pdf"} <= set(files)
        log.write_text(
            f"SELFTEST {'PASSED' if ok else 'FAILED'}\n"
            f"identification method: {sim.id_method}\n"
            f"frames: {s['frames']}  acquired: {s['acquired']}  time locked: {s['time_locked_pct']} %\n"
            f"files: {', '.join(files)}\n", encoding="utf-8")
        return 0 if ok else 1
    except Exception:
        log.write_text("SELFTEST FAILED\n" + traceback.format_exc(), encoding="utf-8")
        return 1


def main():
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        seconds = float(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 20.0
        sys.exit(selftest(seconds))

    from PySide6 import QtWidgets

    from fsoc.gui.main_window import MainWindow
    from fsoc.gui.theme import STYLESHEET

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FSOC Coarse PAT Simulator")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.showMaximized()          # fill the screen (taskbar stays visible)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
