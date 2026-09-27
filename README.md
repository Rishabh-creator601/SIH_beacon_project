# FSOC Virtual Camera Tracking Simulator

Software simulation of the **coarse-alignment stage of PAT** (Pointing,
Acquisition and Tracking) for Free-Space Optical Communication. A virtual
pan-tilt camera searches the sky, detects a moving optical beacon, locks on
and keeps it centred despite turbulence, vibration, clouds, noise and
decoy lights, while logging tracking performance.

## Quick start

```bash
pip install -r requirements.txt
python app.py                                           # desktop application (recommended)
python main.py                                          # lightweight OpenCV viewer, random scene
python main.py --config scenarios/02_strong_turbulence.yaml
python main.py --headless --duration 120                # fast batch run, report only
```

Viewer keys: `space` pause, `t` show true positions (beacon red, decoys orange),
`r` reset, `q` quit.

Every run writes a performance log to `runs/<run name>/`:

| File | Content |
|---|---|
| `report.html` | Formatted report: run info, performance table, **interactive 3D views** (sky dome with beacon path and camera pointing; az/el/time), charts. Self-contained, works offline |
| `report.pdf` | The same report as a printable PDF (3D views rendered as images) |
| `frames.csv` | Per-frame data (states, positions, errors, timing) |
| `summary.json` / `summary.txt` | All summary metrics |

## Standalone executable

Build inside a minimal virtual environment so only what the app needs is bundled:

```bash
python -m venv build/venv
buildenv\Scripts\python -m pip install numpy opencv-python-headless PyYAML PySide6-Essentials pyqtgraph PyOpenGL matplotlib reportlab pyinstaller
buildenv\Scripts\python tools/build_exe.py            # builds both variants into dist/
```

| Variant | Path | Notes |
|---|---|---|
| Folder (fast start) | `dist/FSOC_Simulator/FSOC_Simulator.exe` | Opens in a few seconds; share the folder as a zip |
| Single file | `dist/FSOC_Simulator.exe` | One file; unpacks itself on every start, so it opens more slowly |

Both open only the application window (no console). Reports go to a `runs` folder next to the .exe.
Check a build without opening the GUI: `FSOC_Simulator.exe --selftest 20` then read `runs/selftest.txt`.

## Evaluation (multi-seed)

```bash
python tools/evaluate.py --suite random --seeds 1-40 --methods hybrid none
python tools/evaluate.py --suite scenarios --seeds 1-10
```
Writes `runs/evaluation/summary.md` (tables with 95 % confidence intervals and breakdowns by
turbulence / decoys / speed / clouds / path), `summary.csv`, `results.csv` and box plots.
Results are saved after every run, so an interrupted evaluation resumes where it stopped.

## Desktop application (`app.py`)

- **Camera feed** with detections, beacon scores, track estimate and tracking window
- **3D situational view**: pan-tilt terminal, sky dome, field-of-view pyramid coloured
  by tracker state, search path, beacon trail, decoys, clouds (drag to rotate, wheel to zoom)
- **Live plots** of pointing error and beacon score; **performance cards**
- Controls: start / pause / restart, **new random scene**, scenario and identification
  method, speed (0.5x to max), **live sliders** for turbulence, vibration and sensor noise,
  report export. Keys: `Space` start/pause, `R` restart, `N` new random scene, `T` truth.

## Random scenes

By default every run is a new random scene (`randomize` in `config/default.yaml`):
random beacon path (lissajous / circular / random walk, random speed), a random number
of decoys (0-5, steady or blinking at wrong rates), random turbulence, vibration and
clouds, and a random initial pointing error (the camera starts aimed at the beacon's
"reported" GPS/ephemeris position). The run's **seed** is shown in the app and saved
in every report; enter it (or pass `--seed`) to replay the exact run.

## Scenarios

The fixed, reproducible test cases (random scenes switched off, seed 42):

A scenario file only lists what differs from `config/default.yaml`.

| File | What it tests |
|---|---|
| `scenarios/01_clear_sky.yaml` | No disturbances: the ideal baseline |
| `scenarios/02_strong_turbulence.yaml` | Heavy turbulence, vibration and sensor noise |
| `scenarios/03_occlusion.yaml` | Dense clouds + beacon dropouts: coasting and re-acquisition |
| `scenarios/04_decoys.yaml` | Other lights of similar brightness: target identification |
| `scenarios/05_fast_target.yaml` | Fast, manoeuvring target: controller and filter agility |
| `scenarios/06_worst_case.yaml` | Everything at once |

## Beacon identification (AI)

Every bright dot looks alike in one frame; the beacon is recognised by its
behaviour over time (it blinks at a known 4 Hz "code"). Each candidate is
followed across frames and scored with P(beacon):

| `identification.method` | How |
|---|---|
| `hybrid` (default) | Average of the two below |
| `cnn` | BeaconNet: small CNN on the candidate's last 24 image patches (0.8 s clip) |
| `blink` | Classical: signal-to-noise of the 4 Hz peak in the candidate's brightness |
| `none` | Baseline: lock onto the brightest dot |

Scores are accumulated as averaged log-odds (memory up to 3 s) before the tracker
accepts (≥ 0.7) or rejects (≤ 0.3, then ignored for 6 s) a candidate.

Retraining the CNN (optional; a trained model ships in `models/`):

```bash
python tools/generate_dataset.py        # dataset v2: incl. harmonic decoys, turbulence up to 3
python tools/train_classifier.py        # trains on it -> models/candidate/ (never overwrites the app's model)
python tools/evaluate.py --suite random --seeds 1-40 --model models/candidate/beacon_cnn.onnx --label hybrid-v2
python tools/compare_methods.py         # closed-loop comparison of none/blink/cnn/hybrid
```

## How it works

```
scene.update ─► render view ─► disturbances ─► detect blobs ─► identify ─► Kalman tracker ─► PID controller ─► move gimbal
                                                                                 └──────────► performance monitor
```

| Module | Role |
|---|---|
| `fsoc/config.py` | Loads `config/default.yaml`, merges a user scenario over it |
| `fsoc/targets.py` | Beacon and decoys; trajectories: static, linear, lissajous, circular, random_walk; blinking |
| `fsoc/scene.py` | Virtual sky: stars and targets as Gaussian spots, per-source motion blur |
| `fsoc/camera.py` | Virtual pan-tilt camera: FOV, projection, slew-rate/acceleration limits |
| `fsoc/disturbances.py` | Vibration, turbulence (twinkle, wander, blur, shimmer), clouds, dropouts, glints, glare, sensor noise |
| `fsoc/detector.py` | Adaptive threshold + connected components + sub-pixel centroid |
| `fsoc/identification.py` | Candidate tracklets, blink SNR test, CNN scorer (OpenCV DNN), evidence accumulation |
| `fsoc/tracker.py` | Kalman filter, gating, SEARCH/ACQUIRING(verify)/TRACKING/COASTING, decoy blacklist |
| `fsoc/controller.py` | PID + velocity feed-forward when tracking, predictive spiral scan when searching |
| `fsoc/metrics.py` | Acquisition time, error, lock retention, false locks, FPS, processing time, report files |
| `fsoc/simulation.py` | Wires everything into one closed loop (GUI-independent) |
| `fsoc/visualization.py` | OpenCV HUD, detection boxes with beacon scores, track estimate, world minimap |
| `tools/` | Dataset generator, CNN training/export, method comparison |
| `models/` | Trained `beacon_cnn.onnx`, training report, curves, confusion matrix |

All angles are in degrees. Every parameter is documented in `config/default.yaml`;
each disturbance has an `enabled` switch and a `strength` multiplier.
