"""Beacon identification: deciding WHICH bright dot is the designated beacon.

In a single frame every light source is just a Gaussian spot, so the
beacon, decoys and stars look alike. What differs is their behaviour over
time: the beacon blinks at a known rate (its "code"), decoys are steady or
blink at other rates, glints vanish, and hot pixels stay fixed on the
sensor while the sky moves. Identification therefore works on short
histories of each candidate:

  CandidateTracker  follows EVERY detection from frame to frame (in world
                    coordinates, so camera motion is removed) and stores a
                    small image patch of it each frame -> a "tracklet".

  BlinkScorer       classical baseline: aperture photometry on the patches,
                    then the fraction of signal power at the beacon's blink
                    frequency (Goertzel / single-bin DFT).

  CNNScorer         AI method: a small convolutional network that looks at
                    the last T patches as a T-channel image (a short video
                    clip) and outputs P(beacon). Trained on simulator data
                    by tools/train_classifier.py, run with OpenCV's DNN
                    module (no PyTorch needed at runtime).

  hybrid            average of the two (default): on held-out validation
                    scenes it made the fewest false "this is the beacon"
                    calls, because the CNN is better at recognising the
                    beacon and the blink test better at rejecting decoys.

Each detection gets ``score`` in [0, 1] (0.5 = not enough history yet) and
``extra['tid']`` (its tracklet id), which the tracker uses to accept or
reject candidates.
"""

import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .config import PROJECT_ROOT

PATCH = 24          # patch size (pixels) - must match the trained model
UNKNOWN = 0.5       # score before a tracklet has enough history


def normalize_stack(stack):
    """(T, S, S) uint8 patches -> float32 network input, brightness-invariant.

    Subtracting the background and dividing by the brightest value keeps the
    *relative* temporal pattern (blinking, fading, vanishing) while making
    the result independent of absolute beacon power or cloud dimming.
    """
    x = stack.astype(np.float32)
    x -= np.median(x)
    x /= max(float(x.max()), 20.0)
    return np.clip(x, -0.5, 1.5)


def aperture_flux(patch, r=3):
    """Background-subtracted brightness in a small box at the patch centre."""
    c = patch.shape[0] // 2
    core = patch[c - r:c + r + 1, c - r:c + r + 1].astype(np.float32)
    return float(core.sum() - np.median(patch) * core.size)


class Tracklet:
    def __init__(self, tid, det, history):
        self.id = tid
        self.az, self.el = det.az, det.el
        self.vaz = self.vel = 0.0
        self.misses = 0
        self.age = 0
        self.patches = deque(maxlen=history)
        # Brightness is kept twice as long: frequency analysis gets much
        # sharper with a longer window, and a number per frame is cheap.
        self.flux = deque(maxlen=2 * history)
        self.score = UNKNOWN
        self.ready = False
        self.evidence = 0.0            # running mean log-odds of "is the beacon"
        self.n_scored = 0              # clips scored so far

    def predict(self, dt):
        self.az += self.vaz * dt
        self.el += self.vel * dt

    def update(self, det, dt):
        # Smoothed velocity from the jump between the predicted and measured
        # position, spread over the frames since the last detection.
        if dt > 0 and self.age > 0:
            span = dt * (self.misses + 1)
            self.vaz += 0.5 * (det.az - self.az) / span
            self.vel += 0.5 * (det.el - self.el) / span
        self.az, self.el = det.az, det.el
        self.misses = 0


class CandidateTracker:
    """Greedy nearest-neighbour association of all detections across frames."""

    def __init__(self, history, gate_deg=0.25, max_misses=20):
        self.history = history
        self.gate = gate_deg
        self.max_misses = max_misses
        self.tracklets = {}
        self._next_id = 1

    def update(self, detections, frame, camera, pose, dt):
        tracks = list(self.tracklets.values())
        for tr in tracks:
            tr.predict(dt)

        # Greedy matching, closest pairs first. The gate widens while a
        # tracklet goes undetected (blink-off, fades), because its predicted
        # position becomes less certain with every missed frame.
        pairs = []
        for i, d in enumerate(detections):
            for tr in tracks:
                dist = math.hypot(d.az - tr.az, d.el - tr.el)
                if dist < min(self.gate * (1.0 + 0.5 * tr.misses), 1.0):
                    pairs.append((dist, i, tr))
        pairs.sort(key=lambda p: p[0])
        used_det, used_tr = set(), set()
        for _, i, tr in pairs:
            if i in used_det or tr.id in used_tr:
                continue
            tr.update(detections[i], dt)
            detections[i].extra["tid"] = tr.id
            used_det.add(i)
            used_tr.add(tr.id)
        for tr in tracks:
            if tr.id not in used_tr:
                tr.misses += 1
                if tr.misses > self.max_misses:
                    del self.tracklets[tr.id]
        for i, d in enumerate(detections):
            if i not in used_det:
                tr = Tracklet(self._next_id, d, self.history)
                self._next_id += 1
                self.tracklets[tr.id] = tr
                d.extra["tid"] = tr.id

        # Record a patch for every live tracklet - also when it was not
        # detected this frame, so blink-off frames and fades are captured.
        for tr in self.tracklets.values():
            x, y = camera.world_to_pixel(tr.az, tr.el, pose)
            patch = cv2.getRectSubPix(frame, (PATCH, PATCH), (float(x), float(y)))
            tr.patches.append(patch)
            tr.flux.append(aperture_flux(patch))
            tr.age += 1
            tr.ready = len(tr.patches) == self.history

    def stack(self, tr):
        return np.stack(tr.patches)


class BlinkScorer:
    """Is there a spectral peak at the beacon's blink frequency?

    Signal-to-noise test on the candidate's brightness history: power at the
    blink frequency f0 divided by the average power at the OTHER frequencies
    (away from f0 and its 2nd harmonic). Broadband flicker such as
    scintillation raises both equally and cancels out; a decoy blinking at
    another rate raises the noise estimate instead of the signal.
    """

    def __init__(self, blink_hz, fps):
        self.f0 = blink_hz
        self.fps = fps
        self._basis = {}               # window length -> (DFT matrix, signal idx, noise mask)

    SUBHARMONICS = (2, 3, 4)

    def _dft(self, n):
        if n not in self._basis:
            grid = np.arange(0.5, self.fps / 2 - 0.25, 0.25)
            # Rows: the noise grid, then f0, then f0/2, f0/3, f0/4 exactly.
            freqs = np.concatenate([grid, [self.f0], [self.f0 / k for k in self.SUBHARMONICS]])
            k = np.arange(n)
            w = np.hanning(n)
            E = w[None, :] * np.exp(-2j * np.pi * freqs[:, None] * k[None, :] / self.fps)
            noise = np.zeros(len(freqs), bool)
            noise[:len(grid)] = (np.abs(grid - self.f0) > 1.0) & (np.abs(grid - 2 * self.f0) > 1.0)
            self._basis[n] = (E, len(grid), noise)
        return self._basis[n]

    def snr(self, flux):
        x = np.asarray(flux, np.float64)
        E, sig, noise = self._dft(len(x))
        P = np.abs(E @ (x - x.mean())) ** 2
        snr = P[sig] / (P[noise].mean() + 1e-9)
        # Harmonic check: a light blinking at f0/2 (or f0/3, f0/4) as a square
        # wave also has power AT f0 - its harmonic - and would look like the
        # beacon. Such a light has more power at the sub-frequency than at
        # f0; the real beacon has almost none there. Penalise accordingly.
        sub = P[sig + 1:sig + 1 + len(self.SUBHARMONICS)].max()
        return float(snr * min(1.0, P[sig] / (sub + 1e-9)))

    def score(self, tracklets):
        # SNR ~1 for pure noise, >>1 for a real blink. Logistic mapping with
        # SNR 3 -> 0.5, SNR 10 -> ~0.9, SNR 1 -> ~0.1 (calibrated on the
        # validation set, where the beacon's median SNR is ~7 and every
        # decoy type's median is 0.5-1.5).
        out = {}
        for tr in tracklets:
            z = 4.2 * (np.log10(self.snr(tr.flux) + 1e-9) - np.log10(3.0))
            out[tr.id] = float(1.0 / (1.0 + np.exp(-z)))
        return out


class CNNScorer:
    """Runs the trained ONNX classifier on a batch of tracklet clips."""

    def __init__(self, model_path):
        path = Path(model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"CNN model not found: {path}\n"
                "Train it with: python tools/generate_dataset.py && python tools/train_classifier.py\n"
                "or set identification.method to 'blink'.")
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def score(self, tracklets, stacker):
        if not tracklets:
            return {}
        batch = np.stack([normalize_stack(stacker(tr)) for tr in tracklets])
        self.net.setInput(batch)
        logits = self.net.forward().reshape(len(tracklets), -1)
        # Two logits (not-beacon, beacon) -> softmax probability of beacon.
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p[:, 1] / p.sum(axis=1)
        return {tr.id: float(v) for tr, v in zip(tracklets, p)}


class BeaconIdentifier:
    """Scores every detection with P(beacon).

    method: hybrid (mean of CNN and blink - best precision on validation
    data) | cnn | blink | none.
    """

    METHODS = ("hybrid", "cnn", "blink", "none")

    def __init__(self, cfg, fps):
        self.method = cfg.get("method", "hybrid")
        if self.method not in self.METHODS:
            raise ValueError(f"identification.method must be one of {self.METHODS}")
        self.history = int(cfg.get("window_frames", 24))
        self.evidence_frames = int(cfg.get("evidence_frames", 90))
        self.min_evidence = int(cfg.get("min_evidence_frames", 24))
        self.candidates = CandidateTracker(self.history, cfg.get("association_gate_deg", 0.25))
        self.blink = self.cnn = None
        if self.method in ("blink", "hybrid"):
            self.blink = BlinkScorer(float(cfg.get("beacon_blink_hz", 4.0)), fps)
        if self.method in ("cnn", "hybrid"):
            self.cnn = CNNScorer(cfg.get("model_path", "models/beacon_cnn.onnx"))

    @property
    def enabled(self):
        return self.method != "none"

    def _raw_scores(self, ready):
        if self.method == "blink":
            return self.blink.score(ready)
        cnn = self.cnn.score(ready, self.candidates.stack)
        if self.method == "cnn":
            return cnn
        blink = self.blink.score(ready)
        return {tid: 0.5 * (cnn[tid] + blink[tid]) for tid in cnn}

    def process(self, frame, detections, camera, pose, dt):
        if not self.enabled:
            return                              # keep the brightness score
        self.candidates.update(detections, frame, camera, pose, dt)
        ready = [tr for tr in self.candidates.tracklets.values() if tr.ready]
        raw = self._raw_scores(ready)
        for tr in ready:
            # Evidence accumulation (a leaky sequential test): each clip's
            # probability becomes log-odds, which are averaged over roughly
            # the last `evidence_frames` clips. Single noisy clips (a deep
            # scintillation fade, a blink hidden by noise) then barely move
            # the verdict, while consistent evidence builds up quickly.
            p = min(max(raw[tr.id], 0.02), 0.98)
            logodds = math.log(p / (1.0 - p))
            tr.n_scored += 1
            alpha = max(1.0 / tr.n_scored, 1.0 / self.evidence_frames)
            tr.evidence += alpha * (logodds - tr.evidence)
            tr.score = 1.0 / (1.0 + math.exp(-tr.evidence))
        for d in detections:
            tr = self.candidates.tracklets.get(d.extra.get("tid"))
            decided = tr is not None and tr.ready and tr.n_scored >= self.min_evidence
            d.score = tr.score if decided else UNKNOWN
            d.extra["ready"] = decided
            d.extra["n_scored"] = tr.n_scored if tr is not None else 0
