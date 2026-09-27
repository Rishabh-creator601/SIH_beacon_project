"""Beacon candidate detection.

Classical bright-spot detector:
  1. light smoothing to suppress single-pixel noise,
  2. threshold (adaptive: background median + k * robust noise),
  3. connected-component labelling,
  4. intensity-weighted centroid for sub-pixel position.

It returns *candidates*; deciding which candidate is the beacon is the
tracker's job (and, later, the CNN classifier's).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    x: float                 # sub-pixel centroid (image coords)
    y: float
    area: int                # pixels above threshold
    peak: float              # brightest pixel value
    flux: float              # summed brightness above background
    bbox: tuple              # (x, y, w, h)
    score: float = 0.0       # "how beacon-like" (higher = better)
    az: float = 0.0          # world position, filled in by the simulation
    el: float = 0.0
    extra: dict = field(default_factory=dict)


class BlobDetector:
    def __init__(self, cfg):
        self.mode = cfg.get("threshold_mode", "adaptive")
        self.k_sigma = float(cfg.get("k_sigma", 8.0))
        self.min_threshold = float(cfg.get("min_threshold", 110))
        self.fixed_threshold = float(cfg.get("fixed_threshold", 120))
        self.min_area = int(cfg.get("min_area", 2))
        self.max_area = int(cfg.get("max_area", 400))
        self.blur_ksize = int(cfg.get("blur_ksize", 3))
        # Region-of-interest search around the tracker's prediction.
        self.roi_half = int(cfg.get("roi_half_px", 40))
        self.roi_k_sigma = float(cfg.get("roi_k_sigma", 5.0))
        self.roi_min_threshold = float(cfg.get("roi_min_threshold", 45))
        self.last_threshold = 0.0
        self.last_background = 0.0
        self.last_roi = None

    def _threshold(self, img):
        # Background statistics from a sub-sampled image (fast, and the few
        # bright spots barely affect the median).
        sample = img[::4, ::4]
        bg = float(np.median(sample))
        mad = float(np.median(np.abs(sample - bg)))
        noise = 1.4826 * mad + 1e-3          # robust standard deviation
        self.last_background = bg
        if self.mode == "fixed":
            return self.fixed_threshold, bg
        return max(bg + self.k_sigma * noise, self.min_threshold), bg

    def detect(self, frame, roi_center=None):
        """Find bright blobs in an 8-bit grayscale frame.

        ``roi_center`` (x, y): where the tracker expects the beacon. Inside a
        small window around it a much lower threshold is used, so a beacon
        that fades (blink-off, deep scintillation fade) is still found. A low
        threshold over the whole image would flood the tracker with stars and
        noise; inside a small window it is safe.
        """
        img = frame
        if self.blur_ksize >= 3:
            img = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        thr, bg = self._threshold(img)
        self.last_threshold = thr

        _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        self.last_roi = None
        if roi_center is not None and self.roi_half > 0:
            h, w = img.shape
            cx, cy = int(round(roi_center[0])), int(round(roi_center[1]))
            x0, x1 = max(cx - self.roi_half, 0), min(cx + self.roi_half, w)
            y0, y1 = max(cy - self.roi_half, 0), min(cy + self.roi_half, h)
            if x0 < x1 and y0 < y1:
                noise = max((thr - bg) / self.k_sigma, 1e-3) if self.mode != "fixed" else 3.0
                roi_thr = max(bg + self.roi_k_sigma * noise, self.roi_min_threshold)
                mask[y0:y1, x0:x1] |= ((img[y0:y1, x0:x1] > roi_thr) * 255).astype(np.uint8)
                self.last_roi = (x0, y0, x1, y1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        detections = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not self.min_area <= area <= self.max_area:
                continue
            # Centroid on the un-blurred frame, weighted by brightness above background.
            roi = frame[y:y + h, x:x + w].astype(np.float32) - bg
            roi[labels[y:y + h, x:x + w] != i] = 0.0
            np.clip(roi, 0.0, None, out=roi)
            flux = float(roi.sum())
            if flux <= 0:
                continue
            ys, xs = np.mgrid[0:h, 0:w]
            cx = x + float((roi * xs).sum()) / flux
            cy = y + float((roi * ys).sum()) / flux
            peak = float(frame[y:y + h, x:x + w].max())
            detections.append(Detection(cx, cy, int(area), peak, flux,
                                        (int(x), int(y), int(w), int(h)), score=flux))
        return detections
