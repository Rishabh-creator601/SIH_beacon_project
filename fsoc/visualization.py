"""OpenCV overlay drawing: HUD, detections, track estimate and world minimap."""

from collections import deque

import cv2
import numpy as np

from .tracker import TrackState

STATE_COLORS = {                       # BGR
    TrackState.SEARCH: (0, 0, 255),
    TrackState.ACQUIRING: (0, 165, 255),
    TrackState.TRACKING: (0, 220, 0),
    TrackState.COASTING: (0, 220, 220),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _dashed_rect(img, rect, color, dash=6):
    x0, y0, x1, y1 = rect
    for x in range(x0, x1, 2 * dash):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, 1)
        cv2.line(img, (x, y1 - 1), (min(x + dash, x1), y1 - 1), color, 1)
    for y in range(y0, y1, 2 * dash):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, 1)
        cv2.line(img, (x1 - 1, y), (x1 - 1, min(y + dash, y1)), color, 1)


class Overlay:
    def __init__(self, sim, trail_len=300):
        self.sim = sim
        self.show_truth = False
        self.show_hud = True          # text panel (the Qt app shows stats itself)
        self.show_minimap = True
        self.show_scores = True
        self.show_roi = True
        self.trail = deque(maxlen=trail_len)

    def draw(self, res, paused=False):
        sim, cam = self.sim, self.sim.camera
        img = cv2.cvtColor(res.frame, cv2.COLOR_GRAY2BGR)
        color = STATE_COLORS[res.track.state]
        cx, cy = int(round(cam.cx)), int(round(cam.cy))

        # Boresight crosshair and fine-pointing handover zone.
        cv2.line(img, (cx - 15, cy), (cx - 5, cy), (255, 255, 0), 1)
        cv2.line(img, (cx + 5, cy), (cx + 15, cy), (255, 255, 0), 1)
        cv2.line(img, (cx, cy - 15), (cx, cy - 5), (255, 255, 0), 1)
        cv2.line(img, (cx, cy + 5), (cx, cy + 15), (255, 255, 0), 1)
        cv2.circle(img, (cx, cy), int(sim.metrics.handover_px), (255, 255, 0), 1)

        # All candidates, and the associated one (state colour). With
        # identification on, each candidate shows its P(beacon): green =
        # accepted as beacon-like, red = rejected, yellow = undecided / still
        # collecting history ("?").
        tracker = sim.tracker
        for d in res.detections:
            x, y, w, h = d.bbox
            box = (0, 200, 255)
            if tracker.use_identity and self.show_scores:
                if not d.extra.get("ready"):
                    label = "?"
                else:
                    label = f"{d.score:.2f}"
                    if d.score >= tracker.accept:
                        box = (0, 220, 0)
                    elif d.score <= tracker.reject:
                        box = (0, 0, 255)
                cv2.putText(img, label, (x + w + 4, y - 2), FONT, 0.38, box, 1, cv2.LINE_AA)
            cv2.rectangle(img, (x - 2, y - 2), (x + w + 1, y + h + 1), box, 1)
        sel = res.track.selected
        if sel is not None:
            x, y = int(round(sel.x)), int(round(sel.y))
            cv2.rectangle(img, (x - 12, y - 12), (x + 12, y + 12), color, 2)

        # Kalman estimate and velocity vector (0.5 s ahead).
        if res.track.estimate is not None:
            az, el, vaz, vel = res.track.estimate
            ex, ey = cam.world_to_pixel(az, el, res.pose)
            fx, fy = cam.world_to_pixel(az + 0.5 * vaz, el + 0.5 * vel, res.pose)
            p0 = (int(ex), int(ey))
            cv2.drawMarker(img, p0, (255, 0, 255), cv2.MARKER_CROSS, 10, 1)
            cv2.arrowedLine(img, p0, (int(fx), int(fy)), (255, 0, 255), 1, tipLength=0.2)

        if self.show_truth:
            # Where the beacon really is in this image (rendered from the
            # vibrating line of sight), and decoys in orange.
            for t in sim.scene.targets:
                tx, ty = cam.world_to_pixel(t.az, t.el, res.los)
                cv2.circle(img, (int(tx), int(ty)), 7,
                           (0, 0, 255) if t.is_beacon else (0, 140, 255), 1)

        roi = sim.detector.last_roi
        if self.show_roi and roi is not None and res.track.estimate is not None:
            _dashed_rect(img, roi, (255, 170, 60))
        if self.show_hud:
            self._hud(img, res, color, paused)
        if self.show_minimap:
            self._minimap(img, res)
        return img

    def _hud(self, img, res, color, paused):
        live = self.sim.metrics.live()
        acq = live.get("acq_time")
        ret = live.get("retention")
        status = f"STATE: {res.track.state.value}"
        if live.get("occluded"):
            status += "  (beacon occluded)"
        if paused:
            status += "  [PAUSED]"
        lines = [
            (status, color),
            (f"t = {res.t:6.2f} s   FPS = {live.get('fps', 0):5.1f}", (255, 255, 255)),
            (f"Error = {live.get('err_px', 0):6.1f} px  ({live.get('err_mrad', 0):6.2f} mrad)",
             (255, 255, 255)),
            (f"Acq time = {'--' if acq is None else f'{acq:.2f} s'}", (255, 255, 255)),
            (f"Lock retention = {'--' if ret is None else f'{ret:.1f} %'}"
             f"   losses = {live.get('lock_losses', 0)}", (255, 255, 255)),
            (f"Proc = {live.get('proc_ms', 0):5.2f} ms   blobs = {len(res.detections)}",
             (255, 255, 255)),
            (f"ID: {self.sim.id_method.upper()}   rejected = {self.sim.tracker.rejections}",
             (255, 255, 255)),
        ]
        h = 20 * len(lines) + 10
        roi = img[0:h, 0:380]
        roi[:] = (roi * 0.4).astype(np.uint8)
        for i, (text, c) in enumerate(lines):
            cv2.putText(img, text, (8, 20 + 20 * i), FONT, 0.5, c, 1, cv2.LINE_AA)
        cv2.putText(img, "[space] pause  [t] truth  [r] reset  [q] quit",
                    (8, img.shape[0] - 8), FONT, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    def _minimap(self, img, res, size=(200, 134)):
        """Whole-world overview: camera footprint (box), beacon (red), trail."""
        sim, cam = self.sim, self.sim.camera
        az_min, az_max, el_min, el_max = sim.scene.bounds
        mw, mh = size
        x0, y0 = img.shape[1] - mw - 8, img.shape[0] - mh - 8

        def to_map(az, el):
            return (int(x0 + (az - az_min) / (az_max - az_min) * (mw - 1)),
                    int(y0 + (el_max - el) / (el_max - el_min) * (mh - 1)))

        roi = img[y0:y0 + mh, x0:x0 + mw]
        roi[:] = (roi * 0.3).astype(np.uint8)
        cv2.rectangle(img, (x0, y0), (x0 + mw - 1, y0 + mh - 1), (120, 120, 120), 1)

        clouds = sim.disturb.clouds
        if clouds is not None:
            sx = (mw - 1) / (az_max - az_min)
            for az, el, size in zip(clouds.az, clouds.el, clouds.size):
                cv2.circle(img, to_map(az, el), max(int(size * sx), 2), (110, 110, 110), 1)

        b = sim.scene.beacon
        self.trail.append(to_map(b.az, b.el))
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array(self.trail, np.int32)], False, (0, 0, 140), 1)
        for t in sim.scene.targets:
            cv2.circle(img, to_map(t.az, t.el), 3,
                       (0, 0, 255) if t.is_beacon else (0, 200, 255), -1)

        paz, pel = res.pose
        p1 = to_map(paz - cam.fov_x / 2, pel + cam.fov_y / 2)
        p2 = to_map(paz + cam.fov_x / 2, pel - cam.fov_y / 2)
        cv2.rectangle(img, p1, p2, STATE_COLORS[res.track.state], 1)

        sp = sim.controller.search_point
        if res.track.state == TrackState.SEARCH and sp is not None:
            cv2.circle(img, to_map(*sp), 2, (255, 255, 255), -1)
        cv2.putText(img, "WORLD", (x0 + 4, y0 + 12), FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
