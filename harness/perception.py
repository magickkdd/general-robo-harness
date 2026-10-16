"""Object-level 3D perception from RGBD, classical OpenCV only (no NN, no privileged state).

Pipeline per object per camera: HSV inRange -> open morphology -> largest connected
component -> median depth (CPU renderer returns integer millimeters) -> pinhole
backprojection (K, cam2world_gl, GL/CV axis convention - validated to ~0.5-1.5 cm).
CPU cost < 10 ms per frame at 128x128.

HSV thresholds are calibrated for the fixed lavapipe renderer lighting; recalibrate
if the render backend changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_PIXELS = 8
WRIST_MIN_PIXELS = 200
MAX_DEPTH_M = 2.0
MAX_OBJECT_Z_M = 0.30   # tabletop objects live below this; rejects robot-body colors

# OpenCV HSV ranges (H 0-179, S/V 0-255), empirically calibrated on lavapipe
# renders: pure-colored objects have S>=200 while the wood table (wrist view)
# sits at S~156 and floors at S<10. "red" wraps around hue 0.
_HSV_RANGES = {
    "red":   [((0, 200, 150), (8, 255, 255)), ((172, 200, 150), (179, 255, 255))],
    "green": [((45, 200, 120), (75, 255, 255))],
    "blue":  [((100, 140, 100), (125, 255, 255))],
}

# task -> {object name: color key}. PegInsertionSide is NOT registered: its peg
# renders near-white (S~12), inseparable from the white panda body by color;
# it returns once segmentation upgrades to a learned model.
TASK_OBJECTS = {
    "PickCube-v1": {"cube": "red"},
    "StackCube-v1": {"cube_red": "red", "cube_green": "green"},
    "PushCube-v1": {"cube": "blue"},
    "PullCube-v1": {"cube": "blue"},
}

_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class ObjectReading:
    name: str
    position: tuple | None      # world-frame (x, y, z) meters; None = not visible
    camera: str                 # "base_camera" | "hand_camera" | "none"
    mask_pixels: int
    mask: np.ndarray | None = field(default=None, repr=False)  # chosen-camera mask
    # last-known-position memory (filled by the adapter, not by perceive())
    last_seen_pos: tuple | None = None   # world-frame position when last visible
    last_seen_age: int | None = None     # steps since that sighting


def _object_reading(rgb: np.ndarray, depth_mm: np.ndarray, color: str,
                    sensor_param: dict, cam: str) -> ObjectReading | None:
    """Segment + localize the largest color component that lives near the table.

    Every component is backprojected; components whose world z sits above
    MAX_OBJECT_Z_M are robot-body colors, not tabletop objects.
    """
    depth_mm = np.asarray(depth_mm)
    if depth_mm.ndim == 4:      # (1, H, W, 1) batched sensor output
        depth_mm = depth_mm[0, :, :, 0]
    depth = depth_mm.astype(np.float32) / 1000.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in _HSV_RANGES[color]:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    # no morphology here: the S>=200 thresholds are already tight, and a 3x3
    # opening erases 8-12 px objects (the smallest legitimate detections)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    K = np.asarray(sensor_param[cam]["intrinsic_cv"])[0]
    G2W = np.asarray(sensor_param[cam]["cam2world_gl"])[0]
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_PIXELS:
            continue
        mask_i = (labels == i)
        ys, xs = np.nonzero(mask_i)
        d = depth[ys, xs]
        ok = (d > 1e-4) & (d < MAX_DEPTH_M)
        if not ok.any():
            continue
        u, v, dep = (float(np.median(xs[ok])), float(np.median(ys[ok])),
                     float(np.median(d[ok])))
        x = (u - K[0, 2]) * dep / K[0, 0]
        y = (v - K[1, 2]) * dep / K[1, 1]
        p_w = (G2W @ _GL2CV @ np.array([x, y, dep, 1.0]))[:3]
        if p_w[2] > MAX_OBJECT_Z_M:
            continue
        if best is None or area > best[1]:
            best = (ObjectReading(name="", position=(float(p_w[0]), float(p_w[1]), float(p_w[2])),
                                  camera=cam, mask_pixels=area,
                                  mask=mask_i.astype(np.uint8)), area)
    return best[0] if best else None


def perceive(rgb_base: np.ndarray, depth_base: np.ndarray,
             rgb_wrist: np.ndarray | None, depth_wrist: np.ndarray | None,
             sensor_param: dict, task: str, wrist_uid: str = "hand_camera") -> list[ObjectReading]:
    """Localize all registered objects of the task. Fusion: wrist wins when its
    mask is large (close range), else base, else whichever is visible."""
    cams = {"base_camera": (rgb_base, depth_base)}
    if rgb_wrist is not None and depth_wrist is not None:
        cams[wrist_uid] = (rgb_wrist, depth_wrist)
    readings = []
    for name, color in TASK_OBJECTS.get(task, {}).items():
        cands = []
        for cam, (rgb, depth_mm) in cams.items():
            r = _object_reading(rgb, depth_mm, color, sensor_param, cam)
            if r is not None:
                cands.append(r)
        wrist = next((c for c in cands if c.camera == wrist_uid), None)
        base = next((c for c in cands if c.camera == "base_camera"), None)
        if wrist is not None and wrist.mask_pixels >= WRIST_MIN_PIXELS:
            chosen = wrist
        elif base is not None:
            chosen = base
        elif wrist is not None:
            chosen = wrist
        else:
            chosen = None
        if chosen is None:
            readings.append(ObjectReading(name, None, "none", 0))
        else:
            readings.append(ObjectReading(name, chosen.position, chosen.camera,
                                          chosen.mask_pixels, mask=chosen.mask))
    return readings


def format_object_lines(readings: list[ObjectReading], ee_pos) -> list[str]:
    """Human/VLM-readable lines appended to the state text."""
    lines = []
    ee = np.asarray(ee_pos, dtype=np.float64)
    for r in readings:
        if r.position is None:
            if r.last_seen_pos is not None:
                pos = np.asarray(r.last_seen_pos)
                delta = pos - ee
                dist = float(np.linalg.norm(delta))
                lines.append(
                    f"object {r.name}: not visible now; last seen {r.last_seen_age} step(s) ago "
                    f"at pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m "
                    f"delta_to_ee=({delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f}) m, "
                    f"dist={dist:.3f} m (assume static unless it moves)")
            else:
                lines.append(f"object {r.name}: not visible in either camera")
            continue
        pos = np.asarray(r.position)
        delta = pos - ee
        dist = float(np.linalg.norm(delta))
        tag = ("ground truth" if r.camera == "gt"
               else f"seen by {r.camera}, {r.mask_pixels} px")
        lines.append(
            f"object {r.name}: center pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m "
            f"[{tag}] "
            f"delta_to_ee=({delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f}) m, "
            f"dist={dist:.3f} m")
    return lines


def to_json(readings: list[ObjectReading]) -> dict:
    out = {}
    for r in readings:
        out[r.name] = {
            "pos": [round(float(v), 4) for v in r.position] if r.position else None,
            "camera": r.camera,
            "mask_pixels": r.mask_pixels,
        }
        if r.position is not None:
            out[r.name]["pos"] = [round(float(v), 4) for v in r.position]
        elif r.last_seen_pos is not None:
            out[r.name]["last_seen_pos"] = [round(float(v), 4) for v in r.last_seen_pos]
            out[r.name]["last_seen_age_steps"] = r.last_seen_age
    return out


def draw_overlays(rgb: np.ndarray, readings: list[ObjectReading]) -> np.ndarray:
    """Draw mask contours + centroid cross on a copy of the input image."""
    img = rgb.copy()
    for r in readings:
        if r.mask is None:
            continue
        contours, _ = cv2.findContours(r.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        ys, xs = np.nonzero(r.mask)
        cv2.drawMarker(img, (int(xs.mean()), int(ys.mean())), (255, 255, 0),
                       cv2.MARKER_CROSS, 9, 1)
    return img
