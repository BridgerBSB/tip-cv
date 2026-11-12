"""Pitcher pose per frame from one clip.

Conventions (pinned - see bsb-resources rules index-and-time-origins.md):
  - Frames are read SEQUENTIALLY from 0 with cv2. Never cap.set(CAP_PROP_POS_FRAMES): a seek can land 0-2
    frames early. `frame_idx` below always means "sequential decode index, counted from clip start".
  - Keypoints are COCO-17 in full-frame pixel coordinates (x right, y down).

Tracking: find the pitcher on frame 0 (`pick_pitcher`), then on every later frame run pose on a crop
around the previous box and keep the detection that overlaps it most. Cropping gives the model more
pixels per pitcher than a full 1280x720 frame does.
"""
from __future__ import annotations

import cv2
import numpy as np

KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
      "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
KI = {k: i for i, k in enumerate(KP)}


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_frames(path: str, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(f)
    cap.release()
    return frames, fps


def _detect(model, img, imgsz, device):
    r = model(img, device=device, verbose=False, imgsz=imgsz)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 17, 3))
    boxes = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    kp = r.keypoints.data.cpu().numpy()        # (n, 17, 3) x, y, conf
    return boxes, conf, kp


def track_pitcher(model, frames, first_box, imgsz=640, device=0, crop_w=None):
    """Pose for the pitcher on every frame. Returns (n_frames, 17, 3) keypoints (NaN where lost) and boxes.

    The crop is a FIXED size (full frame height x crop_w, default square) re-centred on the pitcher each
    frame. A crop whose shape changes every frame made the ROCm backend rebuild kernels per shape
    (~0.22 s/frame measured 2026-11-12); a constant shape compiles once.
    """
    H, W = frames[0].shape[:2]
    crop_w = min(W, crop_w or H)
    n = len(frames)
    kps = np.full((n, 17, 3), np.nan)
    boxes = np.full((n, 4), np.nan)
    prev = np.array(first_box, float)
    for i, f in enumerate(frames):
        cx = 0.5 * (prev[0] + prev[2])
        cx0 = int(min(max(0, cx - crop_w / 2), W - crop_w)); cx1 = cx0 + crop_w
        cy0 = 0
        crop = f[:, cx0:cx1]
        b, c, k = _detect(model, crop, imgsz, device)
        if len(b) == 0:
            continue
        b = b + np.array([cx0, cy0, cx0, cy0])
        k[:, :, 0] += cx0
        k[:, :, 1] += cy0
        ious = np.array([iou(prev, bb) for bb in b])
        j = int(np.argmax(ious))
        if ious[j] < 0.2:        # lost him; keep the previous box, leave this frame NaN
            continue
        kps[i] = k[j]
        boxes[i] = b[j]
        prev = b[j]
    return kps, boxes
