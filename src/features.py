"""Per-event body-position features on a common clock.

Clock origins (named in every column; see bsb-resources rule index-and-time-origins.md):
  frame_idx     sequential decode index in the clip (from pose .npz)
  t_rel_release seconds from release; release = frame of peak throwing-wrist speed (pose)
  t_from_fm     seconds from MLB's PITCHER_FIRST_MOVEMENT = t_rel_release + (ball_pitched_utc - first_move_utc)
                Both stamps come from the same MLB trackedEvents record; the clip itself carries no UTC.
Scale: offsets from the shoulder midpoint in that frame, divided by the pitcher's torso length (median
shoulder-mid to hip-mid distance while set). x is signed so + = toward home (the stride direction);
y is + = up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pose import KI

POINTS = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def _mid(k, a, b):
    return 0.5 * (k[:, KI[a], :2] + k[:, KI[b], :2])


LOWER = ["l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def pose_first_move(k: np.ndarray, idx: np.ndarray, fps: float, thresh: float = 0.12, hold: int = 3,
                    set_sec: float = 0.6) -> tuple[int | None, float]:
    """Sample index where the LOWER body first leaves its set position; None if it never does.

    Set position = median of the first `set_sec` seconds tracked. Moved = any hip/knee/ankle more than
    `thresh` torso lengths from its set spot for `hold` samples in a row. Lower body only, because a
    pitcher looks around with his head while set. Same rule for a pitch (leg lift) and a pickoff (step or
    spin), so the two line up on one clock. Matched MLB's PITCHER_FIRST_MOVEMENT within ~0.1 s on Miller
    via the knee version (2026-11-12).
    """
    sm = _mid(k, "l_shoulder", "r_shoulder"); hm = _mid(k, "l_hip", "r_hip")
    first = (idx - idx[0]) / fps <= set_sec
    if first.sum() < 4:
        return None, np.nan
    torso = np.nanmedian(np.linalg.norm(sm[first] - hm[first], axis=1))
    if not np.isfinite(torso) or torso < 20:
        return None, torso
    pts = k[:, [KI[p] for p in LOWER], :2]
    base = np.nanmedian(pts[first], axis=0)
    disp = np.nanmax(np.linalg.norm(pts - base, axis=2), axis=1) / torso
    moved = np.nan_to_num(disp, nan=0.0) > thresh
    for i in range(first.sum(), len(moved) - hold + 1):
        if moved[i:i + hold].all():
            return i, torso
    return None, torso


def event_features(z, pitch_hand: str, pitched_minus_fm: float | None, grid: np.ndarray, min_conf=0.3,
                   anchor: str = "mlb", fwd_sign: float | None = None):
    """Return (dict feature -> values on `grid`, info dict). None if unusable.

    anchor="mlb":  grid is t_from_fm (MLB PITCHER_FIRST_MOVEMENT, placed in the clip through release).
    anchor="pose": grid is t_from_move_pose (pose_first_move). Needs no MLB stamp; used for pickoffs,
                   whose throw is not stamped BALL_WAS_PITCHED.
    """
    idx = z["frame_idx"].astype(float)
    fps = float(z["fps"])
    k = z["kps"].copy()
    k[k[:, :, 2] < min_conf] = np.nan                   # drop low-confidence points
    throw = "r_wrist" if pitch_hand == "R" else "l_wrist"
    front_ankle = "l_ankle" if pitch_hand == "R" else "r_ankle"

    w = k[:, KI[throw], :2]
    sp = np.r_[np.nan, np.linalg.norm(np.diff(w, axis=0), axis=1)]
    if np.all(np.isnan(sp)):
        return None, {"why": "no throwing wrist"}
    rel = int(np.nanargmax(sp))
    t_rel = (idx - idx[rel]) / fps
    if anchor == "pose":
        mv, _ = pose_first_move(k, idx, fps)
        if mv is None:
            return None, {"why": "no first move found (pose)"}
        t_fm = (idx - idx[mv]) / fps
    else:
        if rel >= len(idx) - 1:
            return None, {"why": "peak wrist speed at the last frame (clip cut before release?)"}
        if pitched_minus_fm is None or not np.isfinite(pitched_minus_fm):
            return None, {"why": "no MLB first-movement stamp"}
        t_fm = t_rel + pitched_minus_fm

    sm = _mid(k, "l_shoulder", "r_shoulder")
    hm = _mid(k, "l_hip", "r_hip")
    set_mask = (t_fm > -2.0) & (t_fm < -0.3)
    if set_mask.sum() < 4:
        return None, {"why": "less than 0.3 s of set before the first move"}
    torso = np.nanmedian(np.linalg.norm(sm[set_mask] - hm[set_mask], axis=1))
    if not np.isfinite(torso) or torso < 20:
        return None, {"why": f"torso {torso}"}
    fa = k[:, KI[front_ankle], 0]
    dx = np.nanmedian(fa[(t_rel > -0.1) & (t_rel < 0.05)]) - np.nanmedian(fa[set_mask])
    own_sign = 1.0 if (np.isfinite(dx) and dx >= 0) else -1.0
    # A pickoff strides toward FIRST, not home, so its own stride cannot define "toward home". Callers
    # comparing pickoffs pass the pitcher's sign from his pitches (same camera) as fwd_sign.
    sign = own_sign if fwd_sign is None else fwd_sign

    feats = {}
    for p in POINTS:
        x = sign * (k[:, KI[p], 0] - sm[:, 0]) / torso
        y = -(k[:, KI[p], 1] - sm[:, 1]) / torso
        for name, v in ((p + "_fwd", x), (p + "_up", y)):
            ok = np.isfinite(v)
            feats[name] = np.interp(grid, t_fm[ok], v[ok], left=np.nan, right=np.nan) if ok.sum() > 5 else np.full(len(grid), np.nan)
    # pose-based leg lift, as an independent check on MLB's first-movement stamp
    fk = k[:, KI["l_knee" if pitch_hand == "R" else "r_knee"], 1]
    base = np.nanmedian(fk[set_mask])
    up = np.where((t_fm > -1.0) & ((base - fk) / torso > 0.15))[0]
    lift_t_fm = float(t_fm[up[0]]) if len(up) else np.nan
    return feats, {"why": "", "release_frame": int(idx[rel]), "release_t_clip": float(idx[rel] / fps),
                   "torso_px": float(torso), "fwd_sign": sign, "own_sign": own_sign, "lift_t_from_fm": lift_t_fm}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random pos > random neg), ties half. NaN-safe."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    allv = np.r_[pos, neg]
    ranks = pd.Series(allv).rank().to_numpy()
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
