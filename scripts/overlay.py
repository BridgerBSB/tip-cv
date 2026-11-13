"""Draw a pitcher's median set position for two groups (e.g. before a pickoff vs before a pitch) on one frame.

    python scripts/overlay.py --pull data/pulls/tip2_howard_2026.csv --angle BROADCAST --pitcher 703440 \
        --runner 1b_only --event pitch pickoff --group-col event_type --a pickoff --b pitch --anchor pose \
        --label howard_pk1b --title "Dylan Howard, runner on first"

Each event's body points are averaged over the set window (SET_WIN before the first move), as offsets from
the shoulder midpoint in torso lengths. The median of each group is drawn on a frame from a group-B event,
placed at that frame's own shoulders and torso size. Arms, shoulders and hips only (where the tells live).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.fetch_clips import RUNNER_CHOICES, clip_path  # noqa: E402
from scripts.tell_scan import GRID, SET_WIN, load  # noqa: E402
from src.pose import read_frames  # noqa: E402

SEGS = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
        ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
        ("l_hip", "r_hip")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", nargs="+", required=True)
    ap.add_argument("--angle", default="BROADCAST")
    ap.add_argument("--pitcher", type=int, required=True)
    ap.add_argument("--runner", choices=RUNNER_CHOICES, default="any")
    ap.add_argument("--event", nargs="+", default=["pitch"])
    ap.add_argument("--group-col", default="event_type")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--anchor", choices=["mlb", "pose"], default="pose")
    ap.add_argument("--label", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    meta, names, X, info = load(a)
    win = (GRID >= SET_WIN[0]) & (GRID <= SET_WIN[1])
    S = np.nanmean(X[:, :, win], axis=2)                      # uncentered, per event
    isa = (meta[a.group_col] == a.a).to_numpy(); isb = (meta[a.group_col] == a.b).to_numpy()
    med = {g: {p: (np.nanmedian(S[m, names.index(p + "_fwd")]), np.nanmedian(S[m, names.index(p + "_up")]))
               for p in {q for s in SEGS for q in s}} for g, m in ((a.a, isa), (a.b, isb))}

    # frame: the group-B event closest to the group median, 0.5 s before its first move
    ib = np.where(isb)[0]
    dist = np.nansum((S[ib] - np.nanmedian(S[ib], axis=0)) ** 2, axis=1)
    r = meta.iloc[ib[int(np.argmin(dist))]]
    inf = info.set_index("play_id").loc[r.play_id]
    z = np.load(os.path.join(ROOT, "output", "pose", str(a.pitcher), a.angle, f"{int(r.game_pk)}_{r.play_id}.npz"))
    from src.features import pose_first_move
    k = z["kps"].copy(); k[k[:, :, 2] < 0.3] = np.nan
    idx = z["frame_idx"].astype(float); fps = float(z["fps"])
    mv, torso = pose_first_move(k, idx, fps)
    s = int(np.argmin(np.abs((idx - idx[mv]) / fps + 0.5)))
    frames, _ = read_frames(clip_path(r, a.angle), max_frames=int(idx[s]) + 1)
    im = frames[int(idx[s])].copy()
    from src.pose import KI
    sm = 0.5 * (k[s, KI["l_shoulder"], :2] + k[s, KI["r_shoulder"], :2])
    sign = float(inf.fwd_sign)

    def px(p, g):
        fx, fy = med[g][p]
        return int(round(sm[0] + sign * fx * torso)), int(round(sm[1] - fy * torso))

    col = {a.a: (0, 230, 255), a.b: (60, 60, 230)}        # BGR: yellow = A, red = B
    for g in (a.b, a.a):
        for p, q in SEGS:
            cv2.line(im, px(p, g), px(q, g), col[g], 2, cv2.LINE_AA)
        for p in ("l_wrist", "r_wrist", "l_elbow", "r_elbow"):
            cv2.circle(im, px(p, g), 4, col[g], -1, cv2.LINE_AA)
    xs = [px(p, g)[0] for g in med for p in med[g]]; ys = [px(p, g)[1] for g in med for p in med[g]]
    pad = int(1.4 * torso)
    x0, x1 = max(0, min(xs) - pad), min(im.shape[1], max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad // 2), min(im.shape[0], max(ys) + int(2.2 * torso))
    crop = cv2.resize(im[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    bar = np.full((90, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, a.title or a.label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(bar, f"yellow = before a {a.a} ({isa.sum()})   red = before a {a.b} ({isb.sum()})   median while set",
                (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    out = os.path.join(ROOT, "output", "scan", f"{a.label}_overlay.png")
    cv2.imwrite(out, np.vstack([bar, crop]))
    print("wrote", out, "| frame", r.game_pk, r.date, "torso px", round(torso, 1))


if __name__ == "__main__":
    main()
