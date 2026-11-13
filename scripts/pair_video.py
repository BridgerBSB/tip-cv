"""Side-by-side video: a pickoff and a pitch from the SAME at-bat, lined up on the first move, half speed.

    C:\\Users\\Owner\\.venvs\\cv-rocm\\Scripts\\python.exe scripts/pair_video.py --pull data/pulls/tip2_howard_2026.csv \
        --pitcher 703440 --hand l_wrist --name "Dylan Howard" --label howard --pairs 3

Rows = at-bats picked at random (seeded, not hand-picked) among at-bats with both a pickoff and a pitch.
Each panel: the pitcher's arms tracked; two rings show where the chosen hand typically sits while set -
red before a pitch, yellow before a pickoff (medians over the whole season, placed at his shoulders in
that frame). Encoded H.264 with the bundled ffmpeg (personal laptop only).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.fetch_clips import clip_path  # noqa: E402
from scripts.tell_scan import GRID, SET_WIN, load  # noqa: E402
from src.features import pose_first_move  # noqa: E402
from src.pose import KI, read_frames  # noqa: E402

ARMS = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
        ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist")]
PW, PH = 360, 480            # panel size
T0, T1 = -2.0, 0.6           # seconds around the first move
GRIDV = np.round(np.arange(T0, T1 + 1e-9, 1 / 30), 4)   # shared clock for every panel


def panel_frames(r, a, fwd_sign, med):
    z = np.load(os.path.join(ROOT, "output", "pose", str(a.pitcher), "BROADCAST", f"{int(r.game_pk)}_{r.play_id}.npz"))
    k = z["kps"].copy(); k[k[:, :, 2] < 0.3] = np.nan
    idx = z["frame_idx"].astype(int); fps = float(z["fps"])
    mv, torso = pose_first_move(k, idx.astype(float), fps)
    if mv is None:
        return None
    t = (idx - idx[mv]) / fps
    keep = np.where((t >= T0) & (t <= T1))[0]
    frames, _ = read_frames(clip_path(r, "BROADCAST"), max_frames=int(idx[keep[-1]]) + 1)
    box = np.nanmedian(z["boxes"][keep], axis=0)
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    h = 1.25 * (box[3] - box[1]); w = h * PW / PH
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x1, y1 = int(min(frames[0].shape[1], x0 + w)), int(min(frames[0].shape[0], y0 + h))
    out = []
    for s in keep:
        im = frames[idx[s]].copy(); kk = k[s]
        for p, q in ARMS:
            if np.isfinite(kk[KI[p], 0]) and np.isfinite(kk[KI[q], 0]):
                cv2.line(im, tuple(map(int, kk[KI[p], :2])), tuple(map(int, kk[KI[q], :2])), (255, 255, 255), 2, cv2.LINE_AA)
        sm = 0.5 * (kk[KI["l_shoulder"], :2] + kk[KI["r_shoulder"], :2])
        if t[s] < 0 and np.all(np.isfinite(sm)):
            for g, col in (("pitch", (60, 60, 235)), ("pickoff", (0, 220, 255))):
                fx, fy = med[g]
                c = (int(round(sm[0] + fwd_sign * fx * torso)), int(round(sm[1] - fy * torso)))
                cv2.circle(im, c, max(4, int(0.06 * torso)), col, 2, cv2.LINE_AA)
        if np.isfinite(kk[KI[a.hand], 0]):
            cv2.circle(im, tuple(map(int, kk[KI[a.hand], :2])), 3, (255, 255, 255), -1, cv2.LINE_AA)
        out.append(cv2.resize(im[y0:y1, x0:x1], (PW, PH), interpolation=cv2.INTER_CUBIC))
    return t[keep], out


def on_grid(panel, grid, lab):
    """Resample a panel onto the shared clock; before its clip starts, hold the first frame dimmed."""
    ts, fr = panel
    col = (0, 220, 255) if lab == "PICKOFF" else (120, 120, 255)
    out = []
    for g in grid:
        j = int(np.argmin(np.abs(ts - g)))
        im = fr[j].copy()
        early = g < ts[0] - 1 / 45
        if early:
            im = (im * 0.45).astype(np.uint8)
            cv2.putText(im, "clip starts later", (70, PH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.rectangle(im, (0, 0), (PW, 34), (25, 25, 25), -1)
        cv2.putText(im, f"{lab}  {g:+.2f}s from his move", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
        out.append(im)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", nargs="+", required=True)
    ap.add_argument("--pitcher", type=int, required=True)
    ap.add_argument("--hand", required=True, help="keypoint whose typical spots get rings, e.g. l_wrist")
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    ns = argparse.Namespace(pull=a.pull, angle="BROADCAST", pitcher=a.pitcher, runner="1b_only",
                            event=["pitch", "pickoff"], anchor="pose")
    meta, names, X, info = load(ns)
    win = (GRID >= SET_WIN[0]) & (GRID <= SET_WIN[1])
    S = np.nanmean(X[:, :, win], axis=2)
    pk = (meta.event_type == "pickoff").to_numpy()
    med = {g: (np.nanmedian(S[m, names.index(a.hand + "_fwd")]), np.nanmedian(S[m, names.index(a.hand + "_up")]))
           for g, m in (("pickoff", pk), ("pitch", ~pk))}
    fwd_sign = float(info.loc[info.why == "", "fwd_sign"].iloc[0])

    abs_ = [g for _, g in meta.groupby(["game_pk", "at_bat_index"]) if (g.event_type == "pickoff").any() and (g.event_type == "pitch").any()]
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(abs_))
    rows = []
    for i in order:
        g = abs_[i]
        p1 = panel_frames(g[g.event_type == "pickoff"].iloc[0], a, fwd_sign, med)
        p2 = panel_frames(g[g.event_type == "pitch"].iloc[0], a, fwd_sign, med)
        if p1 and p2:
            g1, g2 = on_grid(p1, GRIDV, "PICKOFF"), on_grid(p2, GRIDV, "PITCH")
            rows.append([np.hstack([g1[j], g2[j]]) for j in range(len(GRIDV))])
        if len(rows) == a.pairs:
            break
    n = len(GRIDV)
    title = np.full((80, 2 * PW, 3), 245, np.uint8)
    cv2.putText(title, f"{a.name}: pickoff vs pitch, same at-bat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(title, "rings = where this hand usually sits while set: red before a pitch, yellow before a pickoff", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    frames = [np.vstack([title] + [r[j] for r in rows]) for j in range(n)]
    import imageio_ffmpeg
    out = os.path.join(ROOT, "output", "video", f"{a.label}_pickoff_vs_pitch.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    H, W = frames[0].shape[:2]
    w = imageio_ffmpeg.write_frames(out, (W, H), fps=15, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=8,
                                    output_params=["-crf", "20"])
    w.send(None)
    for f in frames:
        w.send(np.ascontiguousarray(f[:, :, ::-1]))
    w.close()
    still = os.path.join(ROOT, "output", "video", f"{a.label}_pickoff_vs_pitch_still.jpg")
    cv2.imwrite(still, frames[int(np.argmin(np.abs(GRIDV + 0.5)))])
    print("wrote", out, f"{W}x{H}, {n} frames at 15 fps (half speed)", "| still", still)


if __name__ == "__main__":
    main()
