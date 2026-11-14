"""Side-by-side video: an A event next to a B event from the SAME at-bat, lined up on the first move, half speed.

Pickoff vs pitch:
    C:\\Users\\Owner\\.venvs\\cv-rocm\\Scripts\\python.exe scripts/pair_video.py --pull data/pulls/tip5_howard_2026.csv \
        --pitcher 703440 --hand l_wrist --name "Dylan Howard (RHP), glove hand" --label howard_pickoff
Breaking ball vs fastball (hitting advance):
    ... --group-col pitch_group --a BB --b FB --a-label "BREAKING" --b-label "FASTBALL" --event pitch --hand r_wrist

Rows = at-bats picked at random (seeded, not hand-picked) among at-bats with both an A and a B event.
Each panel: the pitcher's arms tracked; during the chosen window two rings show where `--hand` typically sits
(yellow = A, red = B; medians over his season, placed at his shoulders in that frame). Clips deleted after
tracking are re-downloaded (only the ones shown). Encoded H.264 with the bundled ffmpeg (personal laptop only).
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
from scripts.tell_scan import GRID, LIFT_WIN, SET_WIN, load  # noqa: E402
from src.clips import download  # noqa: E402
from src.features import pose_first_move  # noqa: E402
from src.pose import KI, read_frames  # noqa: E402

ARMS = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
        ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist")]
LEGS = [("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"), ("l_hip", "l_knee"),
        ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle")]
PW, PH = 360, 480            # panel size
T0, T1 = -2.0, 0.6           # seconds around the first move
GRIDV = np.round(np.arange(T0, T1 + 1e-9, 1 / 30), 4)   # shared clock for every panel
COL_A, COL_B = (0, 220, 255), (60, 60, 235)              # BGR: yellow, red


def panel_frames(r, a, fwd_sign, med, win):
    z = np.load(os.path.join(ROOT, "output", "pose", str(a.pitcher), "BROADCAST", f"{int(r.game_pk)}_{r.play_id}.npz"))
    k = z["kps"].copy(); k[k[:, :, 2] < 0.3] = np.nan
    idx = z["frame_idx"].astype(int); fps = float(z["fps"])
    mv, torso = pose_first_move(k, idx.astype(float), fps)
    if mv is None:
        return None
    t = (idx - idx[mv]) / fps
    keep = np.where((t >= T0) & (t <= T1))[0]
    path = clip_path(r, "BROADCAST")
    if not os.path.exists(path):
        download(r.url_BROADCAST, path)
    frames, _ = read_frames(path, max_frames=int(idx[keep[-1]]) + 1)
    box = np.nanmedian(z["boxes"][keep], axis=0)
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    h = 1.25 * (box[3] - box[1]); w = h * PW / PH
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x1, y1 = int(min(frames[0].shape[1], x0 + w)), int(min(frames[0].shape[0], y0 + h))
    segs = ARMS + (LEGS if a.legs else [])
    out = []
    for s in keep:
        im = frames[idx[s]].copy(); kk = k[s]
        for p, q in segs:
            if np.isfinite(kk[KI[p], 0]) and np.isfinite(kk[KI[q], 0]):
                cv2.line(im, tuple(map(int, kk[KI[p], :2])), tuple(map(int, kk[KI[q], :2])), (255, 255, 255), 2, cv2.LINE_AA)
        sm = 0.5 * (kk[KI["l_shoulder"], :2] + kk[KI["r_shoulder"], :2])
        if win[0] <= t[s] <= win[1] and np.all(np.isfinite(sm)) and not a.no_rings:
            for g, col in (("B", COL_B), ("A", COL_A)):
                fx, fy = med[g]
                c = (int(round(sm[0] + fwd_sign * fx * torso)), int(round(sm[1] - fy * torso)))
                cv2.circle(im, c, max(4, int(0.06 * torso)), col, 2, cv2.LINE_AA)
        if np.isfinite(kk[KI[a.hand], 0]):
            cv2.circle(im, tuple(map(int, kk[KI[a.hand], :2])), 3, (255, 255, 255), -1, cv2.LINE_AA)
        out.append(cv2.resize(im[y0:y1, x0:x1], (PW, PH), interpolation=cv2.INTER_CUBIC))
    return t[keep], out


def on_grid(panel, grid, lab, col):
    """Resample a panel onto the shared clock; before its clip starts, hold the first frame dimmed."""
    ts, fr = panel
    out = []
    for g in grid:
        j = int(np.argmin(np.abs(ts - g)))
        im = fr[j].copy()
        if g < ts[0] - 1 / 45:
            im = (im * 0.45).astype(np.uint8)
            cv2.putText(im, "clip starts later", (70, PH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.rectangle(im, (0, 0), (PW, 34), (25, 25, 25), -1)
        cv2.putText(im, f"{lab}  {g:+.2f}s from his move", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        out.append(im)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", nargs="+", required=True)
    ap.add_argument("--pitcher", type=int, required=True)
    ap.add_argument("--hand", required=True, help="keypoint whose typical spots get rings, e.g. l_wrist")
    ap.add_argument("--group-col", default="event_type")
    ap.add_argument("--a", default="pickoff"); ap.add_argument("--b", default="pitch")
    ap.add_argument("--a-label", default="PICKOFF"); ap.add_argument("--b-label", default="PITCH")
    ap.add_argument("--event", nargs="+", default=["pitch", "pickoff"])
    ap.add_argument("--window", choices=["set", "lift"], default="set", help="where the rings are drawn")
    ap.add_argument("--legs", action="store_true", help="also draw hips / legs (for leg-lift tells)")
    ap.add_argument("--no-rings", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    ns = argparse.Namespace(pull=a.pull, angle="BROADCAST", pitcher=a.pitcher, runner="1b_only", event=a.event, anchor="pose")
    meta, names, X, info = load(ns)
    win = SET_WIN if a.window == "set" else LIFT_WIN
    m = (GRID >= win[0]) & (GRID <= win[1])
    S = np.nanmean(X[:, :, m], axis=2)
    isa = (meta[a.group_col] == a.a).to_numpy(); isb = (meta[a.group_col] == a.b).to_numpy()
    med = {g: (np.nanmedian(S[mm, names.index(a.hand + "_fwd")]), np.nanmedian(S[mm, names.index(a.hand + "_up")]))
           for g, mm in (("A", isa), ("B", isb))}
    fwd_sign = float(info.loc[info.why == "", "fwd_sign"].iloc[0])

    abs_ = [g for _, g in meta.groupby(["game_pk", "at_bat_index"]) if (g[a.group_col] == a.a).any() and (g[a.group_col] == a.b).any()]
    rng = np.random.default_rng(a.seed)
    rows = []
    for i in rng.permutation(len(abs_)):
        g = abs_[i]
        p1 = panel_frames(g[g[a.group_col] == a.a].iloc[0], a, fwd_sign, med, win)
        p2 = panel_frames(g[g[a.group_col] == a.b].iloc[0], a, fwd_sign, med, win)
        if p1 and p2:
            g1, g2 = on_grid(p1, GRIDV, a.a_label, COL_A), on_grid(p2, GRIDV, a.b_label, COL_B)
            rows.append([np.hstack([g1[j], g2[j]]) for j in range(len(GRIDV))])
        if len(rows) == a.pairs:
            break
    n = len(GRIDV)
    where = "while set" if a.window == "set" else "during the leg lift"
    title = np.full((80, 2 * PW, 3), 245, np.uint8)
    cv2.putText(title, f"{a.name}: {a.a_label.lower()} vs {a.b_label.lower()}, same at-bat", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    if not a.no_rings:
        cv2.putText(title, f"rings = where this point usually sits {where}: yellow {a.a_label.lower()}, red {a.b_label.lower()}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    frames = [np.vstack([title] + [r[j] for r in rows]) for j in range(n)]
    import imageio_ffmpeg
    out = os.path.join(ROOT, "output", "video", f"{a.label}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    H, W = frames[0].shape[:2]
    w = imageio_ffmpeg.write_frames(out, (W, H), fps=15, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=8,
                                    output_params=["-crf", "20"])
    w.send(None)
    for f in frames:
        w.send(np.ascontiguousarray(f[:, :, ::-1]))
    w.close()
    still_t = -0.5 if a.window == "set" else 0.35
    still = os.path.join(ROOT, "output", "video", f"{a.label}_still.jpg")
    cv2.imwrite(still, frames[int(np.argmin(np.abs(GRIDV - still_t)))])
    print("wrote", out, f"{W}x{H}, {n} frames at 15 fps (half speed), {len(rows)} at-bats", "| still", still)


if __name__ == "__main__":
    main()
