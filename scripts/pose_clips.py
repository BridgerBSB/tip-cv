"""Pose-track the pitcher through each clip; one .npz per clip (resumable: existing files are skipped).

    C:\\Users\\Owner\\.venvs\\cv-rocm\\Scripts\\python.exe scripts/pose_clips.py \
        --pull data/pulls/tip2_miller_2026_reg.csv data/pulls/tip2_miller_2026_post.csv \
        --angle PITCHCAST --runner on2b --event pitch --step 2 --max-sec 5.0

Output output/pose/<pitcher>/<ANGLE>/<game_pk>_<play_id>.npz with
  frame_idx  sequential decode index of each processed frame (step N = every Nth frame)
  kps        (n, 17, 3) COCO keypoints x, y, conf in full-frame pixels (NaN where the pitcher was lost)
  boxes      (n, 4)
  fps, width, height, pick_box (frame-0 choice), pick_reason
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.fetch_clips import clip_path, load_pulls  # noqa: E402
from src.pose import _detect, read_frames, track_pitcher  # noqa: E402


def pick_pitcher(model, frame, device=0):
    """Tallest confident person not cut off by the frame bottom (a runner near the camera often is)."""
    H = frame.shape[0]
    b, c, _ = _detect(model, frame, 1280, device)
    ok = [(bb[3] - bb[1], i) for i, bb in enumerate(b) if c[i] >= 0.5 and bb[3] < H - 4]
    if not ok:
        return None, "no person"
    h, i = max(ok)
    return b[i], f"tallest of {len(ok)} (h {h:.0f}px)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", nargs="+", required=True)
    ap.add_argument("--angle", default="PITCHCAST")
    ap.add_argument("--pitcher", type=int, default=0)
    ap.add_argument("--runner", choices=["any", "on1b", "on2b", "empty"], default="any")
    ap.add_argument("--event", choices=["any", "pitch", "pickoff", "stepoff"], default="any")
    ap.add_argument("--step", type=int, default=2, help="process every Nth frame (2 = 30 per second)")
    ap.add_argument("--max-sec", type=float, default=5.0, help="stop this far into the clip (MLB camera clips: release ~4.3 s in)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "yolo11x-pose.pt"))
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)

    df = load_pulls(a.pull)
    if a.pitcher:
        df = df[df.pitcher_id == a.pitcher]
    if a.event != "any":
        df = df[df.event_type == a.event]
    if a.runner == "empty":
        df = df[~df.on1b.fillna(False) & ~df.on2b.fillna(False) & ~df.on3b.fillna(False)]
    elif a.runner != "any":
        df = df[df[a.runner].fillna(False).astype(bool)]
    rows = [r for r in df.itertuples() if os.path.exists(clip_path(r, a.angle))]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(df)} events, {len(rows)} with a {a.angle} clip on disk", flush=True)

    t0 = time.time(); n_done = 0
    for r in rows:
        out = os.path.join(ROOT, "output", "pose", str(int(r.pitcher_id)), a.angle, f"{int(r.game_pk)}_{r.play_id}.npz")
        if os.path.exists(out):
            continue
        frames, fps = read_frames(clip_path(r, a.angle), max_frames=int(round(a.max_sec * 60)) + 2)
        if not frames:
            print("EMPTY", out); continue
        idx = np.arange(0, len(frames), a.step)
        if fps and fps > 0:
            idx = idx[idx / fps <= a.max_sec]
        box, why = pick_pitcher(model, frames[0])
        if box is None:
            print("NO PITCHER", r.game_pk, r.play_id, why); continue
        kps, boxes = track_pitcher(model, [frames[i] for i in idx], box)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out + ".part.npz", frame_idx=idx, kps=kps, boxes=boxes, fps=fps,
                            width=frames[0].shape[1], height=frames[0].shape[0], pick_box=box, pick_reason=why)
        os.replace(out + ".part.npz", out)
        n_done += 1
        lost = int(np.isnan(kps[:, 0, 0]).sum())
        print(f"{n_done}/{len(rows)} {r.game_pk} {r.pitch_name[:10]:10s} frames {len(idx)} lost {lost}  {why}  "
              f"{(time.time() - t0) / n_done:.1f}s/clip", flush=True)
    print(f"DONE {n_done} new clips in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
