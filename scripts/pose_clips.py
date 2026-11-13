"""Pose-track the pitcher through each clip; one .npz per clip (resumable: existing files are skipped).

    C:\\Users\\Owner\\.venvs\\cv-rocm\\Scripts\\python.exe scripts/pose_clips.py \
        --pull data/pulls/tip2_miller_2026_reg.csv data/pulls/tip2_miller_2026_post.csv \
        --angle PITCHCAST --runner on2b --event pitch --step 2 --max-sec 5.0

Output output/pose/<pitcher>/<ANGLE>/<game_pk>_<play_id>.npz with
  frame_idx  sequential decode index of each processed frame (step N = every Nth frame)
  kps        (n, 17, 3) COCO keypoints x, y, conf in full-frame pixels (NaN where the pitcher was lost)
  boxes      (n, 4)
  fps, width, height, pick_box (choice on the first tracked frame), pick_reason
  shot_start, shot_end  BROADCAST only: the CF shot kept (src/shots.py); other angles = 0, n_frames
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.fetch_clips import RUNNER_CHOICES, clip_path, filter_events, load_pulls  # noqa: E402
from src.pose import _detect, read_frames, track_pitcher  # noqa: E402
from src.shots import shot_bounds  # noqa: E402
from src.clips import download  # noqa: E402


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
    ap.add_argument("--runner", choices=RUNNER_CHOICES, default="any", help="see fetch_clips.filter_events")
    ap.add_argument("--event", nargs="+", default=["any"], help="pitch / pickoff / stepoff (several allowed)")
    ap.add_argument("--step", type=int, default=2, help="process every Nth frame (2 = 30 per second)")
    ap.add_argument("--max-sec", type=float, default=5.0, help="stop this far into the clip (MLB camera clips: release ~4.3 s in)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "yolo11x-pose.pt"))
    ap.add_argument("--stream", action="store_true",
                    help="download each clip that is not on disk, track it, then delete it (keeps the disk empty)")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)

    df = filter_events(load_pulls(a.pull), a.runner, a.event, a.pitcher)
    if a.stream:
        rows = [r for r in df.itertuples() if isinstance(getattr(r, "url_" + a.angle, None), str)]
    else:
        rows = [r for r in df.itertuples() if os.path.exists(clip_path(r, a.angle))]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(df)} events, {len(rows)} with a {a.angle} clip {'URL' if a.stream else 'on disk'}", flush=True)

    t0 = time.time(); n_done = 0
    for r in rows:
        out = os.path.join(ROOT, "output", "pose", str(int(r.pitcher_id)), a.angle, f"{int(r.game_pk)}_{r.play_id}.npz")
        if os.path.exists(out):
            continue
        fetched = False
        if a.stream and not os.path.exists(clip_path(r, a.angle)):
            try:
                download(getattr(r, "url_" + a.angle), clip_path(r, a.angle)); fetched = True
            except Exception as e:  # noqa: BLE001
                print("FETCH FAIL", r.game_pk, r.play_id, e); continue
        frames, fps = read_frames(clip_path(r, a.angle), max_frames=int(round(a.max_sec * 60)) + 2)
        if fetched:
            os.remove(clip_path(r, a.angle))      # frames are in memory; only the pose file is kept
        if not frames:
            print("EMPTY", out); continue
        s0, s1 = 0, len(frames)
        if a.angle == "BROADCAST":          # TV feed: keep only the CF shot the pitcher is set in
            s0, s1, _ = shot_bounds(frames, fps)
        idx = np.arange(s0, s1, a.step)
        if fps and fps > 0:
            idx = idx[idx / fps <= a.max_sec]
        if len(idx) < 20:
            print("SHORT SHOT", r.game_pk, r.play_id, s0, s1); continue
        box, why = pick_pitcher(model, frames[idx[0]])
        if box is None:
            print("NO PITCHER", r.game_pk, r.play_id, why); continue
        kps, boxes = track_pitcher(model, [frames[i] for i in idx], box)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out + ".part.npz", frame_idx=idx, kps=kps, boxes=boxes, fps=fps,
                            width=frames[0].shape[1], height=frames[0].shape[0], pick_box=box, pick_reason=why,
                            shot_start=s0, shot_end=s1)
        os.replace(out + ".part.npz", out)
        n_done += 1
        lost = int(np.isnan(kps[:, 0, 0]).sum())
        label = (r.pitch_name if isinstance(r.pitch_name, str) else r.event_type)[:10]
        print(f"{n_done}/{len(rows)} {r.game_pk} {label:10s} frames {len(idx)} shot {s0}-{s1} lost {lost}  {why}  "
              f"{(time.time() - t0) / n_done:.1f}s/clip", flush=True)
    print(f"DONE {n_done} new clips in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
