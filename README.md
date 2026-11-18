# tip-cv - pitch tipping from video

Find what a pitcher does differently before a pickoff vs a pitch home, and before one pitch type vs
another, from MLB/MiLB per-play video and body tracking. Baserunning first (pickoffs to first), pitch-type
tips second. Opponents our affiliates faced, plus our own pitchers.

- Design and verified data facts: `docs/plans/2026-11-12-pitch-tipping-design.md`
- Clip lists: `browser/tip_pull.js` (run in Chrome on statsapi.mlb.com, signed in) -> `data/pulls/`
- Clips: `python scripts/fetch_clips.py --pull <csv...> --angles PITCHCAST --runner on2b --event pitch`
- Pose: `src/pose.py` (YOLO11 pose, sequential frame decode, pitcher tracked by crop + overlap)

## Setup on a new machine

1. Python 3.12. Install torch for your hardware first (https://pytorch.org/get-started/locally/), then
   `pip install -r requirements.txt`.
2. Pose model: `weights/yolo11x-pose.pt` (not in git). Ultralytics downloads it on first use, or copy it.
3. Pose runs on a GPU (`device=0`). On a CPU it is very slow; the AMD laptop does ~8.6 s per clip
   including the download.
4. Clip lists need a signed-in MLB session in Chrome (`browser/tip_pull.js`). Each person signs in as
   themselves.
5. Video and pose data (`data/`, `output/`) stay on each machine and are rebuilt from the pulls. They are
   never committed.

Astros work laptops block unapproved programs, including the ffmpeg.exe that `imageio-ffmpeg` ships, so
`scripts/pair_video.py` only runs on a personal laptop. Everything else is plain Python.

## Working on it

Branch off `main`, push the branch, open a pull request. `LINEAGE.md` says what replaced what and why;
add an entry when a finding or method changes.

Zac's machine: `C:\Users\Owner\.venvs\cv-rocm\Scripts\python.exe` (torch ROCm, AMD GPU = device 0).
Sister project: command-cv (same clip source; borrowed its CDN download headers).
