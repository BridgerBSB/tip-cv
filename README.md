# tip-cv - pitch tipping from video

Find what a pitcher does differently before a pickoff vs a pitch home, and before one pitch type vs
another, from MLB/MiLB per-play video and body tracking. Baserunning first (pickoffs to first), pitch-type
tips second. Opponents our affiliates faced, plus our own pitchers.

- Design and verified data facts: `docs/plans/2026-11-12-pitch-tipping-design.md`
- Clip lists: `browser/tip_pull.js` (run in Chrome on statsapi.mlb.com, signed in) -> `data/pulls/`
- Clips: `python scripts/fetch_clips.py --pull <csv...> --angles PITCHCAST --runner on2b --event pitch`
- Pose: `src/pose.py` (YOLO11 pose, sequential frame decode, pitcher tracked by crop + overlap)

Python env for video/pose: `C:\Users\Owner\.venvs\cv-rocm\Scripts\python.exe` (torch ROCm, GPU = device 0).
Sister project: `BridgerBSB/command-cv` (same clip source; borrowed its CDN download headers).
