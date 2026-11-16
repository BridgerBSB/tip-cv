# LINEAGE - tip-cv

> READ FIRST. Chronological record: what this project is now, what superseded what, what is
> deprecated (STILL-WIRED vs REMOVED). Newest on top. Results + data facts live in
> `docs/plans/2026-11-12-pitch-tipping-design.md`.

---

## 2026-11-15 - Click sheet v1 is the wrong test; v2 = learn-then-test with clips

**What changed:** Zac opened `output/blind/tip_blind_click_sheet.html` (38 same-at-bat still pairs) and could not
answer: "I can't tell which still shots are what ... unless you tell me the patterns." A one-line cue under each
question is not enough to teach a tell, and a still hides what a clip shows. His v1 answers are to be ignored.
- SUPERSEDED (STILL-WIRED): `scripts/click_sheet.py` v1 (stills only, cue line only). Keep its design pieces:
  same-at-bat pairs, 0.5 s before the first move, everything but the pitcher blurred (clock + TV count graphic
  leak the answer), key kept in `output/blind/key.json`, Perez as the no-tell control, `--score`.
- NEXT (v2, not built): per pitcher, 3 LABELED pairs first (short clips, tell circled, plain-words size), then the
  unlabeled pairs as short clips; Perez stays as the control.

## 2026-11-13/14 - Confirmation layer; MiLB data traps fixed

**What changed:**
- Wrong-shot check in `scripts/tell_scan.py::load` (tracked person off the pitcher's usual spot in that game =
  wrong TV shot, dropped). All results before `0502884` are SUPERSEDED by the `*_qc` runs (Tolbert's first video
  tracked a batter and a catcher).
- `scripts/team_summary.py` within-game label shuffle (`chance_p`) on the best of all measures scanned;
  `scripts/pitchtype_checks.py` (same at-bat, inside each strike count, tempo); `scripts/pickoff_checks.py`.
- Pull filter: SUPERSEDED `{menOn: ["Men_On"]}` (`matchup.splits.menOnBase` = bases AFTER the play; returned 2,008
  of 2,789 events with bases empty). Now `{preOn1b: true}` = previous play's `postOnFirst` (`dc3140e`).
  Bad pull set aside in `data/pulls/_wrong_menon/`.
- Pulls `tip2_howard/perez/verdugo` (no pitch types: MiLB public feed has none below AAA) SUPERSEDED by `tip5_*`
  (pitch type read from the signed-in analytics payload). Pose files are shared (keyed by play_id).
- Videos: half-speed single files SUPERSEDED by `<label>_normal.mp4` + `<label>_0.25x.mp4` (`scripts/pair_video.py
  --speeds`).

## 2026-11-12 - Project created

New repo (`BridgerBSB/tip-cv`, private). Pitch tipping from MLB per-play clips + YOLO11x pose: pickoffs to first
(baserunning) first, pitch-type tells (hitting advance) second. Proved on Mason Miller (Scott's elbow tell).
Sister of `command-cv` (borrowed its CDN download headers + Chrome pull approach).
