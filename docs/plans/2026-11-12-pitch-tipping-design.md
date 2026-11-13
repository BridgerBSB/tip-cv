# Pitch Tipping CV - design (2026-11-12)

## Goal

1. **Baserunning first.** With a runner on first, can we tell whether the pitcher is going home, throwing
   over or stepping off, and what gives it away?
2. **Pitch-type tips second.** Fastball vs everything; fastball vs offspeed vs breaking; pitch vs pitch.
3. **Who:** every opponent pitcher Houston's affiliates faced in 2026, plus our own pitchers.
4. **What counts:** a pitcher who gives it away most of the time, game after game. Not a one-night blip.

## Data (verified 2026-11-12)

| Source | What it gives | Note |
|---|---|---|
| Public game feed `api/v1.1/game/{pk}/feed/live` | every pitch, pickoff (with base) and step-off, with a `playId` | no sign-in |
| `api/v1/game/{pk}/{playId}/analytics?hydrate=analytics(video)` | clip URLs per camera angle + `trackedEvents` (BEGIN_OF_PLAY, PITCHER_FIRST_MOVEMENT, BALL_WAS_PITCHED timestamps) + pre-event count and runners | Okta sign-in (Chrome session); Zac clicks Sign In, Claude does the rest |
| `sporty-clips.mlb.com/*.mp4` | the clip | no sign-in; browser UA + Referer |

- **MiLB:** one angle, the TV broadcast. Opens on whatever shot TV was on and can cut away after the throw.
  Trim each clip to its CF camera segment before measuring.
- **MLB:** 9-12 angles per event (PITCHCAST, CENTERFIELD, BROADCAST, both PITCHER_SIDEVIEWs, both
  BATTER_SIDEVIEWs, HIGH_HOME, MID_HOME). Camera angles are 6.0 s, 1280x720, 60 fps.
- **No angle reaches 3-5 s before the set.** Every clip opens with the pitcher already set. MLB camera
  angles put release 4.20 s into the clip (IQR 4.18-4.23, measured by pose on Miller), i.e. ~3.1 s of set
  before MLB's first-movement stamp. The MiLB broadcast clip measured had release ~3.1 s in. Pre-set
  behaviour needs full-game video; not available from this API.
- **Clock check:** the leg lift found from the knee lands +0.09 s after MLB's PITCHER_FIRST_MOVEMENT stamp
  (IQR +0.09 to +0.13), so the two anchors agree.
- Pickoffs to first in 40 affiliate games: 235 throws by 104 pitchers (~6 a game). An opponent throws over
  only a few times against us, so pickoff reads need the pitcher's whole season pulled, not just vs Houston.

## Method

Per event: pose tracking on the pitcher in every frame from clip start to release. Body points: both
wrists, elbows, shoulders, hips, knees, ankles, head.

- **One clock:** time each frame relative to an anchor. `t_from_first_move` (MLB's PITCHER_FIRST_MOVEMENT)
  where present, else the leg lift found from the front knee. Name the origin in every column.
- **One scale:** offsets from the shoulder midpoint in torso lengths; "forward" = toward home in image.
- **Centered** on the pitcher's own mean for that game and batter side (Scott's method).
- **Split by base state** (empty / runner on 1st / runner on 2nd). Hold time moves 0.2-0.8 s by base state
  inside one start (Niesen), so pooling base states manufactures tells.
- **Score:** AUC per (body point x moment) for each comparison (pitch home vs pickoff; FB vs the rest; ...).

## A tell only counts if

1. Found on half the games and holds on the other half.
2. Shows up game by game (Scott's Miller tell: 22 of 23 games).
3. Survives a blind click sheet: Zac picks which clip of a pair is the offspeed / the pickoff without knowing.
4. The scan covers ~hundreds of (point x moment x base state) cells, so expect a few by luck; report how many
   cells were scanned beside every hit.

## Order

1. **Mason Miller (MLB 695243), runner on 2nd, slider vs four-seam.** Known positive: Scott measured elbow
   AUC .82, wrist .73 over 167 regular-season pitches, 0.2 s before the lift, PitchCast angle. Our pull: 210
   pitches (124 SL / 85 FF) in 29 games. If we reproduce it, the pipeline can see a real tell.
2. **Pickoffs to first**, opponent starters, whole-season pulls.
3. **Pitch-type tips** on the same pitchers.

## Results

**2026-11-12 - Mason Miller, runner on 2nd, slider vs four-seam (`scripts/miller_check.py`).**
- PitchCast (Scott's camera), 244 clips, 0 frames lost: throwing elbow further forward on sliders through
  the whole set (-1.5 to -0.3 s from MLB first movement). AUC .70 regular season (123 SL / 85 FF), .80
  postseason; .68 / .69 / .73 / .80 by period; 20 of 24 games; split by game .70 / .72; gap ~1.3 px (Scott
  ~1.2). At Scott's exact moment (0.2 s before the lift) only .57 vs his .82: on our clips the gap fades as
  he starts to move. The set window was chosen after seeing the map.
- 3B pitcher side view, same 244 pitches: nothing (elbow forward AUC .43, 9 of 24 games, best cell failed
  the split by game). Likely geometry: in the set Miller's chest faces the 3B camera, so an elbow further
  out in front of his body moves toward that camera, which a 2D image cannot show. PitchCast sees the same
  move side to side.
- Consequence: check every camera angle for each pitcher and use the one where the difference shows up.
  MiLB has only the TV angle.

**2026-11-13 - pickoffs to first, our three throw-over leaders (TV angle, runner on 1st, 2B and 3B open,
body-tracking first-move clock, set window -1.0 to -0.1 s; 26 body measures scanned per pitcher).**
- Dylan Howard (RHP, 46 pickoffs / 312 pitches, 23 games): glove hand further back toward his body while
  set before a pickoff - AUC .16 (i.e. pitch-side .84), further back in 13 of 14 games with both, halves by
  game .12 / .19, gap 0.10 torso lengths (~7-9 px, visible). Throwing hand higher: AUC .74, 15 of 18 games,
  halves .74 / .74. Both steady from 2.0 s before the move to the move.
- Ryan Verdugo (RHP, 31 / 262, 21 games): throwing hand higher while set before a pickoff - AUC .77, 15 of
  15 games, halves .76 / .79, gap 0.033 torso (~3 px). Glove hand higher too (.69, 9 of 12).
- Javier Perez (RHP, 44 / 265, 23 games): nothing that holds (best .63, halves .72 / .50).
- Not yet done: the blind look (click sheet), and the pitch clock reading at the first move.

## References

Vault: `bsb-brain/projects/pitch-tipping-cv-references.md` (Scott's Miller post, Niesen "Chasing Ghosts",
Jeremy Park ViTPose+ demo).
