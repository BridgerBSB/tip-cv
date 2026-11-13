"""Find the CF camera shot inside a TV broadcast clip.

MiLB clips are the TV feed: they can open on a dugout or crowd shot and cut to the first-base camera
after a pickoff throw (seen 2026-11-12, Tacoma 814750). A hard cut changes the colour histogram of the
whole frame at once, so we cut where consecutive frames' HSV histograms differ by more than CUT_DIST
(Bhattacharyya distance) and keep the shot that contains `t_anchor` seconds - the pitcher is set then
in every clip measured so far.
"""
from __future__ import annotations

import cv2
import numpy as np

CUT_DIST = 0.35


def _hist(f):
    small = cv2.resize(f, (160, 90))
    h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def shot_bounds(frames, fps: float, t_anchor: float = 1.5) -> tuple[int, int, list[int]]:
    """Return (start, end_exclusive, cut_frames) of the shot containing t_anchor."""
    hs = [_hist(f) for f in frames]
    d = [0.0] + [cv2.compareHist(hs[i - 1], hs[i], cv2.HISTCMP_BHATTACHARYYA) for i in range(1, len(hs))]
    cuts = [i for i, v in enumerate(d) if v > CUT_DIST]
    anchor = min(len(frames) - 1, int(round(t_anchor * fps)))
    start = max([c for c in cuts if c <= anchor], default=0)
    end = min([c for c in cuts if c > anchor], default=len(frames))
    return start, end, cuts
