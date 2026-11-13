"""Validity checks for a pickoff tell (2026-11-13).

1. Same at-bat: compare each pickoff with the pitches in the SAME at-bat (same batter, same runner), so the
   difference cannot come from when pickoffs happen (count, inning, who is on).
2. Usefulness for a runner: of the moments the measure says "pickoff-looking", how often was it a pickoff,
   vs his overall pickoff rate. Cut set on half the games, read on the other half.
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.tell_scan import GRID, SET_WIN, center, load

ap = argparse.ArgumentParser()
ap.add_argument("--pull", nargs="+", required=True); ap.add_argument("--pitcher", type=int, required=True)
ap.add_argument("--feature", required=True); ap.add_argument("--direction", choices=["low", "high"], required=True,
                help="which way the measure goes before a pickoff")
a = ap.parse_args()
ns = argparse.Namespace(pull=a.pull, angle="BROADCAST", pitcher=a.pitcher, runner="1b_only", event=["pitch", "pickoff"], anchor="pose")
meta, names, X, info = load(ns)
win = (GRID >= SET_WIN[0]) & (GRID <= SET_WIN[1])
v = np.nanmean(center(meta, X)[:, names.index(a.feature), win], axis=1)
if a.direction == "low": v = -v                      # larger = more pickoff-looking
meta["v"] = v; meta["pk"] = meta.event_type == "pickoff"

# 1. same at-bat
diffs = []
for _, g in meta.groupby(["game_pk", "at_bat_index"]):
    if g.pk.any() and (~g.pk).any():
        diffs.append(g.loc[g.pk, "v"].mean() - g.loc[~g.pk, "v"].mean())
diffs = np.array([d for d in diffs if np.isfinite(d)])
print(f"[same at-bat] {len(diffs)} at-bats with a pickoff and a pitch: pickoff-looking side in {int((diffs > 0).sum())} of {len(diffs)}")

# 2. runner's view, cut on odd games, read on even games (and the reverse)
games = sorted(meta.game_pk.unique()); odd = meta.game_pk.isin(games[0::2])
base = meta.pk.mean()
for lab, fit, test in (("cut on odd games, read on even", odd, ~odd), ("cut on even games, read on odd", ~odd, odd)):
    f, t = meta[fit & meta.v.notna()], meta[test & meta.v.notna()]
    cut = np.nanpercentile(f.v, 80)                                 # top 20% most pickoff-looking
    flag = t.v >= cut
    print(f"[runner's view, {lab}] flagged {flag.sum()} of {len(t)} moments: {t.pk[flag].mean():.0%} were pickoffs "
          f"vs {t.pk[~flag].mean():.0%} when not flagged (overall {base:.0%}); caught {t.pk[flag].sum()} of {t.pk.sum()} pickoffs")
