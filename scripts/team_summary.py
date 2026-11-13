"""One table for a staff: for each pitcher, pickoff vs pitch and fastball vs breaking / offspeed, runner on
first only, body-tracking clock. For each comparison: the body measure that separates the two most over the
set (-1.0 to -0.1 s before the first move), how it reads on each half of his games, in how many games it
pointed the same way, and (pickoffs) the same at-bat check.

    python scripts/team_summary.py --pull data/pulls/tip2_salem_2026_part*.csv --label salem

Nothing here is a pass/fail gate: every row prints its counts. Writes output/scan/<label>_summary.csv.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.tell_scan import GRID, LIFT_WIN, SET_WIN, by_game, center, load  # noqa: E402
from src.features import auc  # noqa: E402

# (title, column, A, B, events, windows). Pickoffs: only the set counts (a runner must read it before he moves).
# Pitch types: the set AND the leg lift (a hitter can still use what he sees during the lift).
COMPARISONS = [("pickoff vs pitch", "event_type", "pickoff", "pitch", ["pitch", "pickoff"], [("set", SET_WIN)]),
               ("breaking vs fastball", "pitch_group", "BB", "FB", ["pitch"], [("set", SET_WIN), ("leg lift", LIFT_WIN)]),
               ("offspeed vs fastball", "pitch_group", "OS", "FB", ["pitch"], [("set", SET_WIN), ("leg lift", LIFT_WIN)])]


def one(pull, pitcher, comp):
    title, col, va, vb, events, windows = comp
    ns = argparse.Namespace(pull=pull, angle="BROADCAST", pitcher=pitcher, runner="1b_only", event=events, anchor="pose")
    meta, names, X, info = load(ns)
    if not len(meta):
        return None
    isa = (meta[col] == va).to_numpy(); isb = (meta[col] == vb).to_numpy()
    if isa.sum() < 2 or isb.sum() < 2:
        return {"comparison": title, "n_a": int(isa.sum()), "n_b": int(isb.sum())}
    Xc = center(meta, X)
    games = sorted(meta.game_pk.unique()); h1 = meta.game_pk.isin(games[0::2]).to_numpy()
    best = None
    for wname, (w0, w1) in windows:
        win = (GRID >= w0) & (GRID <= w1)
        S = np.nanmean(Xc[:, :, win], axis=2)
        for f, n in enumerate(names):
            v = S[:, f]
            a_ = auc(v[isa], v[isb])
            if np.isfinite(a_) and (best is None or abs(a_ - 0.5) > abs(best[1] - 0.5)):
                best = (n, a_, f, wname, v)
    if best is None:
        return {"comparison": title, "n_a": int(isa.sum()), "n_b": int(isb.sum())}
    n, a_, f, wname, v = best
    wn, gn = by_game(meta, v, isa, isb)
    row = {"comparison": title, "n_a": int(isa.sum()), "n_b": int(isb.sum()), "games": int(meta.game_pk.nunique()),
           "best_measure": n, "window": wname, "auc": a_, "auc_odd_games": auc(v[isa & h1], v[isb & h1]),
           "auc_even_games": auc(v[isa & ~h1], v[isb & ~h1]),
           "games_same_way": (wn if a_ >= 0.5 else gn - wn), "games_both": gn,
           "gap_torso": float(np.nanmedian(v[isa]) - np.nanmedian(v[isb])),
           "measures_scanned": len(names) * len(windows)}
    if title == "pickoff vs pitch":
        d = []
        for _, gix in meta.groupby(["game_pk", "at_bat_index"]).indices.items():
            ga, gb = v[gix][isa[gix]], v[gix][isb[gix]]
            ga, gb = ga[np.isfinite(ga)], gb[np.isfinite(gb)]
            if len(ga) and len(gb):
                d.append(ga.mean() - gb.mean())
        d = np.array(d)
        row["same_ab_same_way"] = int(((d > 0) if a_ >= 0.5 else (d < 0)).sum()); row["same_ab"] = len(d)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--pitchers", nargs="*", type=int, default=None, help="default: everyone with pose files")
    a = ap.parse_args()
    pull = [p for g in a.pull for p in glob.glob(g)]
    df = pd.concat([pd.read_csv(p) for p in pull]).drop_duplicates("play_id")
    names = {}
    ids = a.pitchers or sorted(int(d) for d in os.listdir(os.path.join(ROOT, "output", "pose"))
                               if d.isdigit() and int(d) in set(df.pitcher_id))
    rows = []
    for pid in ids:
        for comp in COMPARISONS:
            r = one(pull, pid, comp)
            if r:
                r["pitcher_id"] = pid; r["hand"] = df.loc[df.pitcher_id == pid, "pitch_hand"].iloc[0]
                rows.append(r)
                if "auc" in r:
                    print(f"{pid} {r['hand']} {r['comparison']:22s} {r['n_a']:3d} vs {r['n_b']:3d} | {r['best_measure']:13s} {r['window']:8s} "
                          f"AUC {r['auc']:.2f} (halves {r['auc_odd_games']:.2f} / {r['auc_even_games']:.2f}) "
                          f"same way in {r['games_same_way']} of {r['games_both']} games"
                          + (f" | same at-bat {r['same_ab_same_way']} of {r['same_ab']}" if "same_ab" in r else ""), flush=True)
    out = os.path.join(ROOT, "output", "scan", f"{a.label}_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
