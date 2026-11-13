"""Scan every body point x moment for a difference between two groups of events.

Pitch type (MLB stamp clock):
    python scripts/tell_scan.py --pull data/pulls/tip2_miller_2026_reg.csv data/pulls/tip2_miller_2026_post.csv \
        --angle PITCHCAST --pitcher 695243 --runner on2b --event pitch \
        --group-col pitch_name --a Slider --b "Four-Seam Fastball" --label miller_on2b --scott
Pickoff vs pitch (pose clock, runner on first only):
    python scripts/tell_scan.py --pull data/pulls/tip2_howard_2026.csv --angle BROADCAST --pitcher 703440 \
        --runner 1b_only --event pitch pickoff --group-col event_type --a pickoff --b pitch --anchor pose \
        --label howard_pk1b

Centering: each feature minus the pitcher's mean for that (game, batter side) over every loaded event
(Scott's method). AUC = P(an A event sits higher than a B event); .50 = no difference.
Prints the set-window ranking (each body point averaged over SET_WIN before the first move, best first, with
game-by-game and a split by game), the full-scan best cell + its split by game, and clock checks.
Writes output/scan/<label>_auc.csv + .png and <label>_setwin.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.fetch_clips import RUNNER_CHOICES, filter_events, load_pulls  # noqa: E402
from src.features import auc, event_features  # noqa: E402

GRID = np.round(np.arange(-2.0, 0.4001, 1 / 30), 4)
SET_WIN = (-1.0, -0.1)       # seconds before the first move: the pitcher is set, nothing has started


def _stamp_gap(r):
    try:
        return (pd.Timestamp(r.ball_pitched_utc) - pd.Timestamp(r.first_move_utc)).total_seconds()
    except Exception:  # noqa: BLE001 - missing stamp -> None
        return None


def load(a):
    df = filter_events(load_pulls(a.pull), a.runner, a.event, a.pitcher)
    anchor = getattr(a, "anchor", "mlb")
    paths = {r.play_id: os.path.join(ROOT, "output", "pose", str(a.pitcher), a.angle, f"{int(r.game_pk)}_{r.play_id}.npz")
             for r in df.itertuples()}
    # "toward home" comes from the pitcher's PITCHES on this camera; a pickoff strides toward first
    fwd = None
    if anchor == "pose" or (df.event_type != "pitch").any():
        signs = []
        for r in df[df.event_type == "pitch"].itertuples():
            if os.path.exists(paths[r.play_id]):
                _, inf = event_features(np.load(paths[r.play_id]), r.pitch_hand, _stamp_gap(r), GRID, anchor=anchor)
                if inf.get("own_sign") is not None:
                    signs.append(inf["own_sign"])
        fwd = float(np.sign(np.sum(signs))) or 1.0 if signs else None
    rows, feats, infos = [], [], []
    for r in df.itertuples():
        f = paths[r.play_id]
        if not os.path.exists(f):
            infos.append({"play_id": r.play_id, "why": "no pose file"}); continue
        fe, info = event_features(np.load(f), r.pitch_hand, _stamp_gap(r), GRID, anchor=anchor, fwd_sign=fwd)
        info["play_id"] = r.play_id
        infos.append(info)
        if fe is None:
            continue
        rows.append(r); feats.append(fe)
    meta = pd.DataFrame([r._asdict() for r in rows])
    names = list(feats[0].keys()) if feats else []
    X = np.stack([np.stack([fe[n] for n in names]) for fe in feats]) if feats else np.zeros((0, 0, len(GRID)))
    return meta, names, X, pd.DataFrame(infos)


def center(meta, X):
    Xc = X.copy()
    for _, ix in meta.groupby(["game_pk", "bat_side"]).indices.items():
        Xc[ix] = X[ix] - np.nanmean(X[ix], axis=0, keepdims=True)
    return Xc


def auc_grid(Xc, ia, ib):
    g = np.full(Xc.shape[1:], np.nan)
    for f in range(Xc.shape[1]):
        for t in range(Xc.shape[2]):
            g[f, t] = auc(Xc[ia, f, t], Xc[ib, f, t])
    return g


def by_game(meta, v, isa, isb):
    wins = n = 0
    for _, gix in meta.groupby("game_pk").indices.items():
        va, vb = v[gix][isa[gix]], v[gix][isb[gix]]
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if len(va) and len(vb):
            n += 1; wins += int(va.mean() > vb.mean())
    return wins, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", nargs="+", required=True)
    ap.add_argument("--angle", default="PITCHCAST")
    ap.add_argument("--pitcher", type=int, required=True)
    ap.add_argument("--runner", choices=RUNNER_CHOICES, default="any")
    ap.add_argument("--event", nargs="+", default=["pitch"])
    ap.add_argument("--group-col", default="pitch_name", help="pitch_name or event_type")
    ap.add_argument("--a", required=True, help="group A value (e.g. Slider, pickoff)")
    ap.add_argument("--b", required=True, help="group B value (e.g. Four-Seam Fastball, pitch)")
    ap.add_argument("--anchor", choices=["mlb", "pose"], default="mlb")
    ap.add_argument("--scott", action="store_true", help="print the Miller elbow/wrist check")
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    meta, names, X, info = load(a)
    bad = info[info.why != ""]
    print(f"events usable {len(meta)}; dropped {len(bad)}: {bad.why.value_counts().to_dict()}")
    if not len(meta):
        return
    ok = info[info.why == ""]
    print(f"release at {ok.release_t_clip.median():.2f} s into clip (IQR {ok.release_t_clip.quantile(.25):.2f}-"
          f"{ok.release_t_clip.quantile(.75):.2f}); knee lift vs the first move: median {ok.lift_t_from_fm.median():+.2f} s")
    Xc = center(meta, X)
    isa = (meta[a.group_col] == a.a).to_numpy(); isb = (meta[a.group_col] == a.b).to_numpy()
    ia, ib = np.where(isa)[0], np.where(isb)[0]
    print(f"groups: {len(ia)} {a.a} vs {len(ib)} {a.b} in {meta.game_pk.nunique()} games")
    out_dir = os.path.join(ROOT, "output", "scan"); os.makedirs(out_dir, exist_ok=True)

    if a.scott:
        arm = "r" if meta.pitch_hand.iloc[0] == "R" else "l"
        lift = float(np.nanmedian(ok.lift_t_from_fm))
        t02 = int(np.argmin(np.abs(GRID - (lift - 0.2))))
        e, w = names.index(f"{arm}_elbow_fwd"), names.index(f"{arm}_wrist_fwd")
        print(f"Scott check at {GRID[t02]:+.2f} s: elbow fwd AUC {auc(Xc[ia, e, t02], Xc[ib, e, t02]):.2f}, "
              f"wrist fwd AUC {auc(Xc[ia, w, t02], Xc[ib, w, t02]):.2f}")

    # set-window ranking: every body point, averaged over the set window
    win = (GRID >= SET_WIN[0]) & (GRID <= SET_WIN[1])
    S = np.nanmean(Xc[:, :, win], axis=2)
    games = sorted(meta.game_pk.unique()); h1 = meta.game_pk.isin(games[0::2]).to_numpy()
    rows = []
    for f, n in enumerate(names):
        v = S[:, f]
        wn, gn = by_game(meta, v, isa, isb)
        rows.append({"feature": n, "auc": auc(v[isa], v[isb]),
                     "auc_odd_games": auc(v[isa & h1], v[isb & h1]), "auc_even_games": auc(v[isa & ~h1], v[isb & ~h1]),
                     "games_a_higher": wn, "games_both": gn,
                     "gap_torso": np.nanmedian(v[isa]) - np.nanmedian(v[isb])})
    sw = pd.DataFrame(rows)
    sw["strength"] = (sw.auc - 0.5).abs()
    sw = sw.sort_values("strength", ascending=False)
    sw.to_csv(os.path.join(out_dir, f"{a.label}_setwin.csv"), index=False)
    print(f"set window {SET_WIN[0]} to {SET_WIN[1]} s before the first move, {len(names)} body measures, top 6:")
    for r in sw.head(6).itertuples():
        print(f"  {r.feature:14s} AUC {r.auc:.2f}  (odd games {r.auc_odd_games:.2f}, even {r.auc_even_games:.2f})  "
              f"{a.a} higher in {r.games_a_higher} of {r.games_both} games  gap {r.gap_torso:+.3f} torso")

    g = auc_grid(Xc, ia, ib)
    pd.DataFrame(g, index=names, columns=[f"{t:+.3f}" for t in GRID]).to_csv(os.path.join(out_dir, f"{a.label}_auc.csv"))
    n_cells = int(np.isfinite(g).sum())
    if n_cells == 0:
        print("scan: not enough events in both groups"); return
    pre = GRID < 0
    gp = np.where(pre[None, :], g, np.nan)
    fb, tb = np.unravel_index(np.nanargmax(np.abs(gp - 0.5)), g.shape)
    g1 = auc_grid(Xc, np.where(isa & h1)[0], np.where(isb & h1)[0])
    g1p = np.where(pre[None, :], g1, np.nan)
    f1, t1 = np.unravel_index(np.nanargmax(np.abs(g1p - 0.5)), g1.shape)
    hold = auc(Xc[np.where(isa & ~h1)[0], f1, t1], Xc[np.where(isb & ~h1)[0], f1, t1])
    print(f"full scan before the first move: {int(np.isfinite(gp).sum())} cells; strongest {names[fb]} at {GRID[tb]:+.2f} s "
          f"AUC {g[fb, tb]:.2f}; best on odd games {names[f1]} at {GRID[t1]:+.2f} s ({g1[f1, t1]:.2f}) reads {hold:.2f} on even games")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=0.2, vmax=0.8,
                   extent=[GRID[0] - 1 / 60, GRID[-1] + 1 / 60, len(names) - 0.5, -0.5])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(SET_WIN[0], SET_WIN[1], color="k", alpha=0.06)
    clock = "MLB pitcher first movement" if a.anchor == "mlb" else "first move (body tracking)"
    ax.set_xlabel(f"seconds from {clock}")
    ax.set_title(f"{a.label}: {a.a} vs {a.b}, AUC by body point and moment ({len(ia)} vs {len(ib)} events)", fontsize=11)
    fig.colorbar(im, ax=ax, label=f"AUC ({a.a} higher)")
    fig.savefig(os.path.join(out_dir, f"{a.label}_auc.png"), dpi=110, bbox_inches="tight")
    print("wrote", os.path.join(out_dir, f"{a.label}_auc.png"))


if __name__ == "__main__":
    main()
