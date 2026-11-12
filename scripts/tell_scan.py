"""Scan every body point x moment for a split between two groups of events (e.g. slider vs four-seam).

    python scripts/tell_scan.py --pull data/pulls/tip2_miller_2026_reg.csv data/pulls/tip2_miller_2026_post.csv \
        --angle PITCHCAST --pitcher 695243 --runner on2b --a Slider --b "Four-Seam Fastball" --label miller_on2b

Centering: each feature minus the pitcher's mean for that (game, batter side) over every loaded event
(Scott's method). AUC = P(an A event sits higher than a B event); .50 = no split.
Prints: Scott's check (throwing elbow / wrist forward at 0.2 s before first movement), game-by-game count,
a split-half check of the best cell, and the clock sanity checks. Writes output/scan/<label>_auc.csv + .png.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.fetch_clips import load_pulls  # noqa: E402
from src.features import auc, event_features  # noqa: E402

GRID = np.round(np.arange(-2.0, 0.4001, 1 / 30), 4)


def load(a):
    df = load_pulls(a.pull)
    df = df[df.pitcher_id == a.pitcher]
    if a.event != "any":
        df = df[df.event_type == a.event]
    if a.runner == "empty":
        df = df[~df.on1b.fillna(False) & ~df.on2b.fillna(False) & ~df.on3b.fillna(False)]
    elif a.runner != "any":
        df = df[df[a.runner].fillna(False).astype(bool)]
    rows, feats, infos = [], [], []
    for r in df.itertuples():
        f = os.path.join(ROOT, "output", "pose", str(a.pitcher), a.angle, f"{int(r.game_pk)}_{r.play_id}.npz")
        if not os.path.exists(f):
            infos.append({"play_id": r.play_id, "why": "no pose file"}); continue
        try:
            dt = (pd.Timestamp(r.ball_pitched_utc) - pd.Timestamp(r.first_move_utc)).total_seconds()
        except Exception:  # noqa: BLE001
            dt = None
        fe, info = event_features(np.load(f), r.pitch_hand, dt, GRID)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", nargs="+", required=True)
    ap.add_argument("--angle", default="PITCHCAST")
    ap.add_argument("--pitcher", type=int, required=True)
    ap.add_argument("--runner", default="any")
    ap.add_argument("--event", default="pitch")
    ap.add_argument("--a", required=True, help="pitch_name for group A")
    ap.add_argument("--b", required=True, help="pitch_name for group B")
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    meta, names, X, info = load(a)
    bad = info[info.why != ""]
    print(f"events usable {len(meta)}; dropped {len(bad)}: {bad.why.value_counts().to_dict()}")
    if not len(meta):
        return
    # clock sanity: release position in the clip, and pose leg lift vs MLB first movement
    ok = info[info.why == ""]
    print(f"release at {ok.release_t_clip.median():.2f} s into clip (IQR {ok.release_t_clip.quantile(.25):.2f}-"
          f"{ok.release_t_clip.quantile(.75):.2f}); pose leg lift vs MLB first move: median "
          f"{ok.lift_t_from_fm.median():+.2f} s (IQR {ok.lift_t_from_fm.quantile(.25):+.2f} to {ok.lift_t_from_fm.quantile(.75):+.2f})")

    Xc = center(meta, X)
    out_dir = os.path.join(ROOT, "output", "scan"); os.makedirs(out_dir, exist_ok=True)
    hand = meta.pitch_hand.iloc[0]
    arm = "r" if hand == "R" else "l"
    lift = float(np.nanmedian(info.loc[info.why == "", "lift_t_from_fm"]))
    # Scott measured 0.2 s before the LIFT; our pose lift sits `lift` s after MLB's first-movement stamp
    t02 = int(np.argmin(np.abs(GRID - (lift - 0.2))))
    print(f"Scott's moment = 0.2 s before the lift = {GRID[t02]:+.2f} s from MLB first movement")

    for gt, label in (("R", "regular season"), ("P", "postseason"), (None, "all")):
        m = meta if gt is None else meta[meta.game_type == gt]
        ia = m.index[m.pitch_name == a.a].to_numpy(); ib = m.index[m.pitch_name == a.b].to_numpy()
        if len(ia) < 3 or len(ib) < 3:
            continue
        e = names.index(f"{arm}_elbow_fwd"); w = names.index(f"{arm}_wrist_fwd")
        print(f"[{label}] {len(ia)} {a.a} vs {len(ib)} {a.b}: throwing elbow fwd AUC {auc(Xc[ia, e, t02], Xc[ib, e, t02]):.2f}, "
              f"wrist fwd AUC {auc(Xc[ia, w, t02], Xc[ib, w, t02]):.2f}  (0.2 s before first movement)")
        if gt == "R":
            wins = n = 0
            for _, gix in m.groupby("game_pk").indices.items():
                gi = m.index[gix]
                va = Xc[[i for i in gi if meta.pitch_name[i] == a.a], e, t02]
                vb = Xc[[i for i in gi if meta.pitch_name[i] == a.b], e, t02]
                va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                if len(va) and len(vb):
                    n += 1; wins += int(va.mean() > vb.mean())
            print(f"[{label}] game by game: {a.a} elbow further forward in {wins} of {n} games with both pitches")

    ia = meta.index[meta.pitch_name == a.a].to_numpy(); ib = meta.index[meta.pitch_name == a.b].to_numpy()
    g = auc_grid(Xc, ia, ib)
    pd.DataFrame(g, index=names, columns=[f"{t:+.3f}" for t in GRID]).to_csv(os.path.join(out_dir, f"{a.label}_auc.csv"))
    n_cells = int(np.isfinite(g).sum())
    if n_cells == 0:
        print("scan: not enough events in both groups yet"); return
    f_best, t_best = np.unravel_index(np.nanargmax(np.abs(g - 0.5)), g.shape)
    print(f"scan: {n_cells} cells (body point x direction x moment); strongest {names[f_best]} at "
          f"{GRID[t_best]:+.2f} s, AUC {g[f_best, t_best]:.2f}")

    # split-half by game: find the best cell on odd games, read it on even games
    games = sorted(meta.game_pk.unique())
    h1 = meta.game_pk.isin(games[0::2]).to_numpy(); h2 = ~h1
    g1 = auc_grid(Xc, np.where(h1 & (meta.pitch_name == a.a))[0], np.where(h1 & (meta.pitch_name == a.b))[0])
    fb, tb = np.unravel_index(np.nanargmax(np.abs(g1 - 0.5)), g1.shape)
    hold = auc(Xc[np.where(h2 & (meta.pitch_name == a.a))[0], fb, tb], Xc[np.where(h2 & (meta.pitch_name == a.b))[0], fb, tb])
    print(f"split-half: best on half 1 = {names[fb]} at {GRID[tb]:+.2f} s (AUC {g1[fb, tb]:.2f}); on half 2 it reads {hold:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=0.2, vmax=0.8,
                   extent=[GRID[0] - 1 / 60, GRID[-1] + 1 / 60, len(names) - 0.5, -0.5])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="k", lw=1); ax.axvline(-0.2, color="k", lw=0.8, ls="--")
    ax.set_xlabel("seconds from MLB pitcher first movement")
    ax.set_title(f"{a.label}: {a.a} vs {a.b}, AUC by body point and moment ({len(ia)} vs {len(ib)} events)", fontsize=11)
    fig.colorbar(im, ax=ax, label=f"AUC ({a.a} higher)")
    fig.savefig(os.path.join(out_dir, f"{a.label}_auc.png"), dpi=110, bbox_inches="tight")
    print("wrote", os.path.join(out_dir, f"{a.label}_auc.png"))


if __name__ == "__main__":
    main()
