"""Confirmation checks for a pitch-type tell (2026-11-14).

1. Same at-bat: at-bats with both pitch groups -> in how many does the A pitch sit on the tell side of the B pitch.
2. Count: AUC inside each strike count (0 / 1 / 2 strikes) - pitch choice follows the count, so a set position
   that changes with the count could look like a tell. If the split holds inside every count, the count is not it.
3. Tempo: seconds from the first move to release (pose), by group - a leg-lift tell may only be a slower delivery.

    python scripts/pitchtype_checks.py --pull data/pulls/tip5_salem_2026_part*.csv --pitcher 838799 \
        --feature r_wrist_up --window set --a BB --b FB
"""
import argparse, glob, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.tell_scan import GRID, LIFT_WIN, SET_WIN, center, load
from src.features import auc, pose_first_move
from src.pose import KI

ap = argparse.ArgumentParser()
ap.add_argument("--pull", nargs="+", required=True); ap.add_argument("--pitcher", type=int, required=True)
ap.add_argument("--feature", required=True); ap.add_argument("--window", choices=["set", "lift"], required=True)
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--group-col", default="pitch_group"); ap.add_argument("--event", nargs="+", default=["pitch"])
a = ap.parse_args()
pull = [p for g in a.pull for p in glob.glob(g)]
ns = argparse.Namespace(pull=pull, angle="BROADCAST", pitcher=a.pitcher, runner="1b_only", event=a.event, anchor="pose")
meta, names, X, info = load(ns)
w = SET_WIN if a.window == "set" else LIFT_WIN
v = np.nanmean(center(meta, X)[:, names.index(a.feature), (GRID >= w[0]) & (GRID <= w[1])], axis=1)
isa = (meta[a.group_col] == a.a).to_numpy(); isb = (meta[a.group_col] == a.b).to_numpy()
overall = auc(v[isa], v[isb]); up = overall >= 0.5
print(f"overall AUC {overall:.2f} ({isa.sum()} {a.a} vs {isb.sum()} {a.b})")

d = []
for _, gix in meta.groupby(["game_pk", "at_bat_index"]).indices.items():
    va, vb = v[gix][isa[gix]], v[gix][isb[gix]]
    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
    if len(va) and len(vb):
        d.append(va.mean() - vb.mean())
d = np.array(d)
print(f"[same at-bat] {int(((d > 0) if up else (d < 0)).sum())} of {len(d)} at-bats on the tell side")

for k in (0, 1, 2):
    m = (meta.strikes_pre == k).to_numpy()
    print(f"[{k} strikes] AUC {auc(v[m & isa], v[m & isb]):.2f}  ({(m & isa).sum()} {a.a} vs {(m & isb).sum()} {a.b})")

tempo = []
for r in meta.itertuples():
    z = np.load(os.path.join(ROOT, "output", "pose", str(a.pitcher), "BROADCAST", f"{int(r.game_pk)}_{r.play_id}.npz"))
    k = z["kps"].copy(); k[k[:, :, 2] < 0.3] = np.nan
    idx = z["frame_idx"].astype(float); fps = float(z["fps"])
    mv, _ = pose_first_move(k, idx, fps)
    wr = k[:, KI["r_wrist" if r.pitch_hand == "R" else "l_wrist"], :2]
    sp = np.r_[np.nan, np.linalg.norm(np.diff(wr, axis=0), axis=1)]
    rel = int(np.nanargmax(sp)) if np.isfinite(sp).any() else None
    tempo.append((idx[rel] - idx[mv]) / fps if (mv is not None and rel is not None and rel > mv) else np.nan)
tempo = np.array(tempo)
print(f"[tempo] first move to release: {a.a} median {np.nanmedian(tempo[isa]):.2f} s vs {a.b} {np.nanmedian(tempo[isb]):.2f} s "
      f"(AUC {auc(tempo[isa], tempo[isb]):.2f})")
