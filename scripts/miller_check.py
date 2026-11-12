"""Miller check: throwing elbow forward over the set window (-1.5 to -0.3 s from MLB first movement),
slider vs four-seam, runner on 2nd, PitchCast. Window chosen AFTER seeing the AUC map (2026-11-12).
Result 2026-11-12: reg AUC .70, post .80, 20 of 24 games, split-half by game .70 / .72, gap 1.3 px."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Owner\tip-cv"); sys.argv = ["x", "--pull", "data/pulls/tip2_miller_2026_reg.csv", "data/pulls/tip2_miller_2026_post.csv",
  "--angle", "PITCHCAST", "--pitcher", "695243", "--runner", "on2b", "--a", "Slider", "--b", "Four-Seam Fastball", "--label", "x"]
import argparse, warnings; warnings.filterwarnings("ignore")
from scripts.tell_scan import load, center, GRID
from src.features import auc
ns = argparse.Namespace(pull=sys.argv[2:4], angle="PITCHCAST", pitcher=695243, runner="on2b", event="pitch")
meta, names, X, info = load(ns); Xc = center(meta, X)
e = names.index("r_elbow_fwd"); w = names.index("r_wrist_fwd")
win = (GRID >= -1.5) & (GRID <= -0.3)
meta["elb_set"] = np.nanmean(Xc[:, e][:, win], axis=1); meta["wr_set"] = np.nanmean(Xc[:, w][:, win], axis=1)
meta["elb_scott"] = Xc[:, e, int(np.argmin(np.abs(GRID + 0.10)))]
m = meta[meta.pitch_name.isin(["Slider", "Four-Seam Fastball"])].copy()
m["sl"] = m.pitch_name == "Slider"
m["month"] = pd.to_datetime(m.date).dt.month
m["period"] = np.where(m.game_type == "P", "Post", np.where(m.month <= 5, "Apr-May", np.where(m.month <= 7, "Jun-Jul", "Aug-Sep")))
def A(d, c): return auc(d.loc[d.sl, c].to_numpy(), d.loc[~d.sl, c].to_numpy())
only2 = m[(m.on1b != True) & (m.on3b != True)]
for lab, d in (("reg, any with 2B occupied", m[m.game_type == "R"]), ("reg, runner on 2nd ONLY", only2[only2.game_type == "R"]), ("post", m[m.game_type == "P"])):
    print(f"{lab:28s} n {d.sl.sum()} SL / {(~d.sl).sum()} FF | elbow fwd over set window AUC {A(d,'elb_set'):.2f} | wrist {A(d,'wr_set'):.2f} | elbow at Scott's moment {A(d,'elb_scott'):.2f}")
for per in ["Apr-May", "Jun-Jul", "Aug-Sep", "Post"]:
    d = m[m.period == per]; print(f"  {per:8s} n {len(d):3d}  elbow fwd (set window) AUC {A(d,'elb_set'):.2f}")
wins = n = 0
for g, d in m[m.game_type == "R"].groupby("game_pk"):
    if d.sl.any() and (~d.sl).any():
        n += 1; wins += int(d.loc[d.sl, "elb_set"].mean() > d.loc[~d.sl, "elb_set"].mean())
print(f"game by game (reg, set window): slider elbow further forward in {wins} of {n} games")
games = sorted(m.game_pk.unique()); h1 = m.game_pk.isin(games[0::2])
print(f"split-half by game: odd games AUC {A(m[h1],'elb_set'):.2f}, even games AUC {A(m[~h1],'elb_set'):.2f}")
# size of the tell in pixels and torso lengths
tor = info.loc[info.why == "", "torso_px"].median()
diff = m.loc[m.sl, "elb_set"].median() - m.loc[~m.sl, "elb_set"].median()
print(f"median gap {diff:.3f} torso lengths = {diff*tor:.1f} px (torso {tor:.0f} px)")
