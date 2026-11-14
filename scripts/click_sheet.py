"""Blind click sheet: can a person see the tell without the tracking?

For each pitcher: pairs of stills from the SAME at-bat, both 0.5 s before his first move (nothing after the move,
so the pitch or throw cannot give it away), sides randomized. The viewer picks which one is the target (pickoff /
breaking ball / changeup). Perez (no tell found) is a blind control: his score shows what guessing looks like.
The sheet is one self-contained HTML; the answer key goes to output/blind/key.json, never into the sheet.
Answers download as Downloads/tip_blind_answers.json; score with --score.

    python scripts/click_sheet.py               # build
    python scripts/click_sheet.py --score <answers.json>
"""
from __future__ import annotations

import argparse, base64, glob, json, os, sys
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.fetch_clips import clip_path  # noqa: E402
from scripts.tell_scan import load  # noqa: E402
from src.clips import download  # noqa: E402
from src.features import pose_first_move  # noqa: E402
from src.pose import read_frames  # noqa: E402

SALEM = sorted(glob.glob(os.path.join(ROOT, "data", "pulls", "tip5_salem_2026_part*.csv")))
SPECS = [  # (key, name, pitcher, pull, group col, target, other, events, question, cue, n)
    ("tolbert", "Cole Tolbert (RHP)", 824613, SALEM, "event_type", "pickoff", "pitch", ["pitch", "pickoff"],
     "Which one is about to throw over to first?", "Look at how far he has turned into his set.", 10),
    ("blum", "Harry Blum (RHP)", 838799, SALEM, "pitch_group", "BB", "FB", ["pitch"],
     "Which one is the breaking ball?", "Look at how low his throwing hand sits at the glove.", 10),
    ("walker", "Ethan Walker (LHP)", 833080, SALEM, "pitch_group", "OS", "FB", ["pitch"],
     "Which one is the changeup?", "Look at how low his throwing hand sits at the glove.", 10),
    ("perez", "Javier Perez (RHP) - control", 823649, [os.path.join(ROOT, "data", "pulls", "tip5_perez_2026.csv")],
     "event_type", "pickoff", "pitch", ["pitch", "pickoff"],
     "Which one is about to throw over to first?", "Look at his glove hand.", 8),
]
OUT_DIR = os.path.join(ROOT, "output", "blind")


def still(r, pitcher, t_before=0.5):
    z = np.load(os.path.join(ROOT, "output", "pose", str(pitcher), "BROADCAST", f"{int(r.game_pk)}_{r.play_id}.npz"))
    k = z["kps"].copy(); k[k[:, :, 2] < 0.3] = np.nan
    idx = z["frame_idx"].astype(float); fps = float(z["fps"])
    mv, _ = pose_first_move(k, idx, fps)
    if mv is None:
        return None
    target = idx[mv] - t_before * fps
    if target < idx[0]:
        return None
    path = clip_path(r, "BROADCAST"); fetched = False
    if not os.path.exists(path):
        download(r.url_BROADCAST, path); fetched = True
    fr, _ = read_frames(path, max_frames=int(target) + 1)
    if fetched:
        os.remove(path)
    im = fr[int(target)].copy()
    b = np.nanmedian(z["boxes"][: max(3, len(z["boxes"]) // 3)], axis=0)
    # Blur everything but the pitcher: the pitch clock and the TV count graphic sit beside him and would give the
    # answer away by timing or count instead of by his body (seen in the first build, 2026-11-14).
    bw, bh = b[2] - b[0], b[3] - b[1]
    keep = np.zeros(im.shape[:2], np.uint8)
    keep[int(max(0, b[1] - 0.08 * bh)):int(min(im.shape[0], b[3] + 0.04 * bh)),
         int(max(0, b[0] - 0.18 * bw)):int(min(im.shape[1], b[2] + 0.18 * bw))] = 1
    im = np.where(keep[:, :, None] == 1, im, cv2.GaussianBlur(im, (0, 0), 25))
    cx, cy, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.3 * (b[3] - b[1])
    w = h * 0.72
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x1, y1 = int(min(im.shape[1], x0 + w)), int(min(im.shape[0], y0 + h))
    crop = cv2.resize(im[y0:y1, x0:x1], (330, int(330 * (y1 - y0) / max(1, x1 - x0))), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def build():
    rng = np.random.default_rng(99)
    trials, key = [], {}
    for keyname, name, pid, pull, col, tgt, oth, events, q, cue, n in SPECS:
        ns = argparse.Namespace(pull=pull, angle="BROADCAST", pitcher=pid, runner="1b_only", event=events, anchor="pose")
        meta, _, _, _ = load(ns)
        abs_ = [g for _, g in meta.groupby(["game_pk", "at_bat_index"]) if (g[col] == tgt).any() and (g[col] == oth).any()]
        made = 0
        for i in rng.permutation(len(abs_)):
            g = abs_[i]
            sa, sb = still(g[g[col] == tgt].iloc[0], pid), still(g[g[col] == oth].iloc[0], pid)
            if not (sa and sb):
                continue
            tid = f"{keyname}_{made + 1}"
            left_is_target = bool(rng.integers(2))
            trials.append({"id": tid, "pitcher": name, "q": q, "cue": cue,
                           "left": sa if left_is_target else sb, "right": sb if left_is_target else sa})
            key[tid] = {"answer": "left" if left_is_target else "right", "pitcher": name, "game_pk": int(g.game_pk.iloc[0]),
                        "at_bat_index": int(g.at_bat_index.iloc[0])}
            made += 1
            if made == n:
                break
        print(f"{name}: {made} pairs")
    order = rng.permutation(len(trials))
    trials = [trials[i] for i in order]
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(key, open(os.path.join(OUT_DIR, "key.json"), "w"), indent=1)
    html = PAGE.replace("__TRIALS__", json.dumps(trials))
    out = os.path.join(OUT_DIR, "tip_blind_click_sheet.html")
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out, f"({len(trials)} pairs, {os.path.getsize(out) / 1e6:.1f} MB)")


def score(path):
    from scipy.stats import binomtest
    key = json.load(open(os.path.join(OUT_DIR, "key.json")))
    ans = json.load(open(path))
    by = {}
    for tid, pick in ans.items():
        if pick not in ("left", "right"):
            continue
        k = key[tid]; by.setdefault(k["pitcher"], []).append(pick == k["answer"])
    for p, v in by.items():
        c, n = sum(v), len(v)
        print(f"{p:32s} {c} of {n} right ({c / n:.0%}); a guesser gets this many or more {binomtest(c, n, 0.5, alternative='greater').pvalue:.0%} of the time")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tip Check</title><style>
:root{--bg:#f6f7f9;--ink:#1d2433;--sub:#5b6577;--line:#d9dde4;--navy:#002D62;--orange:#EB6E1F}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.4 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:760px;margin:0 auto;padding:18px 16px 40px}
h1{font-size:20px;margin:0 0 4px;color:var(--navy)} .sub{color:var(--sub);font-size:14px;margin-bottom:14px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px}
.q{font-weight:650;font-size:17px} .cue{color:var(--sub);font-size:14px;margin:2px 0 10px}
.pair{display:flex;gap:10px} .pair figure{flex:1;margin:0;text-align:center}
.pair img{width:100%;border-radius:6px;border:3px solid transparent;cursor:pointer}
.pair img.sel{border-color:var(--orange)} figcaption{font-size:13px;color:var(--sub)}
.row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer}
button.pri{background:var(--navy);color:#fff;border-color:var(--navy)} .prog{font-size:13px;color:var(--sub)}
</style></head><body><main>
<h1>Tip check</h1><div class="sub">Two stills from the same at-bat, both half a second before he moves. Click the one you think is the
pickoff / breaking ball / changeup. "Can't tell" is fine. Nothing here shows the answer.</div>
<div class="card" id="c"></div><div class="row"><span class="prog" id="p"></span></div>
<div class="row"><button class="pri" id="dl">Download answers</button></div>
<script>
const T = __TRIALS__; let i = 0; const A = {};
try { Object.assign(A, JSON.parse(localStorage.getItem("tipBlind") || "{}")); } catch (e) {}
function save(){ try { localStorage.setItem("tipBlind", JSON.stringify(A)); } catch (e) {} }
function draw(){
  const t = T[i], c = document.getElementById("c");
  c.innerHTML = `<div class="q">${t.pitcher}: ${t.q}</div><div class="cue">${t.cue}</div>
   <div class="pair"><figure><img id="L" src="data:image/jpeg;base64,${t.left}"><figcaption>Left</figcaption></figure>
   <figure><img id="R" src="data:image/jpeg;base64,${t.right}"><figcaption>Right</figcaption></figure></div>
   <div class="row"><button id="b">Back</button><button id="n">Can't tell</button><button id="f">Next</button></div>`;
  const mark = () => { document.getElementById("L").classList.toggle("sel", A[t.id] === "left");
                       document.getElementById("R").classList.toggle("sel", A[t.id] === "right"); };
  document.getElementById("L").onclick = () => { A[t.id] = "left"; save(); mark(); setTimeout(next, 250); };
  document.getElementById("R").onclick = () => { A[t.id] = "right"; save(); mark(); setTimeout(next, 250); };
  document.getElementById("n").onclick = () => { A[t.id] = "cant_tell"; save(); next(); };
  document.getElementById("b").onclick = () => { if (i > 0) { i--; draw(); } };
  document.getElementById("f").onclick = next; mark();
  document.getElementById("p").textContent = `${i + 1} of ${T.length} - answered ${Object.keys(A).length}`;
}
function next(){ if (i < T.length - 1) { i++; draw(); } else { document.getElementById("p").textContent = `Done - answered ${Object.keys(A).length} of ${T.length}. Click Download answers.`; } }
document.getElementById("dl").onclick = () => { const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(A, null, 1)], {type: "application/json"}));
  a.download = "tip_blind_answers.json"; a.click(); };
draw();
</script></main></body></html>"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--score")
    a = ap.parse_args()
    score(a.score) if a.score else build()
