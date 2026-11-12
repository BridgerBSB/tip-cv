"""Download the clips for chosen events from tip_pull CSVs.

    python scripts/fetch_clips.py --pull data/pulls/tip2_miller_2026_reg.csv data/pulls/tip2_miller_2026_post.csv \
        --angles PITCHCAST 3B_PITCHER_SIDEVIEW --runner on2b --event pitch

Writes data/clips/<pitcher_id>/<ANGLE>/<game_pk>_<play_id>.mp4. Skips files already on disk, so a killed run
resumes where it stopped. Stops (does not start another download) when free disk drops under --min-free-gb.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.clips import download  # noqa: E402


def load_pulls(paths: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert len(df), "empty pull"
    return df


def clip_path(row, angle: str) -> str:
    return os.path.join(ROOT, "data", "clips", str(int(row.pitcher_id)), angle,
                        f"{int(row.game_pk)}_{row.play_id}.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", nargs="+", required=True)
    ap.add_argument("--angles", nargs="+", default=["PITCHCAST"])
    ap.add_argument("--runner", choices=["any", "on1b", "on2b", "empty"], default="any",
                    help="on1b/on2b = that base occupied (others may be too); empty = bases empty")
    ap.add_argument("--event", choices=["any", "pitch", "pickoff", "stepoff"], default="any")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    a = ap.parse_args()

    df = load_pulls(a.pull)
    if a.event != "any":
        df = df[df.event_type == a.event]
    if a.runner == "empty":
        df = df[~df.on1b.fillna(False) & ~df.on2b.fillna(False) & ~df.on3b.fillna(False)]
    elif a.runner != "any":
        df = df[df[a.runner].fillna(False).astype(bool)]
    jobs = [(r, ang) for r in df.itertuples() for ang in a.angles
            if isinstance(getattr(r, "url_" + ang, None), str) and getattr(r, "url_" + ang)]
    missing = len(df) * len(a.angles) - len(jobs)
    print(f"events {len(df)} x angles {len(a.angles)} -> {len(jobs)} clips with a URL ({missing} without)")

    stop = threading.Event()
    lock = threading.Lock()
    done = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0}

    def one(job):
        r, ang = job
        if stop.is_set():
            return
        dest = clip_path(r, ang)
        if os.path.exists(dest):
            with lock:
                done["skip"] += 1
            return
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        if free_gb < a.min_free_gb:
            stop.set()
            print(f"STOP: {free_gb:.1f} GB free < {a.min_free_gb}")
            return
        try:
            n = download(getattr(r, "url_" + ang), dest)
            with lock:
                done["ok"] += 1
                done["bytes"] += n
        except Exception as e:  # noqa: BLE001
            with lock:
                done["fail"] += 1
            print(f"FAIL {r.game_pk} {r.play_id} {ang}: {e}")
        k = done["ok"] + done["skip"] + done["fail"]
        if k % 50 == 0:
            print(f"  {k}/{len(jobs)}  ok {done['ok']} skip {done['skip']} fail {done['fail']}  "
                  f"{done['bytes'] / 1e9:.2f} GB", flush=True)

    # executor.submit per job (not ex.map over a generator that eagerly queues everything with no floor check)
    with ThreadPoolExecutor(a.workers) as ex:
        for job in jobs:
            if stop.is_set():
                break
            ex.submit(one, job)
    print(f"DONE ok {done['ok']} skip {done['skip']} fail {done['fail']}  {done['bytes'] / 1e9:.2f} GB new; "
          f"free {shutil.disk_usage(ROOT).free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
