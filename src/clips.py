"""Download sporty-clips mp4s named in a tip_pull CSV.

The CDN refuses a default user agent (Cloudflare "Attention Required" page, ~4.5 KB of HTML saved as
.mp4). A browser UA + statsapi Referer returns the video. Same headers as command-cv
src/ingest/video.py::_cdn_headers (verified there 2026-07; re-verified here 2026-11-12).
"""
from __future__ import annotations

import os
import time
import urllib.request

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
_HEADERS = {"User-Agent": _BROWSER_UA, "Accept": "*/*", "Referer": "https://statsapi.mlb.com/"}
MIN_BYTES = 200_000   # a real 6 s clip is ~4 MB; the Cloudflare page is ~4.5 KB


def download(url: str, dest: str, retries: int = 3) -> int:
    """Fetch url -> dest atomically (writes .part, renames). Returns bytes. Raises on a non-video body."""
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_BYTES:
        return os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            if len(body) < MIN_BYTES or body[:15].lstrip().lower().startswith(b"<!doctype"):
                raise RuntimeError(f"not a video ({len(body)} bytes) from {url[:60]}")
            part = dest + ".part"
            with open(part, "wb") as f:
                f.write(body)
            os.replace(part, dest)
            return len(body)
        except Exception as e:  # noqa: BLE001 - retry network blips, re-raise the last one
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last
