"""Download the reference YOLO11 weights listed in models/models.json.

The weights are published under AGPL-3.0 by their author (Hugging Face model card
surawut/chess-move-tracking-yolo11). Running this script means you accept those terms.
It prints the SHA-256 of every downloaded file; nothing is uploaded anywhere.

Usage:  python scripts/fetch_reference_weights.py --accept-agpl [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "chess-vision-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-agpl", action="store_true", help="confirm you accept the AGPL-3.0 weight license")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()
    if not args.accept_agpl:
        print("The reference weights are licensed AGPL-3.0. Re-run with --accept-agpl to download them.")
        return 2
    manifest = json.loads((MODELS_DIR / "models.json").read_text(encoding="utf-8"))
    for m in manifest["models"]:
        dest = MODELS_DIR / m["file"]
        if dest.exists() and not args.force:
            print(f"exists: {dest.name}")
        else:
            print(f"downloading {m['source_url']}")
            download(m["source_url"], dest)
        digest = sha256(dest)
        expected = m.get("sha256")
        status = "matches manifest" if expected and expected == digest else ("NO MANIFEST HASH" if not expected else "MISMATCH")
        print(f"  sha256 {digest}  size {dest.stat().st_size} bytes  ({status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
