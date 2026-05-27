"""One-off batch job: compute BlurHash placeholders for every movie poster.

Run from the backend/ directory:
    python -m scripts.compute_blurhashes

Strategy:
  1. Query every Movie row with a non-null poster_url and no blur_hash.
  2. Download the poster (small thumbnail variant if URL pattern allows).
  3. Resize to 32px on the long edge — BlurHash quality is independent of source res.
  4. Encode with (x_components=4, y_components=3) — good detail for 2:3 posters.
  5. Persist back to DB.

Idempotent: skips movies that already have a hash. Re-runnable.
"""
from __future__ import annotations

import sys
import time
from io import BytesIO
from typing import Optional

import blurhash
import numpy as np
import requests
from PIL import Image

from app.database import SessionLocal, Movie, init_db

# Network politeness — TMDB allows generous rates but no need to hammer.
REQUEST_TIMEOUT = 10
SLEEP_BETWEEN = 0.05
# BlurHash quality knobs. 4x3 is a good fit for 2:3 movie posters.
X_COMPONENTS = 4
Y_COMPONENTS = 3
# Source image is downscaled before encoding; encoding cost scales with pixels.
MAX_EDGE = 64


def compute_one(url: str) -> Optional[str]:
    """Download → resize → BlurHash encode. Returns hash string or None on failure."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)
        return blurhash.encode(arr, components_x=X_COMPONENTS, components_y=Y_COMPONENTS)
    except Exception as exc:
        print(f"  ✗ {url}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        pending = (
            db.query(Movie)
            .filter(Movie.poster_url.isnot(None))
            .filter((Movie.blur_hash.is_(None)) | (Movie.blur_hash == ""))
            .all()
        )
        total = len(pending)
        print(f"Computing BlurHash for {total} movies...")

        ok = 0
        for i, movie in enumerate(pending, start=1):
            hash_str = compute_one(movie.poster_url)
            if hash_str:
                movie.blur_hash = hash_str
                ok += 1
                if i % 25 == 0:
                    db.commit()
                    print(f"  [{i}/{total}] committed, {ok} successful so far")
            time.sleep(SLEEP_BETWEEN)

        db.commit()
        print(f"Done. {ok}/{total} movies hashed.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
