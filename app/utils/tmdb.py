"""TMDB API integration for fetching movie poster URLs."""
import re
import time
import httpx
from app.config import TMDB_API_KEY

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _clean_title(title: str) -> tuple[str, int | None]:
    """Extract clean title and year from MovieLens format like 'Toy Story (1995)'."""
    match = re.match(r"^(.+?)\s*\((\d{4})\)\s*$", title)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title.strip(), None


def fetch_tmdb_match(title: str) -> tuple[str | None, int | None, str | None]:
    """Search TMDB by title (+ optional year) and return (poster_url, tmdb_id, overview).

    Used by the poster backfill so we capture the TMDB id + overview at the
    same time. The tmdb_id powers /movie/{id}/videos for trailers; the
    overview is shown on the movie details page.
    """
    if not TMDB_API_KEY:
        return None, None, None

    clean_title, year = _clean_title(title)

    params: dict[str, str] = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
    }
    if year:
        params["year"] = str(year)

    try:
        response = httpx.get(
            f"{TMDB_BASE_URL}/search/movie",
            params=params,
            timeout=10.0,
        )
        if response.status_code != 200:
            return None, None, None

        data = response.json()
        results = data.get("results", [])
        if not results:
            return None, None, None
        top = results[0]
        poster_path = top.get("poster_path")
        poster = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
        tmdb_id = top.get("id")
        overview_raw = top.get("overview")
        overview = overview_raw.strip() if isinstance(overview_raw, str) and overview_raw.strip() else None
        return poster, (int(tmdb_id) if tmdb_id is not None else None), overview
    except Exception:
        return None, None, None


def fetch_trailer_key(tmdb_id: int) -> str | None:
    """Return the YouTube video key for `tmdb_id`'s best trailer, or None.

    Preference order:
      1. Official YouTube Trailer
      2. Any YouTube Trailer
      3. Any YouTube Teaser
    """
    if not TMDB_API_KEY or not tmdb_id:
        return None

    try:
        response = httpx.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}/videos",
            params={"api_key": TMDB_API_KEY},
            timeout=10.0,
        )
        if response.status_code != 200:
            return None
        videos = response.json().get("results", []) or []
    except Exception:
        return None

    youtube = [v for v in videos if v.get("site") == "YouTube" and v.get("key")]
    if not youtube:
        return None

    for video in youtube:
        if video.get("type") == "Trailer" and video.get("official") is True:
            return video["key"]
    for video in youtube:
        if video.get("type") == "Trailer":
            return video["key"]
    for video in youtube:
        if video.get("type") == "Teaser":
            return video["key"]
    return youtube[0]["key"]


def populate_poster_urls(db_session, movie_model):
    """Populate poster_url + tmdb_id + overview for movies missing them.

    Returns count updated.
    """
    if not TMDB_API_KEY:
        print("TMDB_API_KEY not set — skipping poster population.")
        return 0

    movies = db_session.query(movie_model).filter(
        movie_model.poster_url.is_(None)
    ).all()
    total = len(movies)
    print(f"Starting poster + overview population for {total} movies...", flush=True)

    updated = 0
    processed = 0
    for movie in movies:
        processed += 1
        url, tmdb_id, overview = fetch_tmdb_match(movie.title)
        hits: list[str] = []
        if url:
            movie.poster_url = url
            updated += 1
            hits.append("poster")
        if tmdb_id and not movie.tmdb_id:
            movie.tmdb_id = tmdb_id
            hits.append(f"tmdb_id={tmdb_id}")
        if overview and not movie.overview:
            movie.overview = overview
            hits.append(f"overview({len(overview)})")
        print(
            f"  [{processed}/{total}] {movie.title!r}: " +
            (", ".join(hits) if hits else "no match"),
            flush=True,
        )

        # Rate limit: TMDB allows ~40 requests per 10 seconds
        time.sleep(0.05)

        if updated % 50 == 0 and updated > 0:
            db_session.commit()
            print(f"  Updated {updated} posters so far...", flush=True)

    db_session.commit()
    print(f"Populated {updated} poster URLs out of {total} movies.", flush=True)
    return updated


def backfill_overviews(db_session, movie_model) -> int:
    """One-shot pass to populate overview for movies that already have a
    poster_url/tmdb_id but no overview (legacy rows from before the field was
    added). Cheap — only hits TMDB for movies with overview IS NULL.
    Returns count updated.
    """
    if not TMDB_API_KEY:
        print("TMDB_API_KEY not set — skipping overview backfill.")
        return 0

    movies = db_session.query(movie_model).filter(
        movie_model.overview.is_(None)
    ).all()
    total = len(movies)
    print(f"Starting overview backfill for {total} movies...", flush=True)

    updated = 0
    processed = 0
    for movie in movies:
        processed += 1
        _url, _tmdb_id, overview = fetch_tmdb_match(movie.title)
        if overview:
            movie.overview = overview
            updated += 1
            print(
                f"  [{processed}/{total}] {movie.title!r}: overview ({len(overview)} chars)",
                flush=True,
            )
        else:
            print(
                f"  [{processed}/{total}] {movie.title!r}: no overview",
                flush=True,
            )
        time.sleep(0.05)
        if updated % 50 == 0 and updated > 0:
            db_session.commit()
            print(f"  Backfilled {updated} overviews so far...", flush=True)

    db_session.commit()
    print(f"Backfilled {updated} overviews out of {total} candidates.", flush=True)
    return updated
