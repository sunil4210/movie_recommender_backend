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


def fetch_poster_url(title: str) -> str | None:
    """Search TMDB for a movie and return its poster URL."""
    if not TMDB_API_KEY:
        return None

    clean_title, year = _clean_title(title)

    params = {
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
            return None

        data = response.json()
        results = data.get("results", [])
        if results and results[0].get("poster_path"):
            return f"{TMDB_IMAGE_BASE}{results[0]['poster_path']}"
    except Exception:
        pass

    return None


def populate_poster_urls(db_session, movie_model):
    """Populate poster_url for all movies missing one. Returns count updated."""
    if not TMDB_API_KEY:
        print("TMDB_API_KEY not set — skipping poster population.")
        return 0

    movies = db_session.query(movie_model).filter(
        movie_model.poster_url.is_(None)
    ).all()

    updated = 0
    for movie in movies:
        url = fetch_poster_url(movie.title)
        if url:
            movie.poster_url = url
            updated += 1

        # Rate limit: TMDB allows ~40 requests per 10 seconds
        time.sleep(0.05)

        if updated % 50 == 0 and updated > 0:
            db_session.commit()
            print(f"  Updated {updated} posters so far...")

    db_session.commit()
    print(f"Populated {updated} poster URLs out of {len(movies)} movies.")
    return updated
