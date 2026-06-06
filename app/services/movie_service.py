from datetime import timedelta
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import Movie, Rating
from app.utils.tmdb import fetch_tmdb_match, fetch_trailer_key

# Sentinel stored in `Movie.trailer_key` when a TMDB lookup completed but no
# trailer exists. Lets us skip re-querying TMDB on subsequent requests.
_TRAILER_NOT_FOUND = ""


# Window size for the "trending" velocity calculation, in days. Counts ratings
# whose timestamp falls inside this window before MAX(timestamp), so the metric
# stays meaningful for the MovieLens dataset (whose ratings are from the 1990s)
# while still picking up genuine post-deploy activity once real users join.
TRENDING_WINDOW_DAYS = 90
TRENDING_MIN_RATINGS_IN_WINDOW = 3


def get_movie_with_stats(movie: Movie, db: Session) -> dict:
    """Build movie response dict with rating stats."""
    stats = (
        db.query(
            func.avg(Rating.rating).label("avg"),
            func.count(Rating.id).label("count")
        )
        .filter(Rating.movie_id == movie.id)
        .first()
    )

    return {
        "id": movie.id,
        "title": movie.title,
        "genres": (movie.genres or "unknown").split("|"),
        "year": movie.release_year,
        "average_rating": round(float(stats.avg), 2) if stats.avg else None,
        "total_ratings": stats.count or 0,
        "poster_url": movie.poster_url,
        "blur_hash": movie.blur_hash,
        "overview": movie.overview,
    }


def search_movies(
    db: Session,
    query: Optional[str] = None,
    genre: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
) -> Tuple[List[dict], int]:
    """Search movies by title and/or genre.

    Query handling:
      - Splits the query into whitespace-separated tokens.
      - Returns any movie whose `title` or `genres` contains *at least one*
        token (OR semantics) so partial / typo'd multi-word queries still
        surface related results — e.g. "star wors" still finds Star Wars
        because the "star" token hits.
      - Result rows are ranked by (title-token hits + genre-token hits) DESC,
        so the best match floats to the top, with shorter titles winning ties
        between movies that hit the same number of tokens.
    """
    from sqlalchemy import case, or_

    q = db.query(Movie)

    tokens: List[str] = []
    if query:
        tokens = [t for t in query.strip().split() if t]

    if tokens:
        like_conditions = []
        for token in tokens:
            like = f"%{token}%"
            like_conditions.append(Movie.title.ilike(like))
            like_conditions.append(Movie.genres.ilike(like))
        q = q.filter(or_(*like_conditions))

        title_score = sum(
            (case((Movie.title.ilike(f"%{t}%"), 2), else_=0) for t in tokens),
            start=0,
        )
        genre_score = sum(
            (case((Movie.genres.ilike(f"%{t}%"), 1), else_=0) for t in tokens),
            start=0,
        )
        q = q.order_by(
            (title_score + genre_score).desc(),
            func.length(Movie.title).asc(),
        )

    if genre:
        q = q.filter(Movie.genres.ilike(f"%{genre}%"))

    total = q.count()
    movies = q.offset((page - 1) * per_page).limit(per_page).all()

    results = [get_movie_with_stats(m, db) for m in movies]
    return results, total


def get_popular_movies(db: Session, limit: int = 20) -> List[dict]:
    """Get most popular movies by average rating (min 20 ratings)."""
    popular = (
        db.query(
            Rating.movie_id,
            func.avg(Rating.rating).label("avg_rating"),
            func.count(Rating.id).label("count")
        )
        .group_by(Rating.movie_id)
        .having(func.count(Rating.id) >= 20)
        .order_by(func.avg(Rating.rating).desc())
        .limit(limit)
        .all()
    )

    results = []
    for movie_id, avg_rating, count in popular:
        movie = db.query(Movie).filter(Movie.id == movie_id).first()
        if movie:
            results.append({
                "id": movie.id,
                "title": movie.title,
                "genres": (movie.genres or "unknown").split("|"),
                "year": movie.release_year,
                "average_rating": round(float(avg_rating), 2),
                "total_ratings": count,
                "poster_url": movie.poster_url,
                "blur_hash": movie.blur_hash,
                "overview": movie.overview,
            })

    return results


def get_trending_movies(db: Session, limit: int = 20) -> List[dict]:
    """Get trending movies by rating velocity.

    "Trending" = the most ratings inside a recent time window, where the window
    is `TRENDING_WINDOW_DAYS` long and anchored to the latest rating timestamp
    in the database (not wall-clock `now`). Anchoring to MAX(timestamp) keeps
    the MovieLens 90s data usable as trending content while still surfacing
    new activity once real users join.

    Sort: ratings-in-window DESC, then average rating DESC as a tie-breaker
    so two equally-active movies are split by quality.
    """
    latest_ts = db.query(func.max(Rating.timestamp)).scalar()
    if latest_ts is None:
        return []

    window_start = latest_ts - timedelta(days=TRENDING_WINDOW_DAYS)

    in_window_count = func.count(Rating.id).filter(
        Rating.timestamp >= window_start
    ).label("recent_count")

    trending = (
        db.query(
            Rating.movie_id,
            func.avg(Rating.rating).label("avg_rating"),
            func.count(Rating.id).label("total_count"),
            in_window_count,
        )
        .group_by(Rating.movie_id)
        .having(in_window_count >= TRENDING_MIN_RATINGS_IN_WINDOW)
        .order_by(in_window_count.desc(), func.avg(Rating.rating).desc())
        .limit(limit)
        .all()
    )

    results = []
    for movie_id, avg_rating, total_count, _recent_count in trending:
        movie = db.query(Movie).filter(Movie.id == movie_id).first()
        if movie:
            results.append({
                "id": movie.id,
                "title": movie.title,
                "genres": (movie.genres or "unknown").split("|"),
                "year": movie.release_year,
                "average_rating": round(float(avg_rating), 2),
                "total_ratings": total_count,
                "poster_url": movie.poster_url,
                "blur_hash": movie.blur_hash,
            })

    return results


def get_trailer_key(db: Session, movie_id: int) -> str | None:
    """Return the cached YouTube trailer key for `movie_id`, fetching from
    TMDB on first call.

    Cache states stored in `Movie.trailer_key`:
      - `None`        → not yet looked up.
      - `""`          → looked up, TMDB had nothing → don't re-query.
      - `"<key>"`     → use it.
    """
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if movie is None:
        return None

    # Cached hit (positive or negative).
    if movie.trailer_key is not None:
        return movie.trailer_key or None

    # Need a TMDB id. Populate lazily if missing.
    if movie.tmdb_id is None:
        _poster, tmdb_id, overview = fetch_tmdb_match(movie.title)
        changed = False
        if tmdb_id is not None:
            movie.tmdb_id = tmdb_id
            changed = True
        if overview and not movie.overview:
            movie.overview = overview
            changed = True
        if changed:
            db.commit()

    if movie.tmdb_id is None:
        # Couldn't map to TMDB at all — mark negative so we don't keep trying.
        movie.trailer_key = _TRAILER_NOT_FOUND
        db.commit()
        return None

    key = fetch_trailer_key(movie.tmdb_id)
    movie.trailer_key = key or _TRAILER_NOT_FOUND
    db.commit()
    return key
