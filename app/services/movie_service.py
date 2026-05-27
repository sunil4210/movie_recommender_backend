from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import Movie, Rating


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
    }


def search_movies(
    db: Session,
    query: Optional[str] = None,
    genre: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
) -> Tuple[List[dict], int]:
    """Search movies by title and/or genre."""
    q = db.query(Movie)

    if query:
        q = q.filter(Movie.title.ilike(f"%{query}%"))

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
            })

    return results


def get_trending_movies(db: Session, limit: int = 20) -> List[dict]:
    """Get trending movies by recent high ratings."""
    trending = (
        db.query(
            Rating.movie_id,
            func.avg(Rating.rating).label("avg_rating"),
            func.count(Rating.id).label("count")
        )
        .group_by(Rating.movie_id)
        .having(func.count(Rating.id) >= 5)
        .order_by(func.max(Rating.timestamp).desc())
        .limit(limit)
        .all()
    )

    results = []
    for movie_id, avg_rating, count in trending:
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
            })

    return results
