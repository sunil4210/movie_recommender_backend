from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from app.auth import require_auth
from app.database import get_db, Movie, User
from app.models import MovieResponse, MovieListResponse, TrailerResponse
from app.services.movie_service import (
    get_movie_with_stats, search_movies, get_popular_movies, get_trending_movies,
    get_trailer_key,
)
from app.utils.tmdb import populate_poster_urls, backfill_overviews

router = APIRouter(prefix="/movies", tags=["Movies"])


@router.get("", response_model=MovieListResponse)
def list_movies(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get all movies (paginated)."""
    results, total = search_movies(db, page=page, per_page=per_page)
    return MovieListResponse(
        movies=[MovieResponse(**m) for m in results],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/search", response_model=MovieListResponse)
def search(
    q: Optional[str] = Query(None, description="Search by title"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Search movies by title and/or genre."""
    results, total = search_movies(db, query=q, genre=genre, page=page, per_page=per_page)
    return MovieListResponse(
        movies=[MovieResponse(**m) for m in results],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/popular", response_model=list[MovieResponse])
def popular(
    limit: int = Query(20, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get most popular movies."""
    results = get_popular_movies(db, limit=limit)
    return [MovieResponse(**m) for m in results]


@router.get("/trending", response_model=list[MovieResponse])
def trending(
    limit: int = Query(20, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get trending movies (by recent ratings)."""
    results = get_trending_movies(db, limit=limit)
    return [MovieResponse(**m) for m in results]


@router.post("/populate-posters")
def populate_posters(
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Fetch and store poster URLs from TMDB for all movies missing one.

    Auth-gated: each call iterates the whole catalog with TMDB lookups and
    would burn our API quota if exposed anonymously.
    """
    count = populate_poster_urls(db, Movie)
    return {"updated": count}


@router.post("/backfill-overviews")
def backfill_overviews_route(
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Fill `overview` for movies that already have a poster but no synopsis.

    Cheap, idempotent — only hits TMDB for rows where overview IS NULL.
    Same auth rationale as `populate-posters`.
    """
    count = backfill_overviews(db, Movie)
    return {"updated": count}


@router.get("/{movie_id}", response_model=MovieResponse)
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    """Get specific movie details."""
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return MovieResponse(**get_movie_with_stats(movie, db))


@router.get("/{movie_id}/trailer", response_model=TrailerResponse)
def get_trailer(movie_id: int, db: Session = Depends(get_db)):
    """Return the official YouTube trailer for a movie.

    Resolves the TMDB id lazily on first call, then caches the result so
    repeated requests skip TMDB entirely. Returns 404 when no trailer is
    available."""
    if not db.query(Movie).filter(Movie.id == movie_id).first():
        raise HTTPException(status_code=404, detail="Movie not found")

    key = get_trailer_key(db, movie_id)
    if not key:
        raise HTTPException(status_code=404, detail="No trailer available for this movie")

    return TrailerResponse(
        youtube_key=key,
        embed_url=f"https://www.youtube.com/embed/{key}",
    )
