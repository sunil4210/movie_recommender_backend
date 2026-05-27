from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db, Movie
from app.models import MovieResponse, MovieListResponse
from app.services.movie_service import (
    get_movie_with_stats, search_movies, get_popular_movies, get_trending_movies
)
from app.utils.tmdb import populate_poster_urls

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
def populate_posters(db: Session = Depends(get_db)):
    """Fetch and store poster URLs from TMDB for all movies missing one."""
    count = populate_poster_urls(db, Movie)
    return {"updated": count}


@router.get("/{movie_id}", response_model=MovieResponse)
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    """Get specific movie details."""
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return MovieResponse(**get_movie_with_stats(movie, db))
