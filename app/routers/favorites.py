from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db, Favorite, Movie, Rating, User
from app.models import FavoriteCreate, FavoriteResponse
from app.auth import require_auth
from sqlalchemy import func

router = APIRouter(prefix="/favorites", tags=["Favorites"])


@router.get("/{user_id}", response_model=list[FavoriteResponse])
def get_user_favorites(
    user_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Get user's favorite movies."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view other user's favorites")
    favorites = db.query(Favorite).filter(Favorite.user_id == user_id).all()
    results = []
    for fav in favorites:
        movie = db.query(Movie).filter(Movie.id == fav.movie_id).first()
        # Get rating stats
        stats = db.query(
            func.avg(Rating.rating),
            func.count(Rating.id),
        ).filter(Rating.movie_id == fav.movie_id).first()
        avg_rating = round(float(stats[0]), 1) if stats[0] else 0.0
        total_ratings = stats[1] or 0

        results.append(FavoriteResponse(
            id=fav.id,
            user_id=fav.user_id,
            movie_id=fav.movie_id,
            movie_title=movie.title if movie else "Unknown",
            genres=(movie.genres or "unknown").split("|") if movie else [],
            poster_url=movie.poster_url if movie else None,
            blur_hash=movie.blur_hash if movie else None,
            average_rating=avg_rating,
            total_ratings=total_ratings,
            created_at=fav.created_at,
        ))
    return results


@router.post("", response_model=FavoriteResponse, status_code=201)
def add_favorite(
    fav_data: FavoriteCreate,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Add movie to favorites."""
    if current_user.id != fav_data.user_id:
        raise HTTPException(status_code=403, detail="Cannot add favorites for other users")
    movie = db.query(Movie).filter(Movie.id == fav_data.movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    existing = (
        db.query(Favorite)
        .filter(Favorite.user_id == fav_data.user_id, Favorite.movie_id == fav_data.movie_id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Movie already in favorites")

    fav = Favorite(user_id=fav_data.user_id, movie_id=fav_data.movie_id)
    db.add(fav)
    db.commit()
    db.refresh(fav)

    stats = db.query(
        func.avg(Rating.rating),
        func.count(Rating.id),
    ).filter(Rating.movie_id == fav.movie_id).first()

    return FavoriteResponse(
        id=fav.id,
        user_id=fav.user_id,
        movie_id=fav.movie_id,
        movie_title=movie.title,
        genres=(movie.genres or "unknown").split("|"),
        poster_url=movie.poster_url,
        blur_hash=movie.blur_hash,
        average_rating=round(float(stats[0]), 1) if stats[0] else 0.0,
        total_ratings=stats[1] or 0,
        created_at=fav.created_at,
    )


@router.delete("/{user_id}/{movie_id}")
def remove_favorite(
    user_id: int,
    movie_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Remove movie from favorites."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot remove other user's favorites")
    fav = (
        db.query(Favorite)
        .filter(Favorite.user_id == user_id, Favorite.movie_id == movie_id)
        .first()
    )
    if not fav:
        raise HTTPException(status_code=404, detail="Favorite not found")

    db.delete(fav)
    db.commit()
    return {"message": "Removed from favorites"}
