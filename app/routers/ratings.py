from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime

from app.database import get_db, Rating, Movie, User
from app.models import RatingCreate, RatingUpdate, RatingResponse
from app.services.recommender import recommender
from app.auth import require_auth

router = APIRouter(prefix="/ratings", tags=["Ratings"])


@router.get("/user/{user_id}", response_model=list[RatingResponse])
def get_user_ratings(
    user_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Get all ratings by a user."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view other user's ratings")
    ratings = db.query(Rating).filter(Rating.user_id == user_id).all()
    results = []
    for r in ratings:
        movie = db.query(Movie).filter(Movie.id == r.movie_id).first()
        results.append(RatingResponse(
            id=r.id,
            user_id=r.user_id,
            movie_id=r.movie_id,
            movie_title=movie.title if movie else "Unknown",
            rating=r.rating,
            timestamp=r.timestamp,
        ))
    return results


@router.post("", response_model=RatingResponse, status_code=201)
def submit_rating(
    rating_data: RatingCreate,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Submit a new rating (or update existing one)."""
    if current_user.id != rating_data.user_id:
        raise HTTPException(status_code=403, detail="Cannot submit ratings for other users")
    # Validate rating range
    if not 1.0 <= rating_data.rating <= 5.0:
        raise HTTPException(status_code=400, detail="Rating must be between 1.0 and 5.0")

    # Check if movie exists
    movie = db.query(Movie).filter(Movie.id == rating_data.movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Check for existing rating
    existing = (
        db.query(Rating)
        .filter(Rating.user_id == rating_data.user_id, Rating.movie_id == rating_data.movie_id)
        .first()
    )

    if existing:
        existing.rating = rating_data.rating
        existing.timestamp = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        recommender.record_new_rating()
        return RatingResponse(
            id=existing.id,
            user_id=existing.user_id,
            movie_id=existing.movie_id,
            movie_title=movie.title,
            rating=existing.rating,
            timestamp=existing.timestamp,
        )

    rating = Rating(
        user_id=rating_data.user_id,
        movie_id=rating_data.movie_id,
        rating=rating_data.rating,
        timestamp=datetime.utcnow(),
    )
    db.add(rating)
    db.commit()
    db.refresh(rating)

    recommender.record_new_rating()

    return RatingResponse(
        id=rating.id,
        user_id=rating.user_id,
        movie_id=rating.movie_id,
        movie_title=movie.title,
        rating=rating.rating,
        timestamp=rating.timestamp,
    )


@router.put("/{rating_id}", response_model=RatingResponse)
def update_rating(
    rating_id: int,
    rating_data: RatingUpdate,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Update an existing rating."""
    if not 1.0 <= rating_data.rating <= 5.0:
        raise HTTPException(status_code=400, detail="Rating must be between 1.0 and 5.0")

    rating = db.query(Rating).filter(Rating.id == rating_id).first()
    if not rating:
        raise HTTPException(status_code=404, detail="Rating not found")
    if rating.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot update other user's rating")

    rating.rating = rating_data.rating
    rating.timestamp = datetime.utcnow()
    db.commit()
    db.refresh(rating)

    movie = db.query(Movie).filter(Movie.id == rating.movie_id).first()
    recommender.record_new_rating()

    return RatingResponse(
        id=rating.id,
        user_id=rating.user_id,
        movie_id=rating.movie_id,
        movie_title=movie.title if movie else "Unknown",
        rating=rating.rating,
        timestamp=rating.timestamp,
    )


@router.delete("/{rating_id}")
def delete_rating(
    rating_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Delete a rating."""
    rating = db.query(Rating).filter(Rating.id == rating_id).first()
    if not rating:
        raise HTTPException(status_code=404, detail="Rating not found")
    if rating.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot delete other user's rating")

    db.delete(rating)
    db.commit()
    return {"message": "Rating deleted"}


@router.get("/movie/{movie_id}", response_model=list[RatingResponse])
def get_movie_ratings(movie_id: int, db: Session = Depends(get_db)):
    """Get all ratings for a movie."""
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    ratings = db.query(Rating).filter(Rating.movie_id == movie_id).limit(50).all()
    return [
        RatingResponse(
            id=r.id,
            user_id=r.user_id,
            movie_id=r.movie_id,
            movie_title=movie.title,
            rating=r.rating,
            timestamp=r.timestamp,
        )
        for r in ratings
    ]
