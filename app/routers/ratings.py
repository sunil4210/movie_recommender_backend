from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy import case, desc
from sqlalchemy.orm import Session
from datetime import datetime

from app.database import get_db, Rating, Movie, User
from app.models import RatingCreate, RatingUpdate, RatingResponse, MovieReviewResponse
from app.services.recommender import recommender
from app.auth import require_auth


def _normalize_comment(comment: str | None) -> str | None:
    if comment is None:
        return None
    text = comment.strip()
    if not text:
        return None
    return text[:1000]


def _display_name(user: User | None) -> str:
    if user is None:
        return "Unknown"
    first = (user.first_name or "").strip()
    last = (user.last_name or "").strip()
    name = (first + " " + last).strip()
    return name or user.username or f"User {user.id}"

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
            comment=r.comment,
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

    comment = _normalize_comment(rating_data.comment)

    if existing:
        existing.rating = rating_data.rating
        # Overwrite when the client sent a non-empty comment, clear when the
        # client explicitly sent an empty string, keep prior text when omitted.
        if rating_data.comment is not None:
            existing.comment = comment
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
            comment=existing.comment,
            timestamp=existing.timestamp,
        )

    rating = Rating(
        user_id=rating_data.user_id,
        movie_id=rating_data.movie_id,
        rating=rating_data.rating,
        comment=comment,
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
        comment=rating.comment,
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
    if rating_data.comment is not None:
        rating.comment = _normalize_comment(rating_data.comment)
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
        comment=rating.comment,
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
            comment=r.comment,
            timestamp=r.timestamp,
        )
        for r in ratings
    ]


SORT_NEWEST = "newest"
SORT_OLDEST = "oldest"
SORT_HIGHEST = "highest"
SORT_LOWEST = "lowest"
_ALLOWED_SORTS = {SORT_NEWEST, SORT_OLDEST, SORT_HIGHEST, SORT_LOWEST}


@router.get("/movie/{movie_id}/reviews", response_model=list[MovieReviewResponse])
def get_movie_reviews(
    movie_id: int,
    limit: int = Query(1000, ge=1, le=5000),
    sort: str = Query(SORT_NEWEST),
    pin_user_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    """Public list of reviews left on a movie.

    Sort options: newest (default), oldest, highest, lowest. Rows that have a
    written comment are surfaced before bare star-only ratings within the
    same sort. When `pin_user_id` is provided that user's review (if any) is
    moved to position 0 of the result regardless of sort."""
    if sort not in _ALLOWED_SORTS:
        raise HTTPException(status_code=400, detail=f"sort must be one of {sorted(_ALLOWED_SORTS)}")

    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    has_comment = case(
        (Rating.comment.isnot(None), case((Rating.comment != "", 1), else_=0)),
        else_=0,
    )

    if sort == SORT_OLDEST:
        secondary = Rating.timestamp.asc()
    elif sort == SORT_HIGHEST:
        secondary = Rating.rating.desc()
    elif sort == SORT_LOWEST:
        secondary = Rating.rating.asc()
    else:
        secondary = Rating.timestamp.desc()

    rows = (
        db.query(Rating, User)
        .join(User, User.id == Rating.user_id)
        .filter(Rating.movie_id == movie_id)
        .order_by(desc(has_comment), secondary)
        .limit(limit)
        .all()
    )

    items = [
        MovieReviewResponse(
            id=r.id,
            user_id=r.user_id,
            user_name=_display_name(u),
            movie_id=r.movie_id,
            rating=r.rating,
            comment=r.comment or "",
            timestamp=r.timestamp,
        )
        for r, u in rows
    ]

    if pin_user_id is not None:
        for i, it in enumerate(items):
            if it.user_id == pin_user_id and i != 0:
                items.insert(0, items.pop(i))
                break

    return items


@router.get("/user/{user_id}/movie/{movie_id}", response_model=RatingResponse | None)
def get_user_movie_rating(
    user_id: int,
    movie_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Return the auth'd user's rating for a single movie (or null if none)."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view other user's rating")

    r = (
        db.query(Rating)
        .filter(Rating.user_id == user_id, Rating.movie_id == movie_id)
        .first()
    )
    if r is None:
        return None

    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    return RatingResponse(
        id=r.id,
        user_id=r.user_id,
        movie_id=r.movie_id,
        movie_title=movie.title if movie else "",
        rating=r.rating,
        comment=r.comment,
        timestamp=r.timestamp,
    )
