from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from app.database import get_db, Feedback, Movie, Rating, User
from app.models import FeedbackCreate, FeedbackResponse
from app.services.recommender import recommender
from app.auth import require_auth

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("", response_model=FeedbackResponse, status_code=201)
def submit_feedback(
    data: FeedbackCreate,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Submit thumbs up/down feedback on a movie.

    feedback_type must be 'thumbs_up' or 'thumbs_down'.
    This also creates/updates a rating (5.0 for thumbs_up, 1.0 for thumbs_down)
    to feed back into the collaborative filtering model.
    """
    if current_user.id != data.user_id:
        raise HTTPException(status_code=403, detail="Cannot submit feedback for other users")
    if data.feedback_type not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(status_code=400, detail="feedback_type must be 'thumbs_up' or 'thumbs_down'")

    movie = db.query(Movie).filter(Movie.id == data.movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Upsert feedback
    existing = (
        db.query(Feedback)
        .filter(Feedback.user_id == data.user_id, Feedback.movie_id == data.movie_id)
        .first()
    )

    if existing:
        existing.feedback_type = data.feedback_type
        existing.created_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        feedback = existing
    else:
        feedback = Feedback(
            user_id=data.user_id,
            movie_id=data.movie_id,
            feedback_type=data.feedback_type,
            created_at=datetime.utcnow(),
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)

    # Also feed an implicit rating into the CF model so thumbs gestures still
    # influence recommendations. Crucially, NEVER overwrite a row the user
    # already rated explicitly — their 3.5★ is more accurate than a binary
    # thumbs-up snap to 5.0, and silently clobbering it ruins their history.
    existing_rating = (
        db.query(Rating)
        .filter(Rating.user_id == data.user_id, Rating.movie_id == data.movie_id)
        .first()
    )
    if existing_rating is None:
        implicit_rating = 5.0 if data.feedback_type == "thumbs_up" else 1.0
        db.add(Rating(
            user_id=data.user_id,
            movie_id=data.movie_id,
            rating=implicit_rating,
            timestamp=datetime.utcnow(),
        ))
        db.commit()
        recommender.record_new_rating()

    return FeedbackResponse(
        id=feedback.id,
        user_id=feedback.user_id,
        movie_id=feedback.movie_id,
        feedback_type=feedback.feedback_type,
        created_at=feedback.created_at,
    )


@router.get("/user/{user_id}", response_model=list[FeedbackResponse])
def get_user_feedback(
    user_id: int,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Get all feedback submitted by a user."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view other user's feedback")
    feedbacks = db.query(Feedback).filter(Feedback.user_id == user_id).all()
    return [
        FeedbackResponse(
            id=f.id,
            user_id=f.user_id,
            movie_id=f.movie_id,
            feedback_type=f.feedback_type,
            created_at=f.created_at,
        )
        for f in feedbacks
    ]


@router.get("/movie/{movie_id}")
def get_movie_feedback_stats(movie_id: int, db: Session = Depends(get_db)):
    """Get feedback statistics for a movie."""
    thumbs_up = db.query(Feedback).filter(
        Feedback.movie_id == movie_id, Feedback.feedback_type == "thumbs_up"
    ).count()
    thumbs_down = db.query(Feedback).filter(
        Feedback.movie_id == movie_id, Feedback.feedback_type == "thumbs_down"
    ).count()

    return {
        "movie_id": movie_id,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "total": thumbs_up + thumbs_down,
    }
