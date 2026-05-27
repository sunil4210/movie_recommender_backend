from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db, User, Rating
from app.models import UserSignup, UserLogin, Token, UserResponse, UserUpdate, ChangePassword
from app.auth import hash_password, verify_password, create_access_token, require_auth

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _generate_unique_username(db: Session, first_name: str, last_name: str, email: str) -> str:
    """Generate a unique internal `username` value for the users table.

    The frontend does not collect a username — user identity is first/last/email only.
    The DB column is still NOT NULL UNIQUE (it's used by the MovieLens placeholder
    users like `movielens_user_42`), so we synthesize one here:

      1. Start from `<first>_<last>` lowercased, spaces stripped.
      2. If that's empty (shouldn't happen — pydantic enforces both fields), fall back
         to the email's local part.
      3. Append `_2`, `_3`, … until the candidate is unique.

    Returned value is never shown to the user — it's purely an internal handle.
    """
    base = f"{first_name}_{last_name}".lower().replace(" ", "")
    if not base.strip("_"):
        base = email.split("@")[0].lower()
    candidate = base
    suffix = 1
    while db.query(User).filter(User.username == candidate).first() is not None:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


@router.post("/signup", response_model=UserResponse)
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """Register a new user (first/last name + email + password).

    The IDs space includes MovieLens 100K users (943 baked-in IDs), so new signups
    take the next ID above the current max — never collides with the dataset.
    """
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # New ID must sit above MovieLens placeholder users (so we don't overwrite their ratings).
    max_id = db.query(User).order_by(User.id.desc()).first()
    new_id = (max_id.id + 1) if max_id else 1

    username = _generate_unique_username(
        db, user_data.first_name, user_data.last_name, user_data.email
    )

    user = User(
        id=new_id,
        username=username,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        age=user_data.age,
        gender=user_data.gender,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        age=user.age,
        gender=user.gender,
        total_ratings=0,
        favorite_genres=[],
        created_at=user.created_at,
    )


@router.post("/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login and get access token."""
    user = db.query(User).filter(User.email == user_data.email).first()

    if not user or user.password_hash == "$movielens_placeholder$":
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=token)


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Get current authenticated user info."""
    total_ratings = db.query(Rating).filter(Rating.user_id == current_user.id).count()

    # Get favorite genres from most-rated genres
    from sqlalchemy import func
    from app.database import Movie
    genre_stats = (
        db.query(Movie.genres)
        .join(Rating, Rating.movie_id == Movie.id)
        .filter(Rating.user_id == current_user.id)
        .filter(Rating.rating >= 4.0)
        .all()
    )

    genre_counts = {}
    for (genres_str,) in genre_stats:
        if genres_str:
            for g in genres_str.split("|"):
                genre_counts[g] = genre_counts.get(g, 0) + 1

    sorted_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)
    favorite_genres = sorted_genres[:5]

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        age=current_user.age,
        gender=current_user.gender,
        total_ratings=total_ratings,
        favorite_genres=favorite_genres,
        created_at=current_user.created_at,
    )


@router.put("/profile", response_model=UserResponse)
def update_profile(
    update_data: UserUpdate,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Update user profile."""
    if update_data.first_name is not None:
        current_user.first_name = update_data.first_name
    if update_data.last_name is not None:
        current_user.last_name = update_data.last_name
    if update_data.age is not None:
        current_user.age = update_data.age
    if update_data.gender is not None:
        current_user.gender = update_data.gender

    db.commit()
    db.refresh(current_user)

    total_ratings = db.query(Rating).filter(Rating.user_id == current_user.id).count()

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        age=current_user.age,
        gender=current_user.gender,
        total_ratings=total_ratings,
        favorite_genres=[],
        created_at=current_user.created_at,
    )


@router.put("/change-password")
def change_password(
    data: ChangePassword,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Change user password."""
    if not verify_password(data.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    current_user.password_hash = hash_password(data.new_password)
    db.commit()
    return {"message": "Password changed successfully"}
