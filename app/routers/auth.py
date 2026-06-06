import logging

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db, User, Rating
from app.models import (
    UserSignup, UserLogin, Token, UserResponse, UserUpdate, ChangePassword,
    SignupResponse, VerifyEmailRequest, ResendOtpRequest,
    ForgotPasswordRequest, ResetPasswordRequest, GenericMessage,
)
from app.auth import hash_password, verify_password, create_access_token, require_auth
from app.services import otp_service, email_service

logger = logging.getLogger(__name__)

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


async def _issue_and_send_otp(db: Session, email: str, purpose: str) -> None:
    """Generate an OTP for `email` and email it. Cooldown errors map to 429."""
    try:
        code = otp_service.issue_otp(db, email, purpose)  # type: ignore[arg-type]
    except otp_service.OtpCooldownError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )
    except otp_service.OtpError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        await email_service.send_otp_email(email, code, purpose)
    except Exception as exc:
        logger.error("OTP email delivery failed for %s: %s", email, exc)
        raise HTTPException(
            status_code=502,
            detail="We couldn't send the email. Please try again shortly.",
        )


@router.post("/signup", response_model=SignupResponse)
async def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """Register a new user. The account is created unverified — the client
    must complete OTP verification before it can log in.

    The IDs space includes MovieLens 100K users (943 baked-in IDs), so new signups
    take the next ID above the current max — never collides with the dataset.
    """
    # Normalize email so case differences ("Foo@Bar.com" vs "foo@bar.com")
    # never split a user into two rows. OTP service also normalizes; keeping
    # them in sync prevents "OTP not found" on signup→verify with mixed case.
    email = user_data.email.strip().lower()
    existing = db.query(User).filter(User.email == email).first()
    if existing is not None:
        if not existing.email_verified:
            # Account exists but never verified — let user retry. Re-send OTP
            # and update their password/profile in case they're starting over.
            existing.password_hash = hash_password(user_data.password)
            existing.first_name = user_data.first_name
            existing.last_name = user_data.last_name
            existing.age = user_data.age
            existing.gender = user_data.gender
            db.commit()
            await _issue_and_send_otp(db, existing.email, "signup")
            return SignupResponse(email=existing.email)
        raise HTTPException(status_code=400, detail="Email already registered")

    # New ID must sit above MovieLens placeholder users (so we don't overwrite their ratings).
    max_id = db.query(User).order_by(User.id.desc()).first()
    new_id = (max_id.id + 1) if max_id else 1

    username = _generate_unique_username(
        db, user_data.first_name, user_data.last_name, email
    )

    user = User(
        id=new_id,
        username=username,
        email=email,
        password_hash=hash_password(user_data.password),
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        age=user_data.age,
        gender=user_data.gender,
        email_verified=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    await _issue_and_send_otp(db, user.email, "signup")
    return SignupResponse(email=user.email)


@router.post("/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login and get access token. Requires verified email."""
    email = user_data.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user or user.password_hash == "$movielens_placeholder$":
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.email_verified:
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please verify your email to continue.",
            headers={"X-Auth-Error": "email_not_verified"},
        )

    token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=token)


@router.post("/verify-email", response_model=Token)
def verify_email(data: VerifyEmailRequest, db: Session = Depends(get_db)):
    """Verify a signup OTP. On success the user is marked verified and a JWT
    is returned so the client can log in immediately."""
    email = data.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")

    try:
        otp_service.verify_otp(db, email, data.code, "signup")
    except otp_service.OtpError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    user.email_verified = True
    db.commit()

    token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=token)


@router.post("/resend-otp", response_model=GenericMessage)
async def resend_otp(data: ResendOtpRequest, db: Session = Depends(get_db)):
    """Re-issue an OTP for either signup verification or password reset.

    Always responds with the same message so the endpoint doesn't leak whether
    the email belongs to a registered account."""
    purpose = data.purpose
    if purpose not in {"signup", "reset"}:
        raise HTTPException(status_code=400, detail="Invalid OTP purpose")

    email = data.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    safe_message = GenericMessage(
        message="If that email is registered, a new code is on its way."
    )

    if user is None or user.password_hash == "$movielens_placeholder$":
        return safe_message
    if purpose == "signup" and user.email_verified:
        # Already verified — nothing to do, but return the same response.
        return safe_message

    await _issue_and_send_otp(db, email, purpose)
    return safe_message


@router.post("/forgot-password", response_model=GenericMessage)
async def forgot_password(data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Kick off a password reset via OTP. Always returns 200 to avoid leaking
    whether the email is registered."""
    email = data.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    safe_message = GenericMessage(
        message="If that email is registered, a reset code is on its way."
    )

    if user is None or user.password_hash == "$movielens_placeholder$":
        return safe_message

    await _issue_and_send_otp(db, email, "reset")
    return safe_message


@router.post("/reset-password", response_model=GenericMessage)
def reset_password(data: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Verify a reset OTP and set a new password."""
    email = data.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if user is None or user.password_hash == "$movielens_placeholder$":
        raise HTTPException(status_code=400, detail="Invalid code or email")

    if len(data.new_password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters.",
        )

    try:
        otp_service.verify_otp(db, email, data.code, "reset")
    except otp_service.OtpError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    user.password_hash = hash_password(data.new_password)
    # If user reset before ever verifying, treat completed reset as proof of
    # email ownership so they're not locked into a never-finishable state.
    user.email_verified = True
    db.commit()

    return GenericMessage(message="Password updated. You can now log in.")


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Get current authenticated user info."""
    total_ratings = db.query(Rating).filter(Rating.user_id == current_user.id).count()

    # Favourite genres: pick the genres the user rated highest (≥4★) most often.
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
