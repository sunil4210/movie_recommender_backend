from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime


# --- Auth Models ---

class UserSignup(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    age: Optional[int] = None
    gender: Optional[str] = None


class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class ChangePassword(BaseModel):
    current_password: str
    new_password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    total_ratings: int = 0
    favorite_genres: List[str] = []
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- Movie Models ---

class MovieResponse(BaseModel):
    id: int
    title: str
    genres: List[str]
    year: Optional[int] = None
    average_rating: Optional[float] = None
    total_ratings: int = 0
    poster_url: Optional[str] = None
    blur_hash: Optional[str] = None

    class Config:
        from_attributes = True


class MovieListResponse(BaseModel):
    movies: List[MovieResponse]
    total: int
    page: int
    per_page: int


# --- Rating Models ---

class RatingCreate(BaseModel):
    user_id: int
    movie_id: int
    rating: float  # 1.0 to 5.0


class RatingUpdate(BaseModel):
    rating: float


class RatingResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    movie_title: str = ""
    rating: float
    timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- Recommendation Models ---

class RecommendationResponse(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    predicted_rating: float
    poster_url: Optional[str] = None
    blur_hash: Optional[str] = None
    reason: Optional[str] = None


# --- Favorite Models ---

class FeedbackCreate(BaseModel):
    user_id: int
    movie_id: int
    feedback_type: str  # 'thumbs_up' or 'thumbs_down'


class FeedbackResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    feedback_type: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class FavoriteCreate(BaseModel):
    user_id: int
    movie_id: int


class FavoriteResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    movie_title: str = ""
    genres: List[str] = []
    poster_url: Optional[str] = None
    blur_hash: Optional[str] = None
    average_rating: Optional[float] = None
    total_ratings: int = 0
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
