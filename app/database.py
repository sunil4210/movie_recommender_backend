import os
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey,
    UniqueConstraint, CheckConstraint, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

from app.config import DATABASE_URL, RATINGS_FILE, MOVIES_FILE, USERS_FILE

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite only
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- SQLAlchemy ORM Models ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    release_year = Column(Integer, nullable=True)
    genres = Column(String(255), nullable=True)  # Pipe-separated
    imdb_url = Column(String(500), nullable=True)
    poster_url = Column(String(500), nullable=True)
    # 4x3-component BlurHash string (~30 chars) computed offline from poster image.
    # Rendered client-side as a low-cost placeholder while the full poster loads.
    blur_hash = Column(String(64), nullable=True)


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie"),
        CheckConstraint("rating >= 1.0 AND rating <= 5.0", name="ck_rating_range"),
    )


class Favorite(Base):
    __tablename__ = "favorites"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie_fav"),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    feedback_type = Column(String(20), nullable=False)  # 'thumbs_up' or 'thumbs_down'
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie_feedback"),
    )


# --- Dependency ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Database Initialization ---

def init_db():
    """Create all tables and apply any lightweight column migrations."""
    Base.metadata.create_all(bind=engine)
    _ensure_movie_blur_hash_column()


def _ensure_movie_blur_hash_column() -> None:
    """Add movies.blur_hash column on legacy SQLite DBs without a migration tool."""
    from sqlalchemy import text, inspect

    inspector = inspect(engine)
    if "movies" not in inspector.get_table_names():
        return
    cols = {c["name"] for c in inspector.get_columns("movies")}
    if "blur_hash" in cols:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE movies ADD COLUMN blur_hash VARCHAR(64)"))


def load_movielens_data():
    """Load MovieLens 100K data into database."""
    db = SessionLocal()
    try:
        # Check if data already loaded
        if db.query(Movie).count() > 0:
            print("Data already loaded, skipping.")
            return

        print("Loading MovieLens 100K dataset...")

        # Load movies from u.item
        genre_names = [
            "unknown", "Action", "Adventure", "Animation", "Children's",
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
            "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
            "Sci-Fi", "Thriller", "War", "Western"
        ]

        movies_df = pd.read_csv(
            MOVIES_FILE,
            sep="|",
            encoding="latin-1",
            header=None,
            names=["id", "title", "release_date", "video_release_date", "imdb_url"]
                  + genre_names
        )

        for _, row in movies_df.iterrows():
            # Extract genres
            movie_genres = [g for g in genre_names if row.get(g, 0) == 1]
            genres_str = "|".join(movie_genres) if movie_genres else "unknown"

            # Extract year from title (e.g., "Toy Story (1995)")
            title = row["title"]
            year = None
            if "(" in title and ")" in title:
                try:
                    year_str = title[title.rfind("(") + 1:title.rfind(")")]
                    year = int(year_str)
                except ValueError:
                    pass

            movie = Movie(
                id=int(row["id"]),
                title=title,
                release_year=year,
                genres=genres_str,
                imdb_url=row.get("imdb_url", None),
                poster_url=None
            )
            db.merge(movie)

        print(f"Loaded {len(movies_df)} movies.")

        # Load ratings from u.data
        ratings_df = pd.read_csv(
            RATINGS_FILE,
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

        # First, create placeholder users for MovieLens users
        unique_users = ratings_df["user_id"].unique()
        for uid in unique_users:
            user = User(
                id=int(uid),
                username=f"movielens_user_{uid}",
                email=f"user{uid}@movielens.org",
                password_hash="$movielens_placeholder$",  # Not a real login
                age=None,
                gender=None,
            )
            db.merge(user)

        # Try to load user demographics from u.user
        if os.path.exists(USERS_FILE):
            users_df = pd.read_csv(
                USERS_FILE,
                sep="|",
                header=None,
                names=["id", "age", "gender", "occupation", "zip_code"]
            )
            for _, row in users_df.iterrows():
                user = db.query(User).filter(User.id == int(row["id"])).first()
                if user:
                    user.age = int(row["age"]) if pd.notna(row["age"]) else None
                    user.gender = str(row["gender"]) if pd.notna(row["gender"]) else None

        print(f"Loaded {len(unique_users)} users.")

        # Load ratings
        for _, row in ratings_df.iterrows():
            rating = Rating(
                user_id=int(row["user_id"]),
                movie_id=int(row["movie_id"]),
                rating=float(row["rating"]),
                timestamp=datetime.fromtimestamp(int(row["timestamp"]))
            )
            db.merge(rating)

        db.commit()
        print(f"Loaded {len(ratings_df)} ratings.")
        print("MovieLens data loaded successfully!")

    except Exception as e:
        db.rollback()
        print(f"Error loading data: {e}")
        raise
    finally:
        db.close()
