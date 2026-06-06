import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import PROJECT_NAME, API_V1_STR, ALLOWED_ORIGINS, TMDB_API_KEY
from app.database import init_db, load_movielens_data, SessionLocal, Movie
from app.services.recommender import recommender
from app.utils.tmdb import populate_poster_urls, backfill_overviews
from app.routers import auth, movies, recommendations, ratings, favorites, feedback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Initializing database...")
    init_db()

    logger.info("Loading MovieLens data...")
    load_movielens_data()

    logger.info("Loading recommendation engine...")
    recommender.load_data()
    recommender.train_model()

    # Auto-populate poster URLs from TMDB if key is set (background thread).
    # Same pass also writes `overview` for rows without a poster yet. A second
    # pass picks up legacy rows that already have a poster but no overview
    # (the column was added later, so older rows hit `populate_poster_urls`'s
    # `poster_url IS NULL` filter and get skipped).
    if TMDB_API_KEY:
        import threading
        def _populate_tmdb_metadata():
            db = SessionLocal()
            try:
                missing_posters = db.query(Movie).filter(Movie.poster_url.is_(None)).count()
                if missing_posters > 0:
                    logger.info(f"Populating poster URLs for {missing_posters} movies via TMDB (background)...")
                    populate_poster_urls(db, Movie)
                    logger.info("Poster population complete!")

                missing_overviews = db.query(Movie).filter(Movie.overview.is_(None)).count()
                if missing_overviews > 0:
                    logger.info(f"Backfilling overviews for {missing_overviews} movies via TMDB (background)...")
                    backfill_overviews(db, Movie)
                    logger.info("Overview backfill complete!")
            except Exception as e:
                logger.error(f"TMDB metadata population failed: {e}")
            finally:
                db.close()
        threading.Thread(target=_populate_tmdb_metadata, daemon=True).start()
    else:
        logger.warning("TMDB_API_KEY not set — posters will show gradient placeholders. "
                        "Get a free key at https://www.themoviedb.org/settings/api")

    logger.info("Server ready!")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title=PROJECT_NAME,
    version="1.0.0",
    description="Movie Recommendation System API using Collaborative Filtering (SVD)",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=API_V1_STR)
app.include_router(movies.router, prefix=API_V1_STR)
app.include_router(recommendations.router, prefix=API_V1_STR)
app.include_router(ratings.router, prefix=API_V1_STR)
app.include_router(favorites.router, prefix=API_V1_STR)
app.include_router(feedback.router, prefix=API_V1_STR)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "name": PROJECT_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
