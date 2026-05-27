import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/database.db")

# Security
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    import secrets
    SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "43200"))

# ML Model
MODEL_PATH = os.getenv("MODEL_PATH", "./models/svd_model.pkl")
DEFAULT_RECOMMENDATION_COUNT = int(os.getenv("DEFAULT_RECOMMENDATION_COUNT", "10"))
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "100"))

# API
API_V1_STR = os.getenv("API_V1_STR", "/api")
PROJECT_NAME = os.getenv("PROJECT_NAME", "CineMatch Movie Recommender API")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:*,http://127.0.0.1:*").split(",")

# TMDB
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

# Dataset paths
RATINGS_FILE = os.getenv("RATINGS_FILE", "./data/ml-100k/u.data")
MOVIES_FILE = os.getenv("MOVIES_FILE", "./data/ml-100k/u.item")
USERS_FILE = os.getenv("USERS_FILE", "./data/ml-100k/u.user")
