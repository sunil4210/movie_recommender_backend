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

# Email / SMTP — used for OTP delivery (signup verification + password reset).
# When SMTP_HOST is empty the email service falls back to logging the OTP to
# stdout, so the backend stays usable in local dev without any external creds.
#
# Free providers tested with this config:
#   - Brevo:   smtp-relay.brevo.com:587   (300/day free, no card)
#   - Mailtrap (sandbox): sandbox.smtp.mailtrap.io:2525  (dev inbox)
#   - Gmail:   smtp.gmail.com:587         (use an app password, ~500/day)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "noreply@cinematch.local")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "CineMatch")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

# OTP policy
OTP_TTL_MINUTES = int(os.getenv("OTP_TTL_MINUTES", "10"))
OTP_MAX_ATTEMPTS = int(os.getenv("OTP_MAX_ATTEMPTS", "5"))
OTP_RESEND_COOLDOWN_SECONDS = int(os.getenv("OTP_RESEND_COOLDOWN_SECONDS", "60"))
OTP_CODE_LENGTH = int(os.getenv("OTP_CODE_LENGTH", "6"))

# Dataset paths
RATINGS_FILE = os.getenv("RATINGS_FILE", "./data/ml-100k/u.data")
MOVIES_FILE = os.getenv("MOVIES_FILE", "./data/ml-100k/u.item")
USERS_FILE = os.getenv("USERS_FILE", "./data/ml-100k/u.user")
