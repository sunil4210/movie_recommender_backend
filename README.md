# CineMatch - Movie Recommendation  movie_recommender_backend

A FastAPI-based movie recommendation system using **User-User Collaborative Filtering** trained on the MovieLens 100K dataset. SVD and Item-Item KNN are retained as offline evaluation baselines.

## Features

- **User-User CF Recommendations** - Personalized suggestions via cosine similarity over user rating vectors (K=40 nearest neighbors)
- **Evaluation Baselines** - SVD and Item-Item KNN trained alongside for RMSE/MAE comparison
- **Similar Movies** - Item-item cosine similarity using SVD latent factors (movie detail page)
- **Cold Start Handling** - Popular movies fallback for new users with no rating history
- **Auto-Retraining** - Models retrain automatically after every 100 new ratings
- **JWT Authentication** - Secure user registration (first name + last name + email) and login, with email OTP verification on signup and OTP-based password reset
- **Favorites, Ratings & Reviews** - Users can rate movies (1-5 stars), leave optional text reviews, sort and browse reviews per movie, and manage favorites
- **Thumbs Feedback** - Per-movie thumbs up / thumbs down feedback that feeds recommendation diversification
- **TMDB Poster Integration** - Movie poster URLs fetched from TMDB API (optional)
- **TMDB Synopsis (Overview)** - Plot summary fetched from TMDB alongside posters and shown on the movie details page (expandable "Read more")
- **YouTube Trailers** - Official trailer key fetched lazily from TMDB on first view, cached per movie. Plays inline in a themed modal on both the details page and on hover over movie cards
- **BlurHash Placeholders** - Posters fade in over a low-cost compact hash (computed offline) instead of grey boxes
- **Genre Diversification** - Recommendations balanced across genres

## Tech Stack

- **Framework**: FastAPI
- **Database**: SQLite + SQLAlchemy 2.0
- **ML Library**: Scikit-Surprise (`KNNBasic` user-user; SVD/Item-Item baselines)
- **Auth**: JWT (python-jose) + bcrypt (passlib)
- **Data**: MovieLens 100K (943 users, 1,682 movies, 100,000 ratings)

## Prerequisites

- Python 3.10+
- pip

## Setup & Run

### 1. Clone and navigate

```bash
cd movie_recommender_ movie_recommender_backend
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables (optional)

Create a `.env` file in the ` movie_recommender_backend/` directory:

```env
SECRET_KEY=your-secret-key-here
TMDB_API_KEY=your-tmdb-api-key       # Optional: enables movie posters
DATABASE_URL=sqlite:///./data/database.db
DEBUG=True
ACCESS_TOKEN_EXPIRE_MINUTES=43200

# Email OTP (signup verification + password reset).
# Leave SMTP_HOST empty to log OTP codes to the console instead of sending email —
# this keeps local dev working with zero external setup.
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=noreply@cinematch.local
SMTP_FROM_NAME=CineMatch
SMTP_USE_TLS=true
OTP_TTL_MINUTES=10
OTP_MAX_ATTEMPTS=5
OTP_RESEND_COOLDOWN_SECONDS=60
OTP_CODE_LENGTH=6
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | _(auto-generated)_ | JWT signing key (set for persistent sessions across restarts). Also used as a pepper when hashing OTP codes. |
| `TMDB_API_KEY` | _(empty)_ | TMDB API key for poster URLs ([get one free](https://www.themoviedb.org/settings/api)) |
| `DATABASE_URL` | `sqlite:///./data/database.db` | Database connection string |
| `DEBUG` | `True` | Enable debug mode |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `43200` | Token expiry (30 days) |
| `RETRAIN_THRESHOLD` | `100` | New ratings before auto-retrain |
| `MODEL_PATH` | `./models/svd_model.pkl` | Cached SVD model path |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` | _(empty)_ | SMTP credentials for sending OTP emails. Empty → codes logged to console (dev fallback). Free providers: Brevo (300/day), Mailtrap sandbox, Gmail app password. |
| `SMTP_FROM` / `SMTP_FROM_NAME` | `noreply@cinematch.local` / `CineMatch` | "From" address shown to recipients. With a real SMTP host this must be a sender address verified with that provider. |
| `SMTP_USE_TLS` | `true` | Use STARTTLS on the SMTP connection |
| `OTP_TTL_MINUTES` | `10` | How long an OTP stays valid |
| `OTP_MAX_ATTEMPTS` | `5` | Wrong attempts allowed before the code is locked out |
| `OTP_RESEND_COOLDOWN_SECONDS` | `60` | Minimum gap between resends for the same email + purpose |
| `OTP_CODE_LENGTH` | `6` | Number of digits in the generated code |

### 5. Run the server

> **Important:** You must run this command from the ` movie_recommender_backend/` directory (not ` movie_recommender_backend/app/`), and the virtual environment must be activated first.

```bash
# Make sure you're in the  movie_recommender_backend/ directory
cd  movie_recommender_backend

# Activate venv (if not already active)
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server starts at `http://localhost:8000`.

On first startup, the server will:
1. Initialize the SQLite database (applying any column migrations such as `tmdb_id`, `trailer_key`, `overview`, `blur_hash`)
2. Load MovieLens 100K dataset into the database
3. Train the User-User KNN model (production) + SVD + Item-Item KNN baselines (~10-30 seconds first time, ~1 second from cache afterward)
4. Optionally populate TMDB poster URLs **and movie overviews** in the background (per-movie progress is logged to stdout). A second pass picks up legacy rows that already had a poster but were missing the new `overview` column.

### 6. Verify

- Health check: `http://localhost:8000/health`
- API docs (Swagger): `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | Register a new user (first name + last name + email + password). Creates an unverified account and emails a 6-digit OTP. |
| POST | `/api/auth/verify-email` | Verify the signup OTP. On success the account is marked verified and a JWT is returned. |
| POST | `/api/auth/resend-otp` | Re-issue an OTP (`purpose` = `signup` or `reset`). Rate-limited per email. |
| POST | `/api/auth/forgot-password` | Request a password-reset OTP. Always returns 200 to avoid leaking which emails are registered. |
| POST | `/api/auth/reset-password` | Submit the reset OTP plus a new password. |
| POST | `/api/auth/login` | Login and get JWT token. Returns 403 if the email is not verified. |
| GET | `/api/auth/me` | Get current user profile |
| PUT | `/api/auth/profile` | Update first name, last name, age, gender |
| PUT | `/api/auth/change-password` | Change password |

#### Email OTP flow

Signup verification:

1. `POST /api/auth/signup` → creates the user with `email_verified=false`, emails a 6-digit OTP, returns `{email, email_verified: false}`.
2. Client renders the OTP screen and calls `POST /api/auth/verify-email {email, code}`.
3. On success the user is marked verified and a JWT is returned — the client can sign in immediately.

Password reset:

1. `POST /api/auth/forgot-password {email}` — always returns 200, emails an OTP only if the address exists.
2. `POST /api/auth/reset-password {email, code, new_password}` — verifies the OTP and updates the password.

Notes:
- OTPs are stored as `HMAC-SHA256(SECRET_KEY, code)` — plaintext is never persisted.
- A new OTP for the same `(email, purpose)` supersedes any previous unconsumed code.
- Issuing too quickly returns `429` with a `Retry-After` header.
- Wrong codes increment an attempt counter; after `OTP_MAX_ATTEMPTS` the code is locked and the user must request a new one.

### Movies
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/movies` | List movies (paginated) |
| GET | `/api/movies/search` | Search movies by title/genre (multi-token OR + relevance ranking) |
| GET | `/api/movies/popular` | Top-rated movies (≥20 ratings) |
| GET | `/api/movies/trending` | Movies with the most recent rating activity |
| GET | `/api/movies/{id}` | Get movie by ID (includes `overview`, `poster_url`, `blur_hash`) |
| GET | `/api/movies/{id}/trailer` | Get the YouTube trailer key for a movie. 200 returns `{youtube_key, embed_url}`; 404 when TMDB has no trailer (cached so we don't re-query) |
| POST | `/api/movies/populate-posters` | Admin: backfill `poster_url` + `tmdb_id` + `overview` for movies missing a poster |
| POST | `/api/movies/backfill-overviews` | Admin: backfill `overview` for movies that already have a poster but no synopsis |

### Recommendations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/recommendations/{user_id}?algorithm=user_user` | Get personalized recommendations (frontend always passes `user_user`) |
| GET | `/api/recommendations/{user_id}/similar/{movie_id}` | Get similar movies (SVD item factor cosine) |
| POST | `/api/recommendations/refresh` | Force model retrain |
| GET | `/api/recommendations/metrics` | RMSE / MAE / precision@10 / recall@10 across all 3 algorithms |

### Ratings & Reviews
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ratings` | Submit a rating (1-5 stars, optional `comment` ≤ 1000 chars) |
| GET | `/api/ratings/user/{user_id}` | Get user's ratings |
| PUT | `/api/ratings/{rating_id}` | Update a rating and/or its comment |
| DELETE | `/api/ratings/{rating_id}` | Delete a rating |
| GET | `/api/ratings/movie/{movie_id}` | All ratings for a movie |
| GET | `/api/ratings/movie/{movie_id}/reviews` | Ratings with non-empty comments. Supports `sort=newest\|oldest\|highest\|lowest` and `pin_user_id` (current user's review floats to top) |
| GET | `/api/ratings/user/{user_id}/movie/{movie_id}` | The current user's rating for a given movie (used to edit-in-place) |

### Favorites
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/favorites` | Add movie to favorites |
| GET | `/api/favorites/user/{user_id}` | Get user's favorites |
| DELETE | `/api/favorites/{user_id}/{movie_id}` | Remove from favorites |

### Feedback
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/feedback` | Submit thumbs up / thumbs down feedback for a movie |
| GET | `/api/feedback/user/{user_id}` | All feedback by a user |
| GET | `/api/feedback/movie/{movie_id}` | Aggregate thumbs up / thumbs down counts for a movie |

## Smoke Testing

A read-only smoke test in `tests/smoke_test.py` hits every public endpoint against a
running backend and asserts on status codes + key response fields (including
`overview` on `MovieResponse`, trailer key shape, auth boundaries).

```bash
# Server must be running on :8000
./venv/bin/python -m tests.smoke_test
```

Override the base URL with `CINEMATCH_BASE_URL=http://staging:8000`.

## Project Structure

```
 movie_recommender_backend/
├── app/
│   ├── main.py                 # FastAPI app, startup lifecycle
│   ├── config.py               # Environment variables and settings
│   ├── database.py             # SQLAlchemy models + MovieLens data loader
│   ├── models.py               # Pydantic request/response schemas
│   ├── auth.py                 # JWT token + password hashing
│   ├── routers/
│   │   ├── auth.py             # Auth endpoints
│   │   ├── movies.py           # Movie endpoints
│   │   ├── recommendations.py  # Recommendation endpoints
│   │   ├── ratings.py          # Rating endpoints
│   │   ├── favorites.py        # Favorites endpoints
│   │   └── feedback.py         # Feedback endpoints
│   ├── services/
│   │   ├── recommender.py      # User-User KNN (prod) + SVD/Item-Item (baselines) + caching
│   │   ├── movie_service.py    # Movie queries and statistics
│   │   ├── email_service.py    # SMTP delivery of OTP emails (console fallback when SMTP not configured)
│   │   └── otp_service.py      # OTP issue + verify (hashed codes, TTL, attempt cap, cooldown)
│   └── utils/
│       ├── tmdb.py             # TMDB integration: posters, tmdb_id, trailer keys, overviews
│       └── helpers.py          # Utility functions
├── data/
│   └── ml-100k/                # MovieLens 100K dataset
├── models/
│   └── svd_model.pkl           # Cached User-User + SVD + Item-Item KNN models
├── tests/
│   └── smoke_test.py           # End-to-end endpoint smoke test (runs against live server)
├── scripts/
│   └── compute_blurhashes.py   # One-shot job to compute blur_hash for every poster
├── requirements.txt            # Python dependencies
└── RECOMMENDATION_SYSTEM_DOCS.md  # Technical documentation
```

## How the Recommendation Engine Works

See [RECOMMENDATION_SYSTEM_DOCS.md](RECOMMENDATION_SYSTEM_DOCS.md) for full technical documentation including:
- User-User Collaborative Filtering algorithm (cosine similarity, K=40 neighbors)
- Why User-User was chosen over SVD / Item-Item for production
- Data flow diagrams
- Cold start handling
- Genre diversification
- Model caching and retraining strategy
- Viva quick-reference pitch 
