# CineMatch - Movie Recommendation Backend

A FastAPI-based movie recommendation system using **User-User Collaborative Filtering** trained on the MovieLens 100K dataset. SVD and Item-Item KNN are retained as offline evaluation baselines.

## Features

- **User-User CF Recommendations** - Personalized suggestions via cosine similarity over user rating vectors (K=40 nearest neighbors)
- **Evaluation Baselines** - SVD and Item-Item KNN trained alongside for RMSE/MAE comparison
- **Similar Movies** - Item-item cosine similarity using SVD latent factors (movie detail page)
- **Cold Start Handling** - Popular movies fallback for new users with no rating history
- **Auto-Retraining** - Models retrain automatically after every 100 new ratings
- **JWT Authentication** - Secure user registration (first name + last name + email) and login
- **Favorites & Ratings** - Users can rate movies and manage favorites
- **TMDB Poster Integration** - Movie poster URLs fetched from TMDB API (optional)
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
cd backend
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

Create a `.env` file in the `backend/` directory:

```env
SECRET_KEY=your-secret-key-here
TMDB_API_KEY=your-tmdb-api-key       # Optional: enables movie posters
DATABASE_URL=sqlite:///./data/database.db
DEBUG=True
ACCESS_TOKEN_EXPIRE_MINUTES=43200
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | _(auto-generated)_ | JWT signing key (set for persistent sessions across restarts) |
| `TMDB_API_KEY` | _(empty)_ | TMDB API key for poster URLs ([get one free](https://www.themoviedb.org/settings/api)) |
| `DATABASE_URL` | `sqlite:///./data/database.db` | Database connection string |
| `DEBUG` | `True` | Enable debug mode |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `43200` | Token expiry (30 days) |
| `RETRAIN_THRESHOLD` | `100` | New ratings before auto-retrain |
| `MODEL_PATH` | `./models/svd_model.pkl` | Cached SVD model path |

### 5. Run the server

> **Important:** You must run this command from the `backend/` directory (not `backend/app/`), and the virtual environment must be activated first.

```bash
# Make sure you're in the backend/ directory
cd backend

# Activate venv (if not already active)
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server starts at `http://localhost:8000`.

On first startup, the server will:
1. Initialize the SQLite database
2. Load MovieLens 100K dataset into the database
3. Train the User-User KNN model (production) + SVD + Item-Item KNN baselines (~10-30 seconds first time, ~1 second from cache afterward)
4. Optionally populate TMDB poster URLs in the background

### 6. Verify

- Health check: `http://localhost:8000/health`
- API docs (Swagger): `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | Register a new user (first name + last name + email + password — username auto-generated internally) |
| POST | `/api/auth/login` | Login and get JWT token |
| GET | `/api/auth/me` | Get current user profile |
| PUT | `/api/auth/profile` | Update first name, last name, age, gender |
| PUT | `/api/auth/change-password` | Change password |

### Movies
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/movies` | List movies (paginated) |
| GET | `/api/movies/search` | Search movies by title/genre |
| GET | `/api/movies/{id}` | Get movie by ID |
| GET | `/api/movies/popular` | Get popular movies |
| GET | `/api/movies/trending` | Get trending movies |

### Recommendations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/recommendations/{user_id}?algorithm=user_user` | Get personalized recommendations (frontend always passes `user_user`) |
| GET | `/api/recommendations/{user_id}/similar/{movie_id}` | Get similar movies (SVD item factor cosine) |
| POST | `/api/recommendations/refresh` | Force model retrain |
| GET | `/api/recommendations/metrics` | RMSE / MAE / precision@10 / recall@10 across all 3 algorithms |

### Ratings
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ratings` | Submit a rating |
| GET | `/api/ratings/user/{user_id}` | Get user's ratings |
| PUT | `/api/ratings/{rating_id}` | Update a rating |
| DELETE | `/api/ratings/{rating_id}` | Delete a rating |

### Favorites
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/favorites` | Add movie to favorites |
| GET | `/api/favorites/user/{user_id}` | Get user's favorites |
| DELETE | `/api/favorites/{user_id}/{movie_id}` | Remove from favorites |

### Feedback
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/feedback` | Submit feedback on recommendations |

## Project Structure

```
backend/
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
│   │   └── movie_service.py    # Movie queries and statistics
│   └── utils/
│       ├── tmdb.py             # TMDB API integration
│       └── helpers.py          # Utility functions
├── data/
│   └── ml-100k/                # MovieLens 100K dataset
├── models/
│   └── svd_model.pkl           # Cached User-User + SVD + Item-Item KNN models
├── tests/                      # Test directory
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
