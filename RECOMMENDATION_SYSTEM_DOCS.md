# CineMatch Recommendation System — Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Complete Data Flow](#complete-data-flow)
3. [Algorithm: User-User Collaborative Filtering](#algorithm-user-user-collaborative-filtering)
4. [How It Works Step-by-Step](#how-it-works-step-by-step)
5. [Cold Start Problem & Solution](#cold-start-problem--solution)
6. [Similar Movies](#similar-movies)
7. [Feedback Loop](#feedback-loop)
8. [Genre Diversification](#genre-diversification)
9. [Model Caching & Retraining](#model-caching--retraining)
10. [Evaluation Baselines (SVD, Item-Item)](#evaluation-baselines)
11. [API Endpoints](#api-endpoints)
12. [Architecture Diagram](#architecture-diagram)
13. [File Reference](#file-reference)

---

## System Overview

CineMatch uses **User-User Collaborative Filtering** to recommend movies. The system finds users with similar rating patterns and recommends movies those neighbors rated highly that the target user has not yet seen.

**Production algorithm**: User-User Collaborative Filtering (cosine similarity)
**Library**: scikit-surprise (`KNNBasic`, `user_based=True`)
**Dataset**: MovieLens 100K (943 users, 1,682 movies, 100,000 ratings)
**Framework**: FastAPI (Python)

SVD and Item-Item KNN are also trained at startup and retained in code as **evaluation baselines** (used for RMSE/MAE comparison and similar-movies lookup). Production traffic uses User-User CF only.

---

## Complete Data Flow

### Startup Flow (Server Boot)
```
Server starts (uvicorn)
    │
    ▼
1. init_db()                    ← Create SQLite tables if not exist
    │
    ▼
2. load_movielens_data()        ← Parse MovieLens files → insert into DB
    │                              - u.item → movies table (1,682 movies)
    │                              - u.data → ratings table (100,000 ratings)
    │                              - u.user → users table (943 users)
    │
    ▼
3. recommender.load_data()      ← Load ratings + movies from DB into pandas DataFrames
    │
    ▼
4. recommender.train_model()    ← Check for cached model (svd_model.pkl)
    │                              - If cached: load from disk (~1 second)
    │                              - If not: train User-User KNN (~5-15 seconds)
    │                                        + SVD + Item-Item KNN (baselines)
    │
    ▼
Server ready. Accepting requests.
```

### Recommendation Request Flow
```
User opens app → GET /api/recommendations/{user_id}?algorithm=user_user
    │
    ▼
Is user in training set?
    │
    ├── NO (new user) ──→ COLD START: return popular movies
    │                      (movies with ≥20 ratings, sorted by avg rating)
    │
    └── YES ──→ Get all movies user has NOT rated
                    │
                    ▼
                For each unrated movie:
                    predict rating using User-User KNN
                    (weighted avg of K=40 nearest neighbors' ratings)
                    │
                    ▼
                Filter: keep only predicted rating ≥ 3.5
                    │
                    ▼
                Sort by predicted rating (descending)
                    │
                    ▼
                Diversify genres (no more than n/3 per genre)
                    │
                    ▼
                Return top N movies (default: 10)
```

### Rating Submission Flow (Feedback Loop)
```
User rates a movie → POST /api/ratings
    │
    ▼
Save rating to database
    │
    ▼
Increment new_ratings_count
    │
    ▼
Count ≥ 100 (RETRAIN_THRESHOLD)?
    │
    ├── NO ──→ Done (next recommendation uses existing model)
    │
    └── YES ──→ Reload all data from DB
                    │
                    ▼
                Retrain User-User KNN with new data
                    │
                    ▼
                Save new model to disk
                    │
                    ▼
                Reset counter to 0
                Future recommendations reflect new ratings
```

---

## Algorithm: User-User Collaborative Filtering

### Intuition

> "Users who rated movies the same way you did in the past are likely to rate new movies the same way you would."

Find the K users whose taste matches the target user, then recommend what those neighbors liked.

### The Math

#### Step 1 — Cosine similarity between two users
```
              Σ (r_u,i · r_v,i)
sim(u, v) = ───────────────────────────────
            √(Σ r_u,i²) · √(Σ r_v,i²)

Where:
  r_u,i = rating user u gave movie i
  Sum is over movies that BOTH users have rated
```

Result is in `[-1, 1]`. Higher = more similar taste.

#### Step 2 — Predict a rating for an unseen movie
```
            Σ (sim(u, v) · r_v,i)
r̂_u,i = ──────────────────────────────
              Σ |sim(u, v)|

Where v iterates over the K=40 most-similar users who rated movie i
```

It's a weighted average — closer neighbors count more.

### Parameters

```python
KNNBasic(
    k=40,                                 # 40 nearest neighbors
    sim_options={
        "name": "cosine",                 # cosine similarity
        "user_based": True,               # user-user (not item-item)
    },
    verbose=False,
)
```

| Parameter | Value | Why |
|-----------|-------|-----|
| k | 40 | Standard choice for MovieLens 100K — balances noise vs. coverage |
| name | cosine | Robust to differing rating scales between users |
| user_based | True | Compute similarity over users (the U×U matrix) |

### Training Process

1. Load all 100,000 `(user, movie, rating)` triplets.
2. Build trainset using Surprise's `Reader`.
3. `KNNBasic.fit(trainset)` computes the **943 × 943** user-user cosine similarity matrix.
4. Matrix cached in memory + serialized to `models/svd_model.pkl`.

The similarity matrix is the "model". No iterative training — just one matrix computation.

### Why User-User for Production

| Property | Benefit |
|----------|---------|
| Explainability | "Users like you rated this 5★" — one-sentence reason per recommendation |
| Simplicity | One formula (cosine), no hyperparameter tuning beyond K |
| No latent factors | Predictions traceable to actual rating data, not abstract dimensions |
| Defensible math | Whiteboard-friendly: matrix → row vector → cosine → weighted avg |

Trade-offs (which we acknowledge): does not scale linearly with users, and sparser users get noisier predictions. For 943 users × 1,682 movies, it is well within feasible range.

---

## How It Works Step-by-Step

### Example: Recommending for User 1

**Step 1** — User 1 has rated 272 movies (from MovieLens data).

**Step 2** — Compute cosine similarity between User 1 and every other user.

```
sim(User1, User50)  = 0.87  ← very similar taste
sim(User1, User132) = 0.81
sim(User1, User402) = 0.76
...
sim(User1, User800) = 0.12  ← unrelated
```

**Step 3** — Pick the top K=40 most-similar users.

**Step 4** — For each unrated movie, predict User 1's rating:
```
"Indiana Jones" rated by 27 of User1's 40 neighbors
average weighted by similarity → predicted 4.78 ★
```

**Step 5** — Drop predictions below 3.5★, sort by predicted rating, diversify by genre, return top 10.

### Why It Works

If User 1 and User 50 agreed on 200 past movies, and User 50 rated "Indiana Jones" 5★, it is a strong signal User 1 will also like it. We aggregate that signal across many similar users to denoise individual idiosyncrasies.

---

## Cold Start Problem & Solution

### The Problem
New users have no ratings → cannot compute similarity → no neighbors → no predictions.

### The Solution: Popular Movies Fallback

```python
def _get_popular_movies(n, db):
    # Query movies with ≥ 20 ratings
    # Sort by average rating (descending)
    # Return top N
```

Example output:
```
1. Close Shave, A (1995)        — 4.49★ (112 ratings)
2. Schindler's List (1993)      — 4.47★ (298 ratings)
3. Wrong Trousers, The (1993)   — 4.47★ (118 ratings)
```

Once a new user rates ~5-10 movies and the model retrains, they begin receiving personalized User-User recommendations.

---

## Similar Movies

The "Similar Movies" feature (movie-detail page) uses **item latent factors from SVD** (Item-Item cosine over the SVD `qi` matrix). It is independent of the production recommendation path.

```
GET /api/recommendations/{user_id}/similar/{movie_id}
```

Steps:
1. Get the target movie's latent vector (`model.qi[inner_id]`).
2. Cosine similarity vs every other movie's vector.
3. Sort descending, return top N.

Kept because it provides high-quality "more like this" suggestions on the movie detail page and is independent of the per-user recommendation choice.

---

## Feedback Loop

The system improves over time as users submit ratings:

```
User rates movie
       │
       ▼
Rating saved to DB ──→ new_ratings_count++
       │
       ▼
  Count ≥ 100? ──YES──→ Retrain User-User KNN with ALL data
       │                 (including new ratings)
       │
       NO
       │
       ▼
  Continue with current model
  (new rating will be included in next retrain)
```

### Manual Retrain
```
POST /api/recommendations/refresh
```
Forces immediate model retrain. Useful after bulk rating imports.

---

## Genre Diversification

Without diversification, top recommendations may all be the same genre (e.g., all Drama).

### Algorithm
```python
max_per_genre = max(2, n // 3)  # e.g., for top 10: max 3 per genre

for each candidate (sorted by predicted rating):
    if any genre already has max_per_genre selections:
        skip (unless near end of list)
    else:
        add to results
```

### Example (n=10)
```
Before diversification:          After diversification:
1. Drama (4.89)                  1. Drama (4.89)
2. Drama (4.85)                  2. Drama (4.85)
3. Drama (4.82)                  3. Drama (4.82)
4. Drama (4.80)  ← skipped       4. Sci-Fi (4.75)    ← promoted
5. Drama (4.78)  ← skipped       5. Comedy (4.70)    ← promoted
6. Sci-Fi (4.75)                 6. Thriller (4.65)   ← promoted
7. Comedy (4.70)                 7. Sci-Fi (4.60)
8. Thriller (4.65)               8. Romance (4.55)    ← promoted
```

---

## Model Caching & Retraining

### Cache Strategy
- Trained models pickled to `models/svd_model.pkl`.
- On startup: load cached models first (~1 second vs. ~10-30 seconds for training).
- After retraining: overwrite cache.

### What Gets Cached
```python
{
    "model": SVD_model,              # baseline, used for similar-movies
    "user_user_model": KNNBasic,     # PRODUCTION model
    "item_item_model": KNNBasic,     # baseline
    "trainset": trainset,            # raw↔inner ID mapping
}
```

### Retraining Triggers
1. **Automatic**: every 100 new ratings (`RETRAIN_THRESHOLD`).
2. **Manual**: `POST /api/recommendations/refresh`.
3. **Startup**: if no cached model exists.

---

## Evaluation Baselines

Two additional algorithms are trained for offline evaluation only — they are not exposed to UI traffic:

### SVD (Singular Value Decomposition)
Matrix factorization, Netflix-Prize-era algorithm. Used for:
- Cross-validated **RMSE / MAE** scores (reported via `/api/recommendations/metrics`).
- "Similar Movies" cosine on the learned item factor matrix.

Parameters: `n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02`.

### Item-Item KNN
Cosine similarity between movies (transpose of User-User). Used for:
- Comparative RMSE evaluation.

Parameters: `k=40, cosine similarity, user_based=False`.

### Why Keep Them
Holding baselines lets us justify the User-User choice with numbers — "we picked User-User after comparing it against SVD and Item-Item on RMSE/MAE." See `/api/recommendations/metrics` for live numbers.

---

## API Endpoints

### Get Personalized Recommendations
```
GET /api/recommendations/{user_id}?n=10&algorithm=user_user
```
Frontend always passes `algorithm=user_user`. The `algorithm` query param is preserved to allow ad-hoc baseline comparison (e.g., from curl).

**Response**:
```json
[
    {
        "movie_id": 513,
        "title": "Third Man, The (1949)",
        "genres": ["Mystery", "Thriller"],
        "predicted_rating": 4.89,
        "poster_url": "https://...",
        "reason": "Users with similar taste rated this 4.89★"
    }
]
```

### Get Similar Movies
```
GET /api/recommendations/{user_id}/similar/{movie_id}?n=10
```

### Refresh Model
```
POST /api/recommendations/refresh
```

### Get Evaluation Metrics
```
GET /api/recommendations/metrics
```
Returns RMSE/MAE/precision@10/recall@10 for the trained models.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Flutter Web App                        │
│  Login → Home → Recommendations Tab → Movie Details      │
│         │                │                                │
│         │                └── algorithm=user_user (fixed)  │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP REST API
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                         │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐ │
│  │  Auth    │  │  Movies  │  │  Ratings  │  │Favorites│ │
│  │  Router  │  │  Router  │  │  Router   │  │ Router  │ │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └───┬────┘ │
│       │              │              │             │      │
│       ▼              ▼              ▼             ▼      │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Recommendations Router               │   │
│  │  GET /recommendations/{user_id}?algorithm=...     │   │
│  │  GET /recommendations/{uid}/similar/{mid}         │   │
│  │  POST /recommendations/refresh                    │   │
│  │  GET /recommendations/metrics                     │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │           RecommenderEngine (singleton)           │   │
│  │                                                    │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │  PRODUCTION:                                 │ │   │
│  │  │  user_user_model = KNNBasic(k=40, cosine,    │ │   │
│  │  │                              user_based=True)│ │   │
│  │  │  → User × User similarity matrix             │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  │                                                    │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │  BASELINES (evaluation only):                │ │   │
│  │  │  model = SVD(n_factors=100, n_epochs=20)     │ │   │
│  │  │  item_item_model = KNNBasic(... user_based=False)│   │
│  │  └─────────────────────────────────────────────┘ │   │
│  │                                                    │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │  Ratings DataFrame (100K rows)               │ │   │
│  │  │  Movies DataFrame  (1,682 rows + genres)     │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  │                                                    │   │
│  │  Methods:                                          │   │
│  │  ├── load_data()             Load from DB           │   │
│  │  ├── train_model()           Train all 3 (cached)   │   │
│  │  ├── get_recommendations()   User-User predict + filter + diversify │
│  │  ├── get_similar_movies()    SVD qi cosine          │   │
│  │  └── record_new_rating()     Track for retrain      │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │              SQLite Database                       │   │
│  │  users    │  movies    │  ratings    │  favorites  │   │
│  │  (943+)   │  (1,682)   │  (100,000+) │  (user)    │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  models/svd_model.pkl (all 3 models cached)       │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## File Reference

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, startup lifecycle (init DB, load data, train models) |
| `app/config.py` | Environment variables and settings |
| `app/database.py` | SQLAlchemy models (User, Movie, Rating, Favorite) + data loader |
| `app/models.py` | Pydantic schemas for request/response validation |
| `app/auth.py` | JWT token creation, password hashing, auth middleware |
| `app/services/recommender.py` | **Core**: User-User KNN training + prediction; SVD/Item-Item baselines; caching |
| `app/services/movie_service.py` | Movie queries: search, popular, trending |
| `app/routers/auth.py` | Signup (first/last name), login, profile, change password |
| `app/routers/movies.py` | List, search, get-by-id, popular, trending endpoints |
| `app/routers/recommendations.py` | Recommendations, similar movies, refresh, metrics |
| `app/routers/ratings.py` | Submit, update, delete, list ratings |
| `app/routers/favorites.py` | Add, list, remove favorites |
| `data/ml-100k/` | MovieLens 100K dataset files |
| `models/svd_model.pkl` | Cached User-User + SVD + Item-Item models |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Dataset size | 100,000 ratings |
| Users | 943 |
| Movies | 1,682 |
| Production algorithm | User-User KNN (k=40, cosine) |
| Similarity matrix size | 943 × 943 |
| Min predicted rating threshold | 3.5 |
| Cold start fallback | Popular movies (≥20 ratings) |
| Auto-retrain threshold | Every 100 new ratings |
| Training time (User-User) | ~5-15 seconds on 100K |
| Prediction time | < 1 second for full top-10 |
| Model cache load | ~1 second |

---

## Viva Quick-Reference (One-Page Pitch)

> **"CineMatch recommends movies using User-User Collaborative Filtering.** We compute a cosine-similarity matrix between every pair of users based on their rating history. For a target user, we pick the K=40 most-similar users (neighbors), then predict an unseen movie's rating as a similarity-weighted average of those neighbors' ratings on that movie. Predictions below 3.5★ are dropped, the rest are sorted descending and passed through a genre-diversifier (at most n/3 per genre) so the top 10 are not all the same genre.
>
> For new users with no ratings, we fall back to popular movies (≥20 ratings, sorted by average rating). The model retrains every 100 new ratings.
>
> We also train SVD and Item-Item KNN as **evaluation baselines** to justify choosing User-User on RMSE/MAE; SVD's item factors additionally power the 'Similar Movies' lookup on movie detail pages."

---
