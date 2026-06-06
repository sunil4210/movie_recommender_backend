import logging
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate
from sqlalchemy.orm import Session

from app.config import MODEL_PATH, DEFAULT_RECOMMENDATION_COUNT
from app.database import SessionLocal, Rating, Movie

logger = logging.getLogger(__name__)


class RecommenderEngine:
    """Collaborative filtering recommendation engine.

    Production algorithm: **User-User Collaborative Filtering** (KNNBasic, cosine, K=40).
        For a target user, find K most-similar users (by rating-vector cosine), then
        predict an unseen movie's rating as a similarity-weighted average of those
        neighbors' ratings on it.

    Baselines retained for offline RMSE/MAE comparison and the "Similar Movies" feature:
        - SVD (Singular Value Decomposition): 100 latent factors. Powers the
          movie-detail "Similar Movies" cosine over the learned item factor matrix.
        - Item-Item KNN: cosine similarity between movies (transpose of user-user).
          Pure evaluation baseline.

    All three are trained at startup and persisted to a single pickle file. The frontend
    always requests `algorithm=user_user`; the other two are reachable only via curl
    or the `/metrics` endpoint.
    """

    ALGO_SVD = "svd"
    ALGO_USER_USER = "user_user"
    ALGO_ITEM_ITEM = "item_item"

    def __init__(self):
        # SVD baseline (also used for similar-movies cosine on its qi matrix)
        self.model: Optional[SVD] = None
        # Production model — User-User KNN
        self.user_user_model: Optional[KNNBasic] = None
        # Item-Item KNN baseline
        self.item_item_model: Optional[KNNBasic] = None
        # Surprise Trainset — needed for raw↔inner ID mapping during prediction
        self.trainset = None
        # In-memory copies of the rating and movie tables for fast iteration
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        # movie_id → poster_url, avoids hitting DB per movie when scoring
        self._poster_cache: dict = {}
        # movie_id → blur_hash placeholder string
        self._blur_hash_cache: dict = {}
        # Triggers auto-retrain once it hits RETRAIN_THRESHOLD
        self._new_ratings_count = 0
        # Cross-validated RMSE/MAE from last training run (exposed via /metrics)
        self._last_cv_results: Optional[dict] = None

    def load_data(self):
        """Load ratings and movies from database."""
        db = SessionLocal()
        try:
            ratings = db.query(Rating).all()
            self.ratings_df = pd.DataFrame([
                {"user_id": r.user_id, "movie_id": r.movie_id, "rating": r.rating}
                for r in ratings
            ])

            movies = db.query(Movie).all()
            self.movies_df = pd.DataFrame([
                {"movie_id": m.id, "title": m.title, "genres": m.genres or "unknown",
                 "poster_url": m.poster_url, "blur_hash": m.blur_hash}
                for m in movies
            ])

            # Build poster + blurhash caches for fast lookup
            self._poster_cache = {m.id: m.poster_url for m in movies}
            self._blur_hash_cache = {m.id: m.blur_hash for m in movies}

            logger.info(f"Loaded {len(self.ratings_df)} ratings, {len(self.movies_df)} movies")
        finally:
            db.close()

    def train_model(self, force: bool = False):
        """Train (or reload from cache) all three CF models.

        Order of operations:
          1. Try to load all three from `MODEL_PATH` (pickle). If present + valid → return fast.
          2. Otherwise, build a Surprise trainset from the ratings DataFrame.
          3. Fit SVD (100 factors, 20 epochs SGD).
          4. Fit User-User KNN (K=40, cosine on user vectors).
          5. Fit Item-Item KNN (K=40, cosine on movie vectors).
          6. Persist all three plus the trainset to disk.
          7. Run 3-fold cross-validation on SVD for RMSE/MAE metrics.

        Pass `force=True` to skip the cache and retrain from scratch.
        """
        if not force and self._load_cached_model():
            logger.info("Loaded cached models")
            return

        if self.ratings_df is None or self.ratings_df.empty:
            self.load_data()

        if self.ratings_df is None or self.ratings_df.empty:
            logger.warning("No ratings data available for training")
            return

        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(
            self.ratings_df[["user_id", "movie_id", "rating"]], reader
        )

        # Full trainset = use every known rating; we evaluate via CV separately below.
        self.trainset = data.build_full_trainset()

        # 1. SVD — Netflix-style matrix factorization. Decomposes the user×movie
        #    rating matrix into a user-factor matrix (P) and an item-factor matrix (Q).
        #    Predicted rating = P[u] · Q[i]. Trained with SGD over n_epochs.
        #    Kept as a baseline + powers "Similar Movies" cosine on Q (self.model.qi).
        logger.info("Training SVD model...")
        self.model = SVD(
            n_factors=100,      # dimensionality of latent space
            n_epochs=20,        # SGD passes over the data
            lr_all=0.005,       # learning rate (all parameters)
            reg_all=0.02,       # L2 regularization
            random_state=42,    # reproducibility
        )
        self.model.fit(self.trainset)

        # 2. User-User KNN — PRODUCTION model.
        #    Builds a User × User cosine similarity matrix. For an unseen movie,
        #    predicts rating as the K=40 nearest neighbors' similarity-weighted
        #    average rating on that movie.
        logger.info("Training User-User KNN model...")
        self.user_user_model = KNNBasic(
            k=40,
            sim_options={"name": "cosine", "user_based": True},
            verbose=False,
        )
        self.user_user_model.fit(self.trainset)

        # 3. Item-Item KNN — baseline only.
        #    Same idea but similarity between movies (Movie × Movie matrix). Used
        #    for RMSE/MAE comparison against User-User and SVD.
        logger.info("Training Item-Item KNN model...")
        self.item_item_model = KNNBasic(
            k=40,
            sim_options={"name": "cosine", "user_based": False},
            verbose=False,
        )
        self.item_item_model.fit(self.trainset)

        self._save_model()

        # Run cross-validation for metrics
        try:
            logger.info("Running cross-validation for evaluation metrics...")
            cv_results = cross_validate(
                SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42),
                data, measures=["RMSE", "MAE"], cv=3, verbose=False
            )
            self._last_cv_results = {
                "rmse_mean": round(float(np.mean(cv_results["test_rmse"])), 4),
                "rmse_std": round(float(np.std(cv_results["test_rmse"])), 4),
                "mae_mean": round(float(np.mean(cv_results["test_mae"])), 4),
                "mae_std": round(float(np.std(cv_results["test_mae"])), 4),
            }
            logger.info(f"CV Results — RMSE: {self._last_cv_results['rmse_mean']}, MAE: {self._last_cv_results['mae_mean']}")
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")

        logger.info("All models trained and cached")

    def _get_poster_url(self, movie_id: int) -> Optional[str]:
        """Get poster URL from cache or database."""
        if movie_id in self._poster_cache:
            return self._poster_cache.get(movie_id)
        # Fallback to DB lookup
        if self.movies_df is not None:
            row = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if not row.empty:
                url = row.iloc[0].get("poster_url")
                self._poster_cache[movie_id] = url
                return url
        return None

    def _get_blur_hash(self, movie_id: int) -> Optional[str]:
        """Get BlurHash placeholder from cache or movies_df."""
        if movie_id in self._blur_hash_cache:
            return self._blur_hash_cache.get(movie_id)
        if self.movies_df is not None and "blur_hash" in self.movies_df.columns:
            row = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if not row.empty:
                bh = row.iloc[0].get("blur_hash")
                # pandas stores null cells as float('nan'); normalize to None so
                # the Optional[str] response field doesn't fail pydantic validation.
                if pd.isna(bh):
                    bh = None
                self._blur_hash_cache[movie_id] = bh
                return bh
        return None

    def get_recommendations(
        self,
        user_id: int,
        n: int = DEFAULT_RECOMMENDATION_COUNT,
        algorithm: str = ALGO_SVD,
        min_rating: float = 3.5,
        db: Optional[Session] = None,
    ) -> List[dict]:
        """Return top-N personalized movie recommendations for a user.

        Pipeline:
          1. Select the requested model. Frontend always passes `algorithm='user_user'`;
             defaults to SVD if invoked via curl without a query param.
          2. Cold-start guard: if the user has no ratings in the training set, return
             popular movies (≥20 ratings, sorted by avg rating) and skip ML entirely.
          3. For every movie the user has NOT rated, run `model.predict(user_id, movie_id)`.
          4. Drop predictions below `min_rating` (default 3.5★).
          5. Sort by predicted rating descending.
          6. Apply genre diversification (at most n/3 per genre) so the top-N isn't
             dominated by a single genre.
          7. Attach a human-readable `reason` (varies by algorithm).
        """
        # 1. Pick the model. If models aren't loaded yet, train/reload them first.
        model = self._get_model(algorithm)
        if model is None:
            self.train_model()
            model = self._get_model(algorithm)

        # Always read the user's rated movies FROM THE DATABASE (not the cached
        # ratings_df, which is only refreshed every RETRAIN_THRESHOLD ratings).
        # Otherwise newly-registered users + their onboarding ratings are invisible
        # here and the user sees their own picks recommended back to them.
        rated_movies: set = set()
        user_rated_pairs: list = []  # (movie_id, rating) for fallback strategies
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        try:
            db_ratings = (
                db.query(Rating.movie_id, Rating.rating)
                .filter(Rating.user_id == user_id)
                .all()
            )
            for mid, score in db_ratings:
                rated_movies.add(int(mid))
                user_rated_pairs.append((int(mid), float(score)))
        finally:
            if close_db:
                db.close()
                db = None

        # Pull up to 3 top-rated titles for the "reason" string.
        user_rated_titles: list = []
        if user_rated_pairs and self.movies_df is not None:
            top_three = sorted(user_rated_pairs, key=lambda p: p[1], reverse=True)[:3]
            for mid, _ in top_three:
                row = self.movies_df[self.movies_df["movie_id"] == mid]
                if not row.empty:
                    user_rated_titles.append(row.iloc[0]["title"])

        if model is None:
            # Models still unavailable (e.g., no ratings yet) → fall back to popular.
            return self._get_popular_movies(n, db, exclude=rated_movies)

        # 2. Cold start: user not in trainset → no neighbors → can't run CF.
        #    Try item-based recs from their existing ratings before falling back to
        #    popular. This is what makes a brand-new user's recommendations actually
        #    reflect the 5 movies they picked during onboarding.
        try:
            self.trainset.to_inner_uid(user_id)
        except ValueError:
            item_based = self._item_based_recs_for_unseen_user(
                user_rated_pairs, rated_movies, n
            )
            if item_based:
                return item_based
            return self._get_popular_movies(n, db, exclude=rated_movies)

        if self.movies_df is None:
            self.load_data()

        # 3-5. Score every unseen movie, filter, sort.
        predictions = []
        for _, movie in self.movies_df.iterrows():
            mid = movie["movie_id"]
            if mid in rated_movies:
                continue
            pred = model.predict(user_id, mid)
            if pred.est < min_rating:
                continue
            predictions.append({
                "movie_id": int(mid),
                "title": movie["title"],
                "genres": movie["genres"].split("|"),
                "predicted_rating": round(pred.est, 2),
                "poster_url": self._get_poster_url(int(mid)),
                "blur_hash": self._get_blur_hash(int(mid)),
                "reason": self._build_explanation(algorithm, user_rated_titles, movie),
            })

        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)

        # 6. Genre balance so the top-N isn't all Drama (or whichever genre user rated most).
        diversified = self._diversify_recommendations(predictions, n)
        return diversified[:n]

    def get_similar_movies(
        self,
        movie_id: int,
        n: int = 10,
        db: Optional[Session] = None,
    ) -> List[dict]:
        """Find movies similar to a given movie using cosine similarity on SVD latent factors.

        Powers the "Similar Movies" carousel on the movie-detail page. Independent of
        the per-user recommendation algorithm (this is item-based, not user-based).

        How it works:
          1. Look up the source movie's latent factor vector from SVD's Q matrix
             (`model.qi[inner_id]`) — a 100-dim representation.
          2. Compute cosine similarity vs every other movie's vector.
          3. Sort descending, return top N.
        """
        if self.model is None:
            self.train_model()

        if self.model is None or self.trainset is None:
            return []

        # Surprise uses internal contiguous IDs; raw movie_id must be translated.
        try:
            inner_id = self.trainset.to_inner_iid(movie_id)
        except ValueError:
            return []

        # Source movie title — used in the "85% similar to Star Wars" reason string.
        source_title = ""
        if self.movies_df is not None:
            src_row = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if not src_row.empty:
                source_title = src_row.iloc[0]["title"]

        # qi is the learned item factor matrix (n_items × n_factors).
        item_factors = self.model.qi
        target_factors = item_factors[inner_id]

        # Pairwise cosine: dot product divided by product of L2 norms.
        similarities = []
        for other_inner_id in range(len(item_factors)):
            if other_inner_id == inner_id:
                continue
            other_factors = item_factors[other_inner_id]
            dot = np.dot(target_factors, other_factors)
            norm = np.linalg.norm(target_factors) * np.linalg.norm(other_factors)
            similarity = dot / norm if norm > 0 else 0

            try:
                raw_id = self.trainset.to_raw_iid(other_inner_id)
                similarities.append((int(raw_id), similarity))
            except ValueError:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:n]

        results = []
        if self.movies_df is not None:
            for mid, sim in top_similar:
                movie_row = self.movies_df[self.movies_df["movie_id"] == mid]
                if not movie_row.empty:
                    row = movie_row.iloc[0]
                    pct = round(sim * 100)
                    results.append({
                        "movie_id": mid,
                        "title": row["title"],
                        "genres": row["genres"].split("|"),
                        "predicted_rating": round(max(sim * 5, 1.0), 2),
                        "poster_url": self._get_poster_url(mid),
                        "blur_hash": self._get_blur_hash(mid),
                        "reason": f"{pct}% similar to {source_title}"
                    })

        return results

    def get_evaluation_metrics(self) -> dict:
        """Return model evaluation metrics (RMSE, MAE, precision@k, recall@k)."""
        if self._last_cv_results is None and self.model is not None:
            # Run CV if not already done
            try:
                reader = Reader(rating_scale=(1.0, 5.0))
                data = Dataset.load_from_df(
                    self.ratings_df[["user_id", "movie_id", "rating"]], reader
                )
                cv_results = cross_validate(
                    SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42),
                    data, measures=["RMSE", "MAE"], cv=3, verbose=False
                )
                self._last_cv_results = {
                    "rmse_mean": round(float(np.mean(cv_results["test_rmse"])), 4),
                    "rmse_std": round(float(np.std(cv_results["test_rmse"])), 4),
                    "mae_mean": round(float(np.mean(cv_results["test_mae"])), 4),
                    "mae_std": round(float(np.std(cv_results["test_mae"])), 4),
                }
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        # Compute precision@10 and recall@10 on test set
        precision_at_k, recall_at_k = self._precision_recall_at_k(k=10, threshold=3.5)

        metrics = {
            "production_algorithm": "User-User KNN (Cosine, k=40)",
            "evaluation_baselines": ["SVD (n_factors=100)", "Item-Item KNN (Cosine, k=40)"],
            "rmse_mae_computed_on": "SVD baseline (3-fold CV)",
            "total_ratings": len(self.ratings_df) if self.ratings_df is not None else 0,
            "total_movies": len(self.movies_df) if self.movies_df is not None else 0,
            "total_users": self.ratings_df["user_id"].nunique() if self.ratings_df is not None else 0,
        }

        if self._last_cv_results:
            metrics.update(self._last_cv_results)

        metrics["precision_at_10"] = round(precision_at_k, 4)
        metrics["recall_at_10"] = round(recall_at_k, 4)

        return metrics

    def _precision_recall_at_k(self, k: int = 10, threshold: float = 3.5) -> Tuple[float, float]:
        """Compute precision@k and recall@k for top-N recommendation evaluation.

        precision@k = (relevant items in top-k) / k          — how many of our recs are good
        recall@k    = (relevant items in top-k) / (all relevant) — how many goods we found

        "Relevant" = user actually rated the movie ≥ `threshold` (3.5★).
        Computes on an anti-testset (movies the user has NOT rated) using SVD predictions.
        """
        if self.ratings_df is None or self.ratings_df.empty:
            return 0.0, 0.0

        try:
            reader = Reader(rating_scale=(1.0, 5.0))
            data = Dataset.load_from_df(
                self.ratings_df[["user_id", "movie_id", "rating"]], reader
            )
            trainset = data.build_full_trainset()
            testset = trainset.build_anti_testset()

            model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
            model.fit(trainset)
            predictions = model.test(testset)

            # Group predictions by user
            user_est = {}
            for pred in predictions:
                uid = pred.uid
                if uid not in user_est:
                    user_est[uid] = []
                user_est[uid].append((pred.est, pred.iid))

            # Relevant items per user (items rated >= threshold in actual data)
            user_relevant = {}
            for _, row in self.ratings_df.iterrows():
                uid = row["user_id"]
                if row["rating"] >= threshold:
                    if uid not in user_relevant:
                        user_relevant[uid] = set()
                    user_relevant[uid].add(row["movie_id"])

            precisions = []
            recalls = []
            for uid, est_list in user_est.items():
                est_list.sort(key=lambda x: x[0], reverse=True)
                top_k = [iid for _, iid in est_list[:k]]
                relevant = user_relevant.get(uid, set())

                if not relevant:
                    continue

                n_relevant_in_k = sum(1 for iid in top_k if iid in relevant)
                precisions.append(n_relevant_in_k / k)
                recalls.append(n_relevant_in_k / len(relevant))

            return (
                float(np.mean(precisions)) if precisions else 0.0,
                float(np.mean(recalls)) if recalls else 0.0
            )
        except Exception as e:
            logger.warning(f"Precision/recall computation failed: {e}")
            return 0.0, 0.0

    def _get_model(self, algorithm: str):
        """Map algorithm key → trained model instance. Falls back to SVD for unknown keys."""
        if algorithm == self.ALGO_USER_USER:
            return self.user_user_model
        elif algorithm == self.ALGO_ITEM_ITEM:
            return self.item_item_model
        return self.model  # default: SVD

    def _build_explanation(self, algorithm: str, user_top_titles: List[str], movie_row) -> str:
        """Build the `reason` string shown under each recommendation card.

        Tailored per algorithm so the displayed justification matches the actual model:
          - user_user: name a movie the user loved + frame it as taste-neighbor signal
          - item_item: frame as movie-to-movie similarity from rating patterns
          - svd:       frame as latent-factor match
        """
        genres = movie_row["genres"].split("|") if isinstance(movie_row["genres"], str) else []
        genre_str = ", ".join(genres[:2]) if genres else "various genres"

        if algorithm == self.ALGO_USER_USER:
            if user_top_titles:
                return f"Users who liked {user_top_titles[0]} also rated this highly"
            return "Recommended by users with similar taste"
        elif algorithm == self.ALGO_ITEM_ITEM:
            if user_top_titles:
                return f"Similar to {user_top_titles[0]} based on rating patterns"
            return f"Matches your preference for {genre_str}"
        else:
            # SVD
            if user_top_titles:
                return f"Predicted highly for you based on your love of {user_top_titles[0]}"
            return f"Highly rated in {genre_str} — a strong match for your profile"

    def record_new_rating(self):
        """Increment the new-rating counter; auto-retrain when the threshold is hit.

        Called by the ratings router after every successful rating submission.
        Once `RETRAIN_THRESHOLD` (default 100) ratings accumulate, all three models
        are rebuilt from scratch including the newly submitted data, then the counter
        resets to zero.
        """
        self._new_ratings_count += 1
        from app.config import RETRAIN_THRESHOLD
        if self._new_ratings_count >= RETRAIN_THRESHOLD:
            logger.info(f"Retraining model after {self._new_ratings_count} new ratings")
            self.load_data()
            self.train_model(force=True)
            self._new_ratings_count = 0

    def _get_popular_movies(
        self,
        n: int,
        db: Optional[Session] = None,
        exclude: Optional[set] = None,
    ) -> List[dict]:
        """Cold-start fallback: most-popular movies with a minimum rating count.

        Used when a user has no ratings yet (so CF has no signal). Movies are
        required to have ≥20 ratings to filter out obscure entries, then sorted by
        average rating descending.

        `exclude` (set of movie_ids) is skipped so a new user who already picked
        5 movies during onboarding doesn't see them recommended back.
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        exclude = exclude or set()

        try:
            from sqlalchemy import func
            # Pull extra so we still have ≥n after excluding the user's picks.
            limit = n + max(len(exclude), 20)
            popular = (
                db.query(
                    Rating.movie_id,
                    func.avg(Rating.rating).label("avg_rating"),
                    func.count(Rating.id).label("count")
                )
                .group_by(Rating.movie_id)
                .having(func.count(Rating.id) >= 20)
                .order_by(func.avg(Rating.rating).desc())
                .limit(limit)
                .all()
            )

            results = []
            for movie_id, avg_rating, count in popular:
                if int(movie_id) in exclude:
                    continue
                movie = db.query(Movie).filter(Movie.id == movie_id).first()
                if movie:
                    results.append({
                        "movie_id": movie.id,
                        "title": movie.title,
                        "genres": (movie.genres or "unknown").split("|"),
                        "predicted_rating": round(float(avg_rating), 2),
                        "poster_url": movie.poster_url,
                        "blur_hash": movie.blur_hash,
                        "reason": f"Trending — loved by {count} users"
                    })
                if len(results) >= n:
                    break

            return results
        finally:
            if close_db:
                db.close()

    def _item_based_recs_for_unseen_user(
        self,
        user_rated_pairs: List[Tuple[int, float]],
        rated_movies: set,
        n: int,
    ) -> List[dict]:
        """Recommendations for a user not yet in the trainset.

        Strategy: take every movie the user rated ≥3.5★, find each one's nearest
        neighbours in SVD's learned item-factor matrix (cosine on `model.qi`),
        sum the similarities weighted by the user's rating, then return the top-N
        unrated movies.

        This is the only path that lets a freshly-registered user get personalised
        recs immediately — without it they fall straight to popular movies.
        """
        if (self.model is None or self.trainset is None
                or self.movies_df is None or not user_rated_pairs):
            return []

        liked = [(mid, score) for mid, score in user_rated_pairs if score >= 3.5]
        if not liked:
            return []

        item_factors = self.model.qi  # (n_items, n_factors)
        norms = np.linalg.norm(item_factors, axis=1)
        norms = np.where(norms == 0, 1e-9, norms)

        scores: dict = {}
        for source_mid, source_score in liked:
            try:
                src_inner = self.trainset.to_inner_iid(source_mid)
            except ValueError:
                continue
            src_vec = item_factors[src_inner]
            src_norm = norms[src_inner]
            # cosine sim of source vs every other movie
            sims = (item_factors @ src_vec) / (norms * src_norm)
            weight = source_score / 5.0
            for inner_id, sim in enumerate(sims):
                if inner_id == src_inner:
                    continue
                try:
                    raw_id = int(self.trainset.to_raw_iid(inner_id))
                except ValueError:
                    continue
                if raw_id in rated_movies:
                    continue
                scores[raw_id] = scores.get(raw_id, 0.0) + float(sim) * weight

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        # Title of the user's top-rated movie powers the "reason" string.
        top_liked = max(user_rated_pairs, key=lambda p: p[1])
        top_title = ""
        row = self.movies_df[self.movies_df["movie_id"] == top_liked[0]]
        if not row.empty:
            top_title = row.iloc[0]["title"]

        results: List[dict] = []
        for mid, score in ranked:
            row = self.movies_df[self.movies_df["movie_id"] == mid]
            if row.empty:
                continue
            r = row.iloc[0]
            results.append({
                "movie_id": int(mid),
                "title": r["title"],
                "genres": (r["genres"] or "unknown").split("|"),
                "predicted_rating": round(min(5.0, max(1.0, 2.5 + score)), 2),
                "poster_url": self._get_poster_url(int(mid)),
                "blur_hash": self._get_blur_hash(int(mid)),
                "reason": (
                    f"Because you liked {top_title}" if top_title
                    else "Based on the movies you picked"
                ),
            })
            if len(results) >= n * 2:
                break

        return self._diversify_recommendations(results, n)[:n]

    def _diversify_recommendations(self, predictions: List[dict], n: int) -> List[dict]:
        """Cap the number of times any single genre can appear in the top-N.

        Without this, a user who heavily rates Drama gets a top-10 of all-Drama.
        Rule: each genre is allowed at most `max(2, n // 3)` slots. We walk through
        the sorted predictions and skip any candidate whose genre already hit the cap,
        except near the end of the list where we relax the rule to ensure we return
        a full N items.
        """
        if len(predictions) <= n:
            return predictions

        selected = []
        genre_counts = {}
        max_per_genre = max(2, n // 3)

        for pred in predictions:
            if len(selected) >= n:
                break

            genres = pred["genres"]
            over_represented = any(
                genre_counts.get(g, 0) >= max_per_genre for g in genres
            )

            # Relax the genre cap when we're within 2 of the target N so we don't
            # under-fill the recommendation list.
            if not over_represented or len(selected) >= n - 2:
                selected.append(pred)
                for g in genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

        return selected

    def _save_model(self):
        """Persist all trained models + the trainset to a single pickle file."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": self.model,
                "user_user_model": self.user_user_model,
                "item_item_model": self.item_item_model,
                "trainset": self.trainset,
            }, f)
        logger.info(f"Models saved to {MODEL_PATH}")

    def _load_cached_model(self) -> bool:
        """Restore models from disk cache. Returns True on success, False if cache is missing
        or corrupted (caller then triggers a full retrain)."""
        if not os.path.exists(MODEL_PATH):
            return False
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model = data.get("model")
            self.user_user_model = data.get("user_user_model")
            self.item_item_model = data.get("item_item_model")
            self.trainset = data.get("trainset")
            return self.model is not None
        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}")
            return False


# Module-level singleton: every router imports this exact instance. State
# (trained models, cached DataFrames, retrain counter) lives for the process lifetime.
recommender = RecommenderEngine()
