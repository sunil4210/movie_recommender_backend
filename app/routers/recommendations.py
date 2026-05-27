from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import RecommendationResponse
from app.services.recommender import recommender, RecommenderEngine

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.get("/metrics")
def get_metrics():
    """Get model evaluation metrics (RMSE, MAE, Precision@10, Recall@10)."""
    return recommender.get_evaluation_metrics()


@router.get("/{user_id}/similar/{movie_id}", response_model=list[RecommendationResponse])
def get_similar_movies(
    user_id: int,
    movie_id: int,
    n: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get movies similar to a given movie (item-item cosine similarity)."""
    results = recommender.get_similar_movies(movie_id=movie_id, n=n, db=db)
    return [RecommendationResponse(**r) for r in results]


@router.get("/{user_id}", response_model=list[RecommendationResponse])
def get_recommendations(
    user_id: int,
    n: int = Query(10, ge=1, le=50),
    algorithm: str = Query(
        "svd",
        description="Algorithm: 'svd' (default), 'user_user' (User-User KNN), 'item_item' (Item-Item KNN)"
    ),
    db: Session = Depends(get_db),
):
    """Get personalized movie recommendations for a user.

    Supports multiple collaborative filtering algorithms:
    - **svd**: Singular Value Decomposition (latent factor model)
    - **user_user**: User-User KNN with cosine similarity
    - **item_item**: Item-Item KNN with cosine similarity
    """
    if algorithm not in (RecommenderEngine.ALGO_SVD, RecommenderEngine.ALGO_USER_USER, RecommenderEngine.ALGO_ITEM_ITEM):
        algorithm = RecommenderEngine.ALGO_SVD

    results = recommender.get_recommendations(user_id=user_id, n=n, algorithm=algorithm, db=db)
    return [RecommendationResponse(**r) for r in results]


@router.post("/refresh")
def refresh_model():
    """Retrain all recommendation models with latest data."""
    recommender.load_data()
    recommender.train_model(force=True)
    return {"message": "All models retrained successfully (SVD, User-User KNN, Item-Item KNN)"}
