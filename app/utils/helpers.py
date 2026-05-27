from typing import List


def parse_genres(genres_str: str) -> List[str]:
    """Parse pipe-separated genre string."""
    if not genres_str:
        return ["unknown"]
    return [g.strip() for g in genres_str.split("|") if g.strip()]


ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]
