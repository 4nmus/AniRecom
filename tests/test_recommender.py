import pandas as pd

from core import recommend_anime


def test_hybrid_ranking_considers_all_candidates():
    rows = [
        {
            "anime_id": 0,
            "title": "Liked",
            "liked": 1,
            "score": 0,
            "genre__A": 1,
            "genre__B": 1,
            "genre__C": 0,
        }
    ]

    for anime_id in range(1, 16):
        rows.append({
            "anime_id": anime_id,
            "title": f"Candidate {anime_id}",
            "liked": 0,
            "score": 0,
            "genre__A": 1,
            "genre__B": 1,
            "genre__C": 0,
        })

    rows.append({
        "anime_id": 99,
        "title": "Best Hybrid Match",
        "liked": 0,
        "score": 10,
        "genre__A": 1,
        "genre__B": 1,
        "genre__C": 1,
    })

    df = pd.DataFrame(rows)
    genres = ["genre__A", "genre__B", "genre__C"]

    result = recommend_anime(df, genres, n_recommendations=1)

    assert result.iloc[0]["anime_id"] == 99
    assert result.iloc[0]["recommendation_score"] > 0.8