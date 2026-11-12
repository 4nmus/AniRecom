import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core import recommend_anime


def metrics_at_k(recommended_ids, relevant_ids, k):
    recommended = list(recommended_ids)[:k]
    relevant = set(relevant_ids)

    hits = sum(anime_id in relevant for anime_id in recommended)

    dcg = sum(
        1 / np.log2(position + 2)
        for position, anime_id in enumerate(recommended)
        if anime_id in relevant
    )

    ideal_dcg = sum(
        1 / np.log2(position + 2)
        for position in range(min(k, len(relevant)))
    )

    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant) if relevant else 0.0,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def evaluate_models(df, genre_columns, k=10, random_state=42):
    if k < 1:
        raise ValueError("k must be positive.")

    liked_ids = df.loc[df["liked"].eq(1), "anime_id"].unique()

    if len(liked_ids) < 2:
        raise ValueError("Evaluation needs at least two liked anime.")

    train_ids, test_ids = train_test_split(
        liked_ids,
        test_size=0.2,
        random_state=random_state,
    )

    train_df = df.copy()
    train_df["liked"] = (
        train_df["anime_id"].isin(train_ids).astype(int)
    )

    if "preference" in train_df.columns:
        train_df.loc[
            train_df["anime_id"].isin(test_ids),
            "preference",
        ] = "unrated"

    hybrid = recommend_anime(
        train_df, genre_columns, n_recommendations=k
    )

    eligible = train_df.loc[
        train_df["liked"].eq(0)
    ].copy()

    if "preference" in eligible.columns:
        eligible = eligible.loc[
            eligible["preference"].ne("disliked")
        ]

    eligible["_score"] = (
        pd.to_numeric(eligible["score"], errors="coerce")
        .fillna(0)
    )

    baseline = eligible.sort_values(
        ["_score", "anime_id"],
        ascending=[False, True],
        kind="stable",
    ).head(k)

    return {
        "hybrid": metrics_at_k(
            hybrid["anime_id"], test_ids, k
        ),
        "top_rated": metrics_at_k(
            baseline["anime_id"], test_ids, k
        ),
    }