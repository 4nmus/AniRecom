from dataset_loader import load_available_dataset, separate_combined_feature, append_user_columns
from user_system import select_liked
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


RECOMMENDATION_COLUMNS = [
    "anime_id",
    "title",
    "score",
    "similarity",
    "recommendation_score",
]


def recommend_anime(df, genre_columns, n_recommendations=10):
    if n_recommendations < 1:
        raise ValueError("n_recommendations must be positive.")

    if not genre_columns:
        raise ValueError("At least one genre feature is required.")

    liked = df["liked"].eq(1)

    if not liked.any():
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    features = df[genre_columns].fillna(0).to_numpy(
        dtype=np.float64
    )

    profile = features[liked.to_numpy()].mean(
        axis=0, keepdims=True
    )

    similarities = cosine_similarity(
        features, profile
    ).ravel()

    normalized_scores = (
        pd.to_numeric(df["score"], errors="coerce")
        .fillna(0)
        .clip(0, 10)
        / 10.0
    )

    ranked = df[["anime_id", "title", "score"]].copy()
    ranked["similarity"] = similarities
    ranked["recommendation_score"] = (
        0.8 * similarities
        + 0.2 * normalized_scores
    )

    liked_ids = df.loc[liked, "anime_id"]
    ranked = ranked.loc[
        ~ranked["anime_id"].isin(liked_ids)
    ]

    ranked = ranked.sort_values(
        "recommendation_score",
        ascending=False,
        kind="stable",
    )

    return ranked[RECOMMENDATION_COLUMNS].head(
        n_recommendations
    )

def evaluate_recommender(df, genre_columns, n_recommendations=10, test_size=0.2, random_state=42):
    if n_recommendations < 1:
        raise ValueError("n_recommendations must be positive.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    liked_ids = df.loc[df["liked"] == 1, "anime_id"].unique()
    if len(liked_ids) < 2:
        raise ValueError("Evaluation needs at least two liked anime.")

    train_ids, test_ids = train_test_split(
        liked_ids, test_size=test_size, random_state=random_state
    )
    train_df = df.copy()
    train_df["liked"] = train_df["anime_id"].isin(train_ids).astype(int)
    recommendations = recommend_anime(train_df, genre_columns, n_recommendations)
    hits = len(set(recommendations["anime_id"]) & set(test_ids))

    return {
        "train_likes": len(train_ids),
        "test_likes": len(test_ids),
        "precision_at_k": hits / n_recommendations,
        "recall_at_k": hits / len(test_ids),
    }


if __name__ == "__main__":
    df = load_available_dataset()
    original_columns = set(df.columns)
    #separate 0/1 columns
    df = separate_combined_feature(df, "genres", ";")

    genre_columns = [
        column for column in df.columns
        if column.startswith("genre__")
    ]

    df = append_user_columns(df)

    print("Genres:")
    print(genre_columns)

    # Ask the user which anime they liked
    df = select_liked(df)

    if df.loc[df["liked"] == 1, "anime_id"].nunique() >= 2:
        metrics = evaluate_recommender(df, genre_columns)
        print(f"\nEvaluation ({metrics['train_likes']} train / {metrics['test_likes']} test likes):")
        print(f"Precision@10: {metrics['precision_at_k']:.3f}")
        print(f"Recall@10: {metrics['recall_at_k']:.3f}")
    else:
        print("\nEvaluation skipped: select at least two liked anime.")

    recommendations = recommend_anime(df, genre_columns, n_recommendations=10)
    print("\nRecommendations:")
    print(recommendations.to_string(index=False))
    df.to_csv("custom_df.csv", index=False)


