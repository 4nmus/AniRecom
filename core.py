from dataset_loader import load_available_dataset, separate_combined_feature, append_user_columns
from user_system import select_liked
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


def recommend_anime(df, genre_columns, n_recommendations=10):

    #Recommend anime based only on genre similarity.

    liked_df = df[df["liked"] == 1]

    if liked_df.empty:
        print("No liked anime yet.")
        return df.head(0)

    X = df[genre_columns].fillna(0).values

    # Average genre preferences
    user_profile = liked_df[genre_columns].fillna(0).mean(axis=0).values.reshape(1, -1)

    # KNN
    knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    knn.fit(X)

    # extra because some results may already be liked
    n_neighbors = min(len(df), n_recommendations + len(liked_df) + 10)

    distances, indices = knn.kneighbors(user_profile, n_neighbors=n_neighbors)

    recommendations = df.iloc[indices[0]].copy()

    # Convert to similarity:
    recommendations["similarity"] = 1 - distances[0]

    # Remove already liked
    recommendations = recommendations[
        ~recommendations["anime_id"].isin(liked_df["anime_id"])
    ]

    return recommendations[
        ["anime_id", "title", "score", "similarity"]
    ].head(n_recommendations)

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

    genre_columns = [column for column in df.columns if column not in original_columns]

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


