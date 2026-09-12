from dataset_loader import load_available_dataset, separate_combined_feature, append_user_columns
from user_system import select_liked
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def recommend_anime(df, genre_columns, n_recommendations=10):

    #Recommend anime based only on genre similarity.

    liked_df = df[df["liked"] == 1]

    if liked_df.empty:
        print("No liked anime yet.")
        return df.head(0)

    X = df[genre_columns].fillna(0).values

    # Average genre preferences
    user_profile = liked_df[genre_columns].fillna(0).mean(axis=0).values.reshape(1, -1)



