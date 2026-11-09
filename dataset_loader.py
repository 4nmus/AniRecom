from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


REQUIRED_COLUMNS = {"anime_id", "title", "score", "genres"}


def load_available_dataset(dataset_dir=None, dataset_path=None):
    if dataset_path is None:
        directory = (
            Path(dataset_dir)
            if dataset_dir is not None
            else Path(__file__).resolve().parent / "datasets"
        )
        csv_files = sorted(directory.glob("*.csv"))

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected exactly one CSV in {directory}; "
                f"found {len(csv_files)}. Specify dataset_path."
            )
        dataset_path = csv_files[0]

    df = pd.read_csv(dataset_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["anime_id"].isna().any() or df["anime_id"].duplicated().any():
        raise ValueError("anime_id must be unique and non-null.")

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def separate_combined_feature(df, column="genres", split_sign=";"):
    if column not in df.columns:
        raise ValueError(f"Missing genre column: {column}")

    genres = df[column].fillna("").astype(str).apply(
        lambda value: sorted({
            genre.strip()
            for genre in value.split(split_sign)
            if genre.strip()
        })
    )

    encoder = MultiLabelBinarizer()
    encoded = encoder.fit_transform(genres)

    genre_df = pd.DataFrame(
        encoded,
        columns=[f"genre__{name}" for name in encoder.classes_],
        index=df.index,
    )

    if set(genre_df.columns) & set(df.columns):
        raise ValueError("Encoded genre columns already exist.")

    return pd.concat([df, genre_df], axis=1)