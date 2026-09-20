import sys
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

# Loggers
file_handler = logging.FileHandler("logs.log")
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

logging.basicConfig(format='[%(asctime)s - %(name)s: %(levelname)s - %(message)s]',
    level=logging.DEBUG,
    handlers=[
        file_handler, stream_handler
    ],
)

# Dataset functions

def load_available_dataset(dataset_dir=None) -> pd.DataFrame:
    if dataset_dir is None:
        dataset_dir = Path(__file__).resolve().parent / "datasets"
    else:
        dataset_dir = Path(dataset_dir)

    csv_files = sorted(dataset_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in {dataset_dir}"
        )

    dataset_path = csv_files[0]
    logging.info("Loading dataset from %s", dataset_path)

    return pd.read_csv(dataset_path)


def separate_combined_feature(df: pd.DataFrame, column:str, split_sign: str) -> pd.DataFrame:
    # Used to separate combines features. Example df['genres'] = Fantasy; Drama -> df['fantasy'] = 1 , df['drama'] = 1
    separaed_items = df[column].fillna('').str.split(f'{split_sign}')
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(separaed_items)
    df_genre = pd.DataFrame(
        encoded,
        columns=mlb.classes_,
        index=df.index
    )

    df = pd.concat([df, df_genre], axis=1)
    return df

def append_user_columns(df: pd.DataFrame) -> pd.DataFrame:

    df['liked'] = 0

    return df


if __name__ == "__main__":
    x = load_available_dataset()
    print(x.head())