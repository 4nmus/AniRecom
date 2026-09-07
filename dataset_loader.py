import sys
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MultiLabelBinarizer

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


def load_available_dataset() -> pd.DataFrame:
    try:
        path = os.getcwd()
        path_to_files = f'{path}/datasets'
        files = os.scandir(path_to_files)
        for file in files:
            if file.is_file():
                df = pd.read_csv(f'{path}/datasets/{file.name}')
                logging.debug(f'{path}/datasets/{file.name}')
                break
        return df

    except Exception:
        logging.error(f"No available csv found. It shall be in {os.getcwd()}/datasets")


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


if __name__ == "__main__":
    x = load_available_dataset()
    print(x.head())