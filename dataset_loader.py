import sys

import pandas as pd
import numpy as np
import os
import logging

#Loggers
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

#Daa
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

    except:
        logging.error(f"No available csv found. It shall be in {os.getcwd()}/datasets")




if __name__ == "__main__":
    x = load_available_dataset()
    print(x.head())