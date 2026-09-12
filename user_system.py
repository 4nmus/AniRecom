from time import sleep

import pandas as pd
import os



def select_liked(df: pd.DataFrame):
    # Custom selection since each user is individual. Will be changed soon
    likely_seen = df[df['score'] >= 9]
    for idx, row in likely_seen.head(100).iterrows():
        print("DId you watch ? ")
        print(row['english_title'])
        ans = input("Liked it? (yes/no/exit): ").strip().lower()
        if ans == "yes":
            df.loc[idx, 'liked'] = 0
        elif ans == "no":
            df.loc[idx, 'liked'] = 1
        elif ans == "exit":
            break
    return df
