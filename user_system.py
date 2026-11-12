from time import sleep

import pandas as pd
import os



def select_liked(df):
    df = df.copy()

    while True:
        query = input(
            "\nSearch anime title (or 'done'): "
        ).strip()

        if query.lower() in {"done", "exit"}:
            break

        if not query:
            continue

        title_match = (
            df["title"]
            .fillna("")
            .str.contains(query, case=False, regex=False)
        )

        if "english_title" in df.columns:
            title_match |= (
                df["english_title"]
                .fillna("")
                .str.contains(query, case=False, regex=False)
            )

        matches = df.loc[
            title_match, ["anime_id", "title"]
        ].head(10)

        if matches.empty:
            print("No matching anime.")
            continue

        print(matches.to_string(index=False))

        anime_id = input("Choose anime ID: ").strip()
        selected = matches["anime_id"].astype(str).eq(anime_id)

        if not selected.any():
            print("Invalid selection.")
            continue

        answer = input(
            "Preference (yes/no/skip): "
        ).strip().lower()

        actions = {
            "yes": ("liked", 1),
            "no": ("disliked", 0),
            "skip": ("skipped", 0),
        }

        if answer not in actions:
            print("Invalid preference.")
            continue

        preference, liked = actions[answer]
        row_mask = df["anime_id"].astype(str).eq(anime_id)

        df.loc[row_mask, "preference"] = preference
        df.loc[row_mask, "liked"] = liked

    return df
