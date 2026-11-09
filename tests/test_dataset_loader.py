import pandas as pd

from dataset_loader import separate_combined_feature


def test_genres_are_normalized():
    df = pd.DataFrame({
        "genres": ["Action; Drama", "Action;Drama", None]
    })

    result = separate_combined_feature(df)

    assert "genre__Action" in result
    assert "genre__Drama" in result
    assert "genre__ Drama" not in result
    assert "genre__" not in result
    assert result["genre__Drama"].tolist() == [1, 1, 0]