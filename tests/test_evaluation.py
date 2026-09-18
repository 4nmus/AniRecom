import unittest
from unittest.mock import patch

import pandas as pd

from core import evaluate_recommender, recommend_anime
from user_system import select_liked


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "anime_id": [1, 2, 3, 4, 5],
            "title": ["A", "B", "C", "D", "E"],
            "score": [9.0] * 5,
            "Action": [1, 1, 1, 1, 0],
            "Drama": [0, 0, 0, 0, 1],
            "liked": [1, 1, 1, 1, 0],
        })
        self.genres = ["Action", "Drama"]

    def test_holdout_is_hidden_and_original_likes_are_preserved(self):
        original = self.df.copy(deep=True)
        with patch("core.recommend_anime", wraps=recommend_anime) as recommend:
            metrics = evaluate_recommender(self.df, self.genres, n_recommendations=3)

        train_df = recommend.call_args.args[0]
        self.assertEqual(train_df["anime_id"].tolist(), self.df["anime_id"].tolist())
        self.assertEqual(train_df["liked"].sum(), 3)
        recommendations = recommend_anime(train_df, self.genres, 3)
        train_ids = set(train_df.loc[train_df["liked"] == 1, "anime_id"])
        self.assertTrue(train_ids.isdisjoint(recommendations["anime_id"]))
        self.assertEqual(metrics, {
            "train_likes": 3,
            "test_likes": 1,
            "precision_at_k": 1 / 3,
            "recall_at_k": 1.0,
        })
        pd.testing.assert_frame_equal(self.df, original)

    def test_default_split_is_repeatable(self):
        with patch("core.recommend_anime", wraps=recommend_anime) as recommend:
            evaluate_recommender(self.df, self.genres)
            evaluate_recommender(self.df, self.genres)
        pd.testing.assert_frame_equal(
            recommend.call_args_list[0].args[0], recommend.call_args_list[1].args[0]
        )

    def test_insufficient_history_and_invalid_options(self):
        for likes in ([0] * 5, [1, 0, 0, 0, 0]):
            with self.subTest(likes=likes), self.assertRaises(ValueError):
                evaluate_recommender(self.df.assign(liked=likes), self.genres)
        for options in ({"n_recommendations": 0}, {"test_size": 0}, {"test_size": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_recommender(self.df, self.genres, **options)

    def test_feedback_records_yes_as_liked(self):
        df = self.df.assign(english_title=self.df["title"], liked=0)
        with patch("builtins.input", side_effect=["yes", "no", "exit"]), patch("builtins.print"):
            result = select_liked(df)
        self.assertEqual(result["liked"].tolist(), [1, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
