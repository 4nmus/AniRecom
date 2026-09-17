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




if __name__ == "__main__":
    unittest.main()
