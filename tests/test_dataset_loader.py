import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dataset_loader import load_available_dataset


class TestDatasetLoader(unittest.TestCase):

    def test_loads_csv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            expected = pd.DataFrame({
                "anime_id": [1, 2],
                "title": ["Naruto", "Bleach"],
            })

            expected.to_csv(path / "anime.csv", index=False)

            result = load_available_dataset(path)

            pd.testing.assert_frame_equal(result, expected)

    def test_raises_when_no_csv_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_available_dataset(tmpdir)


if __name__ == "__main__":
    unittest.main()