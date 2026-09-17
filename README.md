AniRecom is an adaptive anime recommendation system that learns from feedback loops. Each click from the user influences the end results of recommendations. It learns on previous seen animes, likes, rating etc.

Run `python core.py` and select at least two liked anime to see an evaluation before your recommendations. Evaluation holds out 20% of liked titles using a fixed random seed. Only the remaining likes build the genre profile.

Precision@10 is the number of held-out likes retrieved divided by 10 (even if fewer recommendations are available). Recall@10 is the fraction of held-out likes retrieved.

Run the checks `python -m unittest discover -s tests`.
