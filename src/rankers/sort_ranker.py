import numpy as np
import pandas as pd
from tqdm import tqdm
from rankers.base_ranker import BaseRanker


class SortRanker(BaseRanker):
    def __init__(self, items_to_rank: int, params: dict):
        self.items_to_rank = items_to_rank
        self.rng = np.random.default_rng(seed=42)

    def rank(
        self,
        proba: pd.DataFrame,
        paper_embeddings: dict
    ):
        print(f"Ranking with sort ranker\nitems_to_rank: {self.items_to_rank}")
        ranked = {}
        for author, row in tqdm(proba.iterrows(), "Ranking", len(proba)):
            ranked[author] = list(row.sort_values(ascending=False, kind="stable").keys())[:self.items_to_rank]
        return ranked
        
