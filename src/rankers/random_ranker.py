import numpy as np
import pandas as pd
from rankers.base_ranker import BaseRanker


class RandomRanker(BaseRanker):
    """
Rank the papers randomly
"""
    def __init__(self, items_to_rank: int, params: dict):
        self.rng = np.random.default_rng(seed=42)
        self.items_to_rank = items_to_rank

    def rank(
        self,
        utility: pd.DataFrame,
        paper_embeddings: dict,
        disable_tqdm: bool = False
    ):
        ranked = {}
        for author, row in utility.iterrows():
            author_papers = list(row.keys())
            self.rng.shuffle(author_papers)
            ranked[author] = author_papers[:self.items_to_rank]
        return ranked
        
