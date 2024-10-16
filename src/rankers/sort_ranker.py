import numpy as np
import pandas as pd
from rankers.base_ranker import BaseRanker


class SortRanker(BaseRanker):
    def __init__(self, params: dict):
        self.rng = np.random.default_rng(seed=42)

    def rank(
        self,
        proba: pd.DataFrame,
        paper_embeddings: dict
    ):
        ranked = {}
        for author, row in proba.iterrows():
            ranked[author] = list(row.sort_values(ascending=False).keys())
        return ranked
        
