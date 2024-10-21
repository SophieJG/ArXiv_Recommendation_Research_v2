import numpy as np
import pandas as pd
from tqdm import tqdm
from rankers.base_ranker import BaseRanker


class DiversityRanker(BaseRanker):
    """
Implementation of the algorithm proposed in Sec 7.2 of the paper "Reconciling the accuracy-diversity trade-off in recommendations". Ask
Nikhil for the full version
"""
    def __init__(self, items_to_rank, params: dict):
        self.items_to_rank = items_to_rank
        self.rng = np.random.default_rng(seed=42)
        self.lambd = params["lambda"]

    def rank(
        self,
        proba: pd.DataFrame,
        paper_embeddings: dict
    ):
        print(f"Ranking with diversity ranker\nitems_to_rank: {self.items_to_rank}\nlambda: {self.lambd}")
        ranked = {}
        for author, row in tqdm(proba.iterrows(), "Ranking", len(proba)):
            author_reccomendation = []
            author_embeddings = None  # The embeddings of papers in the set
            for _ in range(self.items_to_rank):
                best_score_w_diversity = -1
                best_item = None
                for key, score in zip(row.keys(), row.values):
                    if key in author_reccomendation:
                        continue
                    if author_embeddings is None:
                        score_w_diversity = score
                    else:
                        paper_embedding = paper_embeddings[key]
                        diversity_score = 1. - np.mean(np.dot(author_embeddings, paper_embedding))
                        score_w_diversity = (1 - self.lambd) * score + self.lambd * diversity_score
                    if score_w_diversity > best_score_w_diversity:
                        best_item = key
                        best_score_w_diversity = score_w_diversity
                author_reccomendation.append(best_item)
                if author_embeddings is None:
                    author_embeddings = paper_embeddings[best_item]
                else:
                    author_embeddings = np.vstack([author_embeddings, paper_embeddings[best_item]])
            ranked[author] = author_reccomendation
                
        return ranked        
