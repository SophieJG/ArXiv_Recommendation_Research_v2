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
        self.lambda_ = params["lambda"]  # Cannot use lambda as the variable name

    def rank(
        self,
        utility: pd.DataFrame,
        paper_embeddings: dict
    ):
        print(f"Ranking with diversity ranker\nitems_to_rank: {self.items_to_rank}\nlambda: {self.lambda_}")
        ranked = {}
        # Iterate over all authors and build the reccomendations for each author separately
        for author, scored_papers in tqdm(utility.iterrows(), "Ranking", len(utility)):
            author_reccomendation = []  # The ordered set of papers recommended for the author
            author_embeddings = None  # The embeddings of papers in the set
            # Use a greedy approach and add the papers to the set one by one
            for _ in range(self.items_to_rank):
                best_score_w_diversity = -1
                best_item = None
                # Go over all papers not in the recommended set and calculate the score for each one
                # The score is the lambda-weighted sum of the paper-author utility and the diversity of the papers already in the set
                # and the candidate
                for key, score in zip(scored_papers.keys(), scored_papers.values):
                    if key in author_reccomendation:
                        continue
                    if author_embeddings is None:
                        score_w_diversity = score
                    else:
                        paper_embedding = paper_embeddings[key]
                        # Because the diversity of the selected set is already fixed, we are interested only in the diversity between the set and the
                        # candidate paper
                        diversity_score = 1. - np.mean(np.dot(author_embeddings, paper_embedding))
                        score_w_diversity = (1 - self.lambda_) * score + self.lambda_ * diversity_score # The lambda combined score
                    if score_w_diversity > best_score_w_diversity:
                        best_item = key
                        best_score_w_diversity = score_w_diversity
                author_reccomendation.append(best_item)
                # Add the selected paper to the list of selected paper embeddings
                if author_embeddings is None:
                    author_embeddings = paper_embeddings[best_item]
                else:
                    author_embeddings = np.vstack([author_embeddings, paper_embeddings[best_item]])
            ranked[author] = author_reccomendation
                
        return ranked        
