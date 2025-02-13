from models.base_model import BaseModel
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import joblib


class CocitationLogistic(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model : LogisticRegression = None

    @staticmethod
    def _process_samples(samples: list) -> list:
        cocitation_counts = []
        # TODO: Write more efficient way to calculate cocitation counts
        for sample in tqdm(samples, "Collecting maximum cocitation counts"):
            paper_refs = set(sample.get("references", []))
            author_paper_refs = [set(p.get("references", [])) for p in sample["author"]["papers"]]
            
            # Calculate maximum cocitation count with any of author's papers
            # TODO: maybe add an option to count the total number of cocitations across all author's papers
            max_cocitations = max(
                (len(paper_refs & author_refs) for author_refs in author_paper_refs),
                default=0
            )
            cocitation_counts.append(max_cocitations)
        
        return cocitation_counts
    
    @staticmethod
    def _get_labels(samples: list) -> list:
        return [s["label"] for s in samples]
    
    def fit(self, train_samples: list, validation_samples: list):
        # No training needed for cocitation model
        X_train = np.array(self._process_samples(train_samples)).reshape(-1, 1)
        y_train = np.array(self._get_labels(train_samples))
        # X_val = np.array(self._process_samples(validation_samples)).reshape(-1, 1)
        # y_val = np.array(self._get_labels(validation_samples))
        self.model = LogisticRegression().fit(X_train, y_train)
        print("Logistic Regression trained parameters:", self.model.get_params())

    def predict_proba(self, samples: list):
        """
        Calculate cocitation logit score for list of samples
        1. For each sample, calculate the maximum number of cocitations between the paper and any of the author's papers
        2. Pass the counts through a trained logistic regression model
        """
        cocitation_counts = self._process_samples(samples)

        print("Computing Logistic scores")
        scores = self.model.predict_proba(np.array(cocitation_counts).reshape(-1, 1))[:, 1]

        print(f"Length of scores: {len(scores)}")
        print(f"Sample score: Cocitation count: {cocitation_counts[0]}, Sigmoid score: {scores[0]}")

        # TODO: Do I need to convert to different return type than numpy array?
        return scores

    def predict_proba_ranking(self, papers: list, authors: list):
        """
        Run inference on the cartesian product between all papers and all authors
        """
        # TODO: make faster implementation
        paper_refs = [set(p.get("references", [])) for p in papers]
        author_paper_refs = [[set(p.get("references", [])) for p in a['author']["papers"]] for a in authors]
        utility = np.zeros((len(authors), len(papers)))
        for i, author_refs in enumerate(author_paper_refs):
            for j, paper_ref in enumerate(paper_refs):
                utility[i, j] = max(
                    (len(paper_ref & a_ref) for a_ref in author_refs),
                    default=0
                )
        
        utility = self.model.predict_proba(utility.reshape(-1, 1)).reshape(len(authors), len(papers))[:,1]
        assert utility.shape == (len(authors), len(papers))
        return utility

    def _save(self, path: str):
        assert self.model is not None
        joblib.dump(self.model, os.path.join(path, "model.joblib"))

    def _load(self, path: str):
        self.model = joblib.load(os.path.join(path, "model.joblib"))