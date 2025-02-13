from models.base_model import BaseModel
from tqdm import tqdm
import pandas as pd
import numpy as np


class CocitationSigmoidModel(BaseModel):
    def __init__(self, params: dict) -> None:
        # tau and theta are parameters of the logistic function used to model the probability of a citation.:
        #   f(x) = 1 / (1 + exp(θ * (τ - x)))
        # where:
        # - θ (theta) is a parameter that controls the steepness of the curve.
        # - τ (tau) is the threshold value.
        # - x is the maximum number of cocitations between a paper and any of the the author's papers
        self.tau = params.get("tau", 5.0) # default set to match Kanakia et al. 2019 MAG paper
        self.theta = params.get("theta", 0.4) # default set to match Kanakia et al. 2019 MAG paper

    def _logistic_sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(self.theta * (self.tau - x)))
    
    def fit(self, train_samples: list, validation_samples: list):
        # No training needed for cocitation model
        print("Cocitation Sigmoid model does not require training - no parameters to fit")
        pass

    def predict_proba(self, samples: list):
        """
        Calculate cocitation logit score for list of samples
        1. For each sample, calculate the maximum number of cocitations between the paper and any of the author's papers
        2. Convert the counts to a sigmoid score
        """
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

        print("Computing Sigmoid scores")
        # Convert to sigmoid score (maps counts to range 0-1)
        scores = self._logistic_sigmoid(np.array(cocitation_counts))

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
        utility = self._logistic_sigmoid(utility)
        assert utility.shape == (len(authors), len(papers))
        return utility

    def _save(self, path: str):
        # No model parameters to save
        print("Cocitation Sigmoid model does not require saving - no parameters to save")
        pass

    def _load(self, path: str):
        # No model parameters to load
        print("Cocitation Sigmoid model does not require loading - no parameters to load")
        pass
