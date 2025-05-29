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
        Run inference on the cartesian product between all papers and all authors.
        For each author-paper pair, finds the maximum number of cocitations between the paper
        and any of the author's papers.
        """
        # Pre-compute all paper references once
        paper_refs = [set(p.get("references", [])) for p in papers]
        
        # Build a reference to paper index mapping for faster lookups
        ref_to_papers = {}
        for i, refs in enumerate(paper_refs):
            for ref in refs:
                if ref not in ref_to_papers:
                    ref_to_papers[ref] = []
                ref_to_papers[ref].append(i)
        
        # Initialize utility matrix
        utility = np.zeros((len(authors), len(papers)))
        
        # Process each author separately
        for i, author in enumerate(authors):
            # Extract all references from author's papers
            author_papers_refs = [set(p.get("references", [])) for p in author['author']["papers"]]
            
            if not author_papers_refs:
                continue  # Skip authors with no papers
                
            # For each paper of the author
            for author_paper_refs in author_papers_refs:
                # For each reference in this author's paper
                paper_cocitations = np.zeros(len(papers))
                
                # Count cocitations for all papers that share any reference with this author paper
                for ref in author_paper_refs:
                    if ref in ref_to_papers:
                        # Get indices of papers that cite this reference
                        for paper_idx in ref_to_papers[ref]:
                            # Increment cocitation count for this paper
                            paper_cocitations[paper_idx] += 1
                
                # Update maximum cocitations for each paper
                utility[i] = np.maximum(utility[i], paper_cocitations)
        
        # Convert raw cocitation counts to probabilities using logistic regression
        utility_reshaped = utility.reshape(-1, 1)
        probas = self.model.predict_proba(utility_reshaped)[:, 1]
        utility = probas.reshape(len(authors), len(papers))
        
        assert utility.shape == (len(authors), len(papers))
        return utility

    def _save(self, path: str):
        assert self.model is not None
        joblib.dump(self.model, os.path.join(path, "model.joblib"))

    def _load(self, path: str):
        self.model = joblib.load(os.path.join(path, "model.joblib"))