import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import joblib
from placeholder_embed import placeholder_embed

from models.base_model import BaseModel


class CosineSimilarityModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = None
        self.threshold = params.get('threshold', 0.5)

    @staticmethod
    def _process_author(sample: dict):
        # Get list of embeddings from author's papers
        author_embeddings = []
        for p in sample["author"]["papers"]:
            if "embedding" in p and p["embedding"] is not None:
                author_embeddings.append(p["embedding"])
        if len(author_embeddings) == 0:
            return [placeholder_embed]
        return np.array(author_embeddings)

    @staticmethod
    def _process_paper(sample: dict):
        return sample["embedding"]

    def _samples_to_arrays(self, samples: list):
        """Convert samples to paper and author embedding arrays"""
        paper_embeddings = []
        author_embeddings = []
        labels = []
        # print("samples: ", samples[:5])
        
        for sample in tqdm(samples, "CosineSim: samples -> arrays"):
            author_embed_list = self._process_author(sample)
            paper_embed = self._process_paper(sample)
            # if author_embed_list is None:
            #     # print("author_embed_list is empty for sample: ", sample)
            #     author_embed_list = [placeholder_embed]
            author_embeddings.append(author_embed_list)
            paper_embeddings.append(paper_embed)
            labels.append(sample["label"])
        # print("paper_embeddings: ", paper_embeddings[:5])
        # print("author_embeddings: ", author_embeddings[:5])
        max_similarities = []
        for target_emb, author_emb_list in zip(paper_embeddings, author_embeddings):
            # Compute cosine similarity between target and each of author's papers
            sims = cosine_similarity([target_emb], author_emb_list)[0]
            max_sim = np.max(sims)
            max_similarities.append(max_sim)

        X = np.array(max_similarities).reshape(-1, 1)
        y = np.array(labels)
            
        return X, y

    def fit(self, train_samples: list, validation_samples: list):
        """
        Use training set to find optimal threshold, validate on validation set
        """
        self.model = LogisticRegression(max_iter=1000)
        X, y = self._samples_to_arrays(train_samples)
        self.model.fit(X, y)
        self.threshold = self.model.coef_[0][0]
        print(f"Optimal threshold: {self.threshold:.3f}")
        
        X_val, y_val = self._samples_to_arrays(validation_samples)
        y_pred = self.model.predict(X_val)
        print(classification_report(y_val, y_pred))

    def _get_similarities(self, paper_embeddings: np.ndarray, author_embeddings_list: list):
        """Calculate max cosine similarity between paper and author's papers"""
        similarities = []
        for paper_emb, author_embs in zip(paper_embeddings, author_embeddings_list):
            if len(author_embs) == 0:
                similarities.append(0)
            else:
                # Get similarities between paper and all author papers
                sims = cosine_similarity(paper_emb.reshape(1,-1), author_embs)
                # Take maximum similarity
                similarities.append(np.max(sims))
        return np.array(similarities)

    def predict_proba(self, samples: list):
        """Run inference on a list of samples"""

        assert self.model is not None
        X, _ = self._samples_to_arrays(samples)
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_ranking(self, papers: list, authors: list):
        """Run inference on cartesian product of papers and authors"""
        paper_embeddings = np.array([self._process_paper(p) for p in papers])
        author_embeddings = [self._process_author(a) for a in authors]
        
        # Calculate similarities for each paper-author pair
        utility = np.zeros((len(authors), len(papers)))
        for i, author_embs in enumerate(author_embeddings):
            if len(author_embs) > 0:
                for j, paper_emb in enumerate(paper_embeddings):
                    sims = cosine_similarity(paper_emb.reshape(1,-1), author_embs)
                    utility[i,j] = np.max(sims)
        return utility
    
    def _save(self, path: str):
        """Save threshold and model"""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "threshold.npy"), self.threshold)
        joblib.dump(self.model, os.path.join(path, "model.joblib"))

    def _load(self, path: str):
        """Load threshold"""
        self.threshold = np.load(os.path.join(path, "threshold.npy"))
        self.model = joblib.load(os.path.join(path, "model.joblib"))