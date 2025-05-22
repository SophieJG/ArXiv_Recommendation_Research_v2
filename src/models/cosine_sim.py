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
from embedding_database import EmbeddingDatabase
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

from models.base_model import BaseModel


class CosineSimilarityModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = None
        self.threshold = params.get('threshold', 0.5)
        # Initialize embedding database
        self.embedding_db = EmbeddingDatabase(
            db_dir=params.get('vector_db_dir'),
            collection_name=params.get('vector_collection_name')
        )
        # Cache for paper embeddings
        self._paper_embeddings_cache = {}

    def _get_embeddings_batch(self, paper_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Get embeddings for a batch of paper IDs"""
        try:
            return self.embedding_db.get_embeddings(paper_ids)
        except Exception as e:
            print(f"Error getting embeddings batch: {e}")
            return np.array([]), np.array([])

    def _process_author(self, sample: dict) -> Optional[np.ndarray]:
        """Get list of embeddings from author's papers"""
        paper_ids = [str(p.get("paper_id")) for p in sample["author"]["papers"] if p.get("paper_id")]
        if not paper_ids:
            return None
            
        # Get embeddings in batch
        ids, embeddings = self._get_embeddings_batch(paper_ids)
        if len(embeddings) == 0:
            return None
        return embeddings

    def _process_paper(self, sample: dict) -> np.ndarray:
        """Get embedding for a single paper"""
        paper_id = str(sample.get("paper_id"))
        if not paper_id:
            return placeholder_embed
            
        try:
            ids, embeddings = self._get_embeddings_batch([paper_id])
            if len(embeddings) > 0:
                return embeddings[0]
        except Exception as e:
            print(f"Error getting embedding for paper {paper_id}: {e}")
            
        return placeholder_embed

    def _samples_to_arrays(self, samples: list) -> Tuple[np.ndarray, np.ndarray]:
        """Convert samples to paper and author embedding arrays with vectorized operations"""
        max_similarities = []
        labels = []
        
        # Process all papers first
        print("CosineSim: processing papers...")
        paper_embeddings = np.array([self._process_paper(s) for s in samples])
        
        # Process all authors
        print("CosineSim: processing authors...")
        author_embeddings = []
        for sample in tqdm(samples, "CosineSim: author samples -> arrays"):
            author_emb = self._process_author(sample)
            if author_emb is None:
                author_emb = np.array([placeholder_embed]) * -1
            author_embeddings.append(author_emb)
            labels.append(sample["label"])
        
        # Vectorized similarity calculation
        print("CosineSim: calculating similarities...")
        for paper_emb, author_embs in zip(paper_embeddings, author_embeddings):
            sims = cosine_similarity(paper_emb.reshape(1,-1), author_embs)[0]
            max_similarities.append(np.max(sims))

        X = np.array(max_similarities).reshape(-1, 1)
        y = np.array(labels)
        return X, y

    @lru_cache(maxsize=1)
    def _get_paper_embeddings(self, paper_ids: tuple) -> np.ndarray:
        """Get embeddings for papers with caching"""
        # Process papers in batches
        batch_size = 5000  # Same as EmbeddingDatabase's max_batch_size
        all_ids = []
        all_embeddings = []
        for i in range(0, len(paper_ids), batch_size):
            batch_ids = list(paper_ids[i:i + batch_size])  # Convert tuple slice back to list
            ids, embeddings = self._get_embeddings_batch(batch_ids)
            all_ids.extend(ids)
            all_embeddings.append(embeddings)
        
        # Combine all embeddings
        paper_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        # Fill in missing embeddings with placeholder
        if len(paper_embeddings) < len(paper_ids):
            # Create a mapping of paper_id to index for quick lookup
            paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
            
            # Create full embeddings array with placeholders
            full_embeddings = np.full((len(paper_ids), len(placeholder_embed)), -1)
            
            # Get indices where we have valid embeddings
            valid_indices = [paper_id_to_idx[pid] for pid in all_ids if pid in paper_id_to_idx]
            
            # Use advanced indexing to place embeddings in correct positions
            full_embeddings[valid_indices] = paper_embeddings
            paper_embeddings = full_embeddings
            
        return paper_embeddings

    def predict_proba_ranking(self, papers: list, authors: list) -> np.ndarray:
        """Run inference on cartesian product of papers and authors with vectorized operations"""
        # Get all paper IDs and process in batches
        paper_ids = [str(p.get("paper_id")) for p in papers]
        paper_embeddings = self._get_paper_embeddings(tuple(paper_ids))
        
        # Process authors
        author_embeddings = []
        for a in authors:
            author_emb = self._process_author(a)
            if author_emb is None:
                author_emb = np.array([placeholder_embed]) * -1
            author_embeddings.append(author_emb)
        
        # Pre-allocate utility matrix
        utility = np.zeros((len(authors), len(papers)))
        
        # Vectorized similarity calculation
        for i, author_embs in enumerate(author_embeddings):
            # Calculate similarities for all papers at once
            sims = cosine_similarity(author_embs, paper_embeddings)
            # Take max similarity for each paper
            utility[i] = np.max(sims, axis=0)
            
        return utility

    def fit(self, train_samples: list, validation_samples: list) -> None:
        """Use training set to find optimal threshold, validate on validation set"""
        self.model = LogisticRegression(max_iter=1000)
        X, y = self._samples_to_arrays(train_samples)
        self.model.fit(X, y)
        self.threshold = self.model.coef_[0][0]
        print(f"Optimal threshold: {self.threshold:.3f}")
        
        X_val, y_val = self._samples_to_arrays(validation_samples)
        y_pred = self.model.predict(X_val)
        print(classification_report(y_val, y_pred))

    def predict_proba(self, samples: list) -> np.ndarray:
        """Run inference on a list of samples"""
        assert self.model is not None
        X, _ = self._samples_to_arrays(samples)
        return self.model.predict_proba(X)[:, 1]

    def _save(self, path: str) -> None:
        """Save threshold and model"""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "threshold.npy"), self.threshold)
        joblib.dump(self.model, os.path.join(path, "model.joblib"))

    def _load(self, path: str) -> None:
        """Load threshold and model"""
        self.threshold = np.load(os.path.join(path, "threshold.npy"))
        self.model = joblib.load(os.path.join(path, "model.joblib"))