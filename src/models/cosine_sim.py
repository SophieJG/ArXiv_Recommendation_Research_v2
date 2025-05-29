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
        # Cache for embedding dimension and placeholder
        self._embedding_dim = None
        self._placeholder_embedding = None

    def _get_placeholder_embedding(self, target_dim: int) -> np.ndarray:
        """Get or create a placeholder embedding of the specified dimension"""
        if self._placeholder_embedding is None or len(self._placeholder_embedding) != target_dim:
            # Create a normalized random embedding as placeholder
            np.random.seed(42)  # For reproducibility
            self._placeholder_embedding = np.random.normal(0, 1, target_dim)
            # Normalize to unit length (typical for embeddings)
            self._placeholder_embedding = self._placeholder_embedding / np.linalg.norm(self._placeholder_embedding)
            self._embedding_dim = target_dim
        return self._placeholder_embedding.copy()

    def _detect_embedding_dimension(self, paper_ids: List[str]) -> int:
        """Detect embedding dimension by trying to get a sample embedding"""
        if self._embedding_dim is not None:
            return self._embedding_dim
            
        # Try to get a small sample to detect dimension
        sample_size = min(10, len(paper_ids))
        for i in range(0, len(paper_ids), max(1, len(paper_ids) // 10)):
            try:
                sample_ids = paper_ids[i:i + sample_size]
                ids, embeddings = self._get_embeddings_batch(sample_ids)
                if len(embeddings) > 0:
                    self._embedding_dim = embeddings.shape[1]
                    return self._embedding_dim
            except Exception:
                continue
        
        # Fallback to 768 if we can't detect
        print("Warning: Could not detect embedding dimension, using default 768")
        self._embedding_dim = 768
        return self._embedding_dim

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
            # Use default dimension if we can't detect
            return self._get_placeholder_embedding(768)
            
        try:
            ids, embeddings = self._get_embeddings_batch([paper_id])
            if len(embeddings) > 0:
                return embeddings[0]
        except Exception as e:
            print(f"Error getting embedding for paper {paper_id}: {e}")
            
        return self._get_placeholder_embedding(768)

    def _samples_to_arrays(self, samples: list) -> Tuple[np.ndarray, np.ndarray]:
        """Convert samples to paper and author embedding arrays"""
        max_similarities = []
        labels = []
        
        print(f"CosineSim: processing {len(samples)} samples...")
        
        # Get all paper IDs first
        paper_ids = [str(s.get("paper_id")) for s in samples]
        
        # Detect embedding dimension from paper IDs
        embedding_dim = self._detect_embedding_dimension(paper_ids)
        
        # Get all paper embeddings in one batch
        print("Getting paper embeddings...")
        ids, paper_embeddings = self._get_embeddings_batch(paper_ids)
        
        # Create mapping of paper_id to index for quick lookup
        paper_id_to_idx = {pid: i for i, pid in enumerate(ids)}
        
        # Process authors and compute similarities
        print("Processing authors and computing similarities...")
        for i, sample in enumerate(tqdm(samples, desc="Processing samples", miniters=len(samples)//100)):
            paper_id = str(sample.get("paper_id"))
            paper_emb = paper_embeddings[paper_id_to_idx[paper_id]] if paper_id in paper_id_to_idx else self._get_placeholder_embedding(embedding_dim)
            
            # Get author embeddings
            author_emb = self._process_author(sample)
            if author_emb is None:
                placeholder = self._get_placeholder_embedding(embedding_dim)
                author_emb = np.array([placeholder]) * -1
            
            # Compute similarity
            sims = cosine_similarity(paper_emb.reshape(1,-1), author_emb)[0]
            max_similarities.append(np.max(sims))
            labels.append(sample["label"])

        X = np.array(max_similarities).reshape(-1, 1)
        y = np.array(labels)
        return X, y

    @lru_cache(maxsize=1)
    def _get_paper_embeddings(self, paper_ids: tuple) -> np.ndarray:
        """Get embeddings for papers with caching - maintains order of paper_ids"""
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
        found_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        # Print stats about found embeddings
        print(f"Found embeddings for {len(found_embeddings)} out of {len(paper_ids)} papers "
              f"({len(found_embeddings)/len(paper_ids)*100:.1f}%)")
        
        # Detect embedding dimension
        if len(found_embeddings) > 0:
            embedding_dim = found_embeddings.shape[1]
        else:
            embedding_dim = self._detect_embedding_dimension(list(paper_ids))
        
        # Create mapping from found paper_id to its embedding index
        found_id_to_embedding_idx = {pid: i for i, pid in enumerate(all_ids)}
        
        # Initialize result array with placeholder embeddings (maintains order)
        placeholder = self._get_placeholder_embedding(embedding_dim)
        result_embeddings = np.tile(placeholder, (len(paper_ids), 1))
        
        # Build arrays for vectorized assignment
        target_indices = []  # Indices in result_embeddings where we'll place found embeddings
        source_indices = []  # Indices in found_embeddings to copy from
        
        for result_idx, paper_id in enumerate(paper_ids):
            if paper_id in found_id_to_embedding_idx:
                target_indices.append(result_idx)
                source_indices.append(found_id_to_embedding_idx[paper_id])
        
        # Vectorized assignment: place found embeddings in correct positions
        if target_indices:
            target_indices = np.array(target_indices)
            source_indices = np.array(source_indices)
            result_embeddings[target_indices] = found_embeddings[source_indices]
            
        return result_embeddings

    def predict_proba_ranking(self, papers: list, authors: list) -> np.ndarray:
        """Run inference on cartesian product of papers and authors with vectorized operations"""
        assert self.model is not None, "Model must be trained before prediction"
        
        # Get all paper IDs and process in batches
        paper_ids = [str(p.get("paper_id")) for p in papers]
        paper_embeddings = self._get_paper_embeddings(tuple(paper_ids))
        
        # Detect embedding dimension
        embedding_dim = paper_embeddings.shape[1] if len(paper_embeddings) > 0 else 768
        
        # Process authors
        author_embeddings = []
        for a in authors:
            author_emb = self._process_author(a)
            if author_emb is None:
                placeholder = self._get_placeholder_embedding(embedding_dim)
                author_emb = np.array([placeholder]) * -1
            author_embeddings.append(author_emb)
        
        # Pre-allocate utility matrix
        utility = np.zeros((len(authors), len(papers)))
        
        # Vectorized similarity calculation
        for i, author_embs in enumerate(author_embeddings):
            # Calculate similarities for all papers at once
            sims = cosine_similarity(author_embs, paper_embeddings)
            # Take max similarity for each paper
            utility[i] = np.max(sims, axis=0)

        # Convert cosine similarities to probabilities using trained logistic regression
        # Reshape to match the expected input format (n_samples, 1)
        similarities_flat = utility.flatten().reshape(-1, 1)
        probabilities_flat = self.model.predict_proba(similarities_flat)[:, 1]
        
        # Reshape back to utility matrix shape
        utility = probabilities_flat.reshape(len(authors), len(papers))
            
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