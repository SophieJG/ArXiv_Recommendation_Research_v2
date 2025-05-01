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
from util import EmbeddingDatabase

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


    def _process_author(self, sample: dict):
        # Get list of embeddings from author's papers
        author_embeddings = []
        paper_ids = []
        
        # First collect all paper IDs
        for p in sample["author"]["papers"]:
            paper_id = str(p.get("paper_id"))
            if paper_id:
                paper_ids.append(paper_id)
        
        if not paper_ids:
            return None
            
        # Try to get embeddings from ChromaDB
        try:
            embeddings = self.embedding_db.get_embeddings(paper_ids)
            for paper_id in paper_ids:
                if paper_id in embeddings:
                    author_embeddings.append(embeddings[paper_id])
        except Exception as e:
            print(f"Error getting embeddings for author papers: {e}")
            
        if len(author_embeddings) == 0:
            return None
        return np.array(author_embeddings)

    def _process_paper(self, sample: dict):
        # Try to get embedding from ChromaDB
        paper_id = str(sample.get("paper_id"))
        if paper_id:
            try:
                embedding = self.embedding_db.get_embeddings([paper_id])
                if paper_id in embedding:
                    return embedding[paper_id]
            except Exception as e:
                print(f"Error getting embedding for paper {paper_id}: {e}")
        # If no embedding found, use placeholder
        print(f"WARNING: No embedding found for target paper {paper_id} (corpus id)")
        return placeholder_embed

    def _samples_to_arrays(self, samples: list):
        """Convert samples to paper and author embedding arrays"""
        paper_embeddings = []
        author_embeddings = []
        labels = []
        # print("samples: ", samples[:5])
        
        max_similarities = []
        labels = []
        for sample in tqdm(samples, "CosineSim: samples -> arrays"):
            author_embed_list = self._process_author(sample)
            paper_embed = self._process_paper(sample)
            if author_embed_list is None:
                author_embed_list = [placeholder_embed]
            
            # Compute cosine similarity between target and each of author's papers
            sims = cosine_similarity([paper_embed], author_embed_list)[0]
            max_sim = np.max(sims)
            max_similarities.append(max_sim)
            labels.append(sample["label"])

        X = np.array(max_similarities).reshape(-1, 1)
        y = np.array(labels)
            
        return X, y

    def fit(self, train_samples: list, validation_samples: list):
        """
        Use training set to find optimal threshold, validate on validation set
        """
        # print("Cosine Similarity model does not require training - no parameters to fit")
        # return
        # Old implementation with LogisticRegression
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
        # X, _ = self._samples_to_arrays(samples)
        # return X
        # Old implementation with LogisticRegression
        assert self.model is not None
        X, _ = self._samples_to_arrays(samples)
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_ranking(self, papers: list, authors: list):
        """Run inference on cartesian product of papers and authors"""
        paper_embeddings = np.array([self._process_paper(p) for p in papers])
        author_embeddings = []
        for a in authors:
            author_emb = self._process_author(a)
            if author_emb is None:
                author_emb = [placeholder_embed]  # Use placeholder if no embeddings found
            author_embeddings.append(author_emb)
        
        # Calculate similarities for each paper-author pair
        utility = np.zeros((len(authors), len(papers)))
        for i, author_embs in enumerate(author_embeddings):
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