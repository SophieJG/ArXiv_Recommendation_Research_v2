import os

import numpy as np


def data_dir(config: dict):
    return os.path.join(config["data"]["base_path"], "data")


def tmp_data_dir(config: dict):
    return os.path.join(config["data"]["base_path"], "data", "tmp")


def models_dir(config: dict):
    return os.path.join(config["data"]["base_path"], "models")


def model_version_path(model_path, model: str, version: str):
    return os.path.join(model_path, f"{model}.{version}")


def papers_path(config: dict):
    return os.path.join(data_dir(config), "papers.json")


def authors_path(config: dict):
    return os.path.join(data_dir(config), "authors.json")


def kaggle_data_path(config: dict):
    return os.path.join(data_dir(config), "kaggle_data.parquet")


def embedding_db_dir(config: dict):
    return os.path.join(config["data"]["vector_db_dir"])


def passthrough_func(x):
    return x


def mean_consine_distance(embedding: list):
    """
Calculate average cosine distance between all embedding vectors in a list. It is assumed that the 
vectors are normalized to have an l2 norm of 1.
"""
    embeddings = np.vstack(embedding)
    return np.mean(np.matmul(embeddings, embeddings.transpose()))


class EmbeddingDatabase:
    def __init__(self, db_dir: str, collection_name: str = "paper_embeddings"):
        """
        Initialize an embedding database using ChromaDB.
        
        Args:
            db_dir: Directory where the ChromaDB data will be stored
            collection_name: Name of the collection to store embeddings in
        """
        import chromadb
        os.makedirs(db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Paper embeddings database"}
        )
        self.max_batch_size = 5000  # ChromaDB's limit is 5461
        
    def store_embeddings(self, paper_ids: list, embeddings: list):
        """
        Store embeddings for a list of papers in batches to handle large datasets.
        
        Args:
            paper_ids: List of paper IDs (strings)
            embeddings: List of embedding vectors (lists of floats)
        """
        # Convert embeddings to lists if they're numpy arrays
        embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        
        # Process in batches
        for i in range(0, len(paper_ids), self.max_batch_size):
            batch_ids = paper_ids[i:i + self.max_batch_size]
            batch_embeddings = embeddings[i:i + self.max_batch_size]
            batch_metadatas = [{"paper_id": pid} for pid in batch_ids]
            
            try:
                # Store in ChromaDB
                self.collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                print(f"Error storing batch {i//self.max_batch_size + 1}/{(len(paper_ids) + self.max_batch_size - 1) // self.max_batch_size}: {e}")
                print(f"Batch size: {len(batch_ids)}")
                raise e
        
    def get_embeddings(self, paper_ids: list):
        """
        Retrieve embeddings for a list of papers.
        
        Args:
            paper_ids: List of paper IDs to retrieve embeddings for
            
        Returns:
            tuple of (ids, embeddings) where:
            - ids: numpy array of paper IDs that were found
            - embeddings: numpy array of corresponding embeddings
        """
        results = self.collection.get(
            ids=paper_ids,
            include=["embeddings"]
        )
        
        # Convert to numpy arrays
        ids = np.array(results["ids"])
        embeddings = np.array(results["embeddings"])
            
        return ids, embeddings
        
    def has_embedding(self, paper_id: str) -> bool:
        """
        Check if an embedding exists for a given paper ID.
        
        Args:
            paper_id: ID of the paper to check
            
        Returns:
            bool indicating if the embedding exists
        """
        try:
            results = self.collection.get(ids=[paper_id])
            return len(results["ids"]) > 0
        except:
            return False
