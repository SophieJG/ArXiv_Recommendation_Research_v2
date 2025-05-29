import os
import numpy as np
import chromadb
import json
import subprocess
import atexit
import signal
import time


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

class RedisEmbeddingDatabase:
    def __init__(self, db_dir: str, collection_name: str = "paper_embeddings", dim=768, 
                 distance_metric="COSINE", prefix="embedding:", port=6379):
        """
        Initialize a local Redis-based embedding database.
        
        Args:
            db_dir: Directory where Redis data will be stored
            collection_name: Name of the collection (used as part of the key prefix)
            dim: Dimension of embedding vectors
            distance_metric: Distance metric for vector search ("COSINE", "IP" for inner product, or "L2")
            prefix: Prefix for Redis keys
            port: Port to use for the local Redis server
        """
        raise NotImplementedError("RedisEmbeddingDatabase is not configured for G2 cluster at this time")
        import redis
        from redis.commands.search.field import VectorField, TextField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        
        # Store parameters
        self.db_dir = os.path.abspath(db_dir)
        self.collection_name = collection_name
        self.prefix = f"{prefix}{collection_name}:"
        self.dim = dim
        self.max_batch_size = 5000
        self.port = port
        
        # Create data directory
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Start local Redis server if not already running
        self._start_redis_server()
        
        # Connect to Redis
        self.client = redis.Redis(host='localhost', port=port, db=0, decode_responses=False)
        
        # Create search index if it doesn't exist
        index_name = f"{collection_name}_idx"
        try:
            # Check if index exists
            self.client.ft(index_name).info()
            self.index_exists = True
        except:
            # Create index
            vector_field = VectorField("embedding",
                                      "FLAT", {
                                          "TYPE": "FLOAT32",
                                          "DIM": dim,
                                          "DISTANCE_METRIC": distance_metric,
                                          "INITIAL_CAP": 10000,
                                      })
            paper_id_field = TextField("paper_id")
            
            self.client.ft(index_name).create_index(
                fields=[vector_field, paper_id_field],
                definition=IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
            )
            self.index_exists = True
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _start_redis_server(self):
        """Start a local Redis server if not already running."""
        import redis
        
        # Check if Redis is already running
        try:
            r = redis.Redis(host='localhost', port=self.port, socket_timeout=1)
            r.ping()
            return  # Redis is already running
        except:
            pass  # Redis is not running, continue to start it
        
        # Start Redis server
        redis_conf = os.path.join(self.db_dir, "redis.conf")
        with open(redis_conf, "w") as f:
            f.write(f"""
dir {self.db_dir}
port {self.port}
daemonize yes
pidfile {os.path.join(self.db_dir, "redis.pid")}
""")
        
        # Start Redis server
        subprocess.Popen(["redis-server", redis_conf])
        
        # Wait for Redis to start
        for _ in range(10):  # Try for 10 seconds
            try:
                r = redis.Redis(host='localhost', port=self.port, socket_timeout=1)
                r.ping()
                break
            except:
                time.sleep(1)
        else:
            raise RuntimeError("Failed to start Redis server")
    
    def cleanup(self):
        """Clean up Redis server when the database is closed."""
        try:
            # Get Redis PID
            pid_file = os.path.join(self.db_dir, "redis.pid")
            if os.path.exists(pid_file):
                with open(pid_file, "r") as f:
                    pid = int(f.read().strip())
                
                # Kill Redis server
                os.kill(pid, signal.SIGTERM)
                
                # Remove PID file
                os.remove(pid_file)
        except:
            pass  # Ignore cleanup errors
    
    def store_embeddings(self, paper_ids: list, embeddings: list):
        """
        Store embeddings for a list of papers in batches.
        
        Args:
            paper_ids: List of paper IDs (strings)
            embeddings: List of embedding vectors (lists of floats or numpy arrays)
        """
        import numpy as np
        
        # Process in batches
        pipe = self.client.pipeline()
        
        for i in range(0, len(paper_ids), self.max_batch_size):
            batch_ids = paper_ids[i:i + self.max_batch_size]
            batch_embeddings = embeddings[i:i + self.max_batch_size]
            
            for j, (pid, emb) in enumerate(zip(batch_ids, batch_embeddings)):
                # Convert numpy array to bytes
                if isinstance(emb, np.ndarray):
                    emb_bytes = emb.astype(np.float32).tobytes()
                else:
                    emb_bytes = np.array(emb, dtype=np.float32).tobytes()
                
                # Store in Redis
                key = f"{self.prefix}{pid}"
                pipe.hset(key, mapping={
                    "embedding": emb_bytes,
                    "paper_id": pid
                })
            
            # Execute batch
            try:
                pipe.execute()
                pipe = self.client.pipeline()  # Create a new pipeline for the next batch
            except Exception as e:
                print(f"Error storing batch {i//self.max_batch_size + 1}/{(len(paper_ids) + self.max_batch_size - 1) // self.max_batch_size}: {e}")
                print(f"Batch size: {len(batch_ids)}")
                raise e
    
    def get_embeddings(self, paper_ids: list):
        """
        Retrieve embeddings for a list of papers in batches.
        
        Args:
            paper_ids: List of paper IDs to retrieve embeddings for
            
        Returns:
            tuple of (ids, embeddings) where:
            - ids: numpy array of paper IDs that were found
            - embeddings: numpy array of corresponding embeddings
        """
        import numpy as np
        
        result_ids = []
        result_embeddings = []
        
        # Process in batches
        for i in range(0, len(paper_ids), self.max_batch_size):
            batch_ids = paper_ids[i:i + self.max_batch_size]
            
            # Create pipeline for batch retrieval
            pipe = self.client.pipeline()
            for pid in batch_ids:
                key = f"{self.prefix}{pid}"
                pipe.hget(key, "embedding")
            
            # Execute batch
            batch_results = pipe.execute()
            
            # Process results
            for pid, emb_bytes in zip(batch_ids, batch_results):
                if emb_bytes is not None:
                    result_ids.append(pid)
                    embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                    result_embeddings.append(embedding)
        
        return np.array(result_ids), np.array(result_embeddings)
    
    def has_embedding(self, paper_id: str) -> bool:
        """
        Check if an embedding exists for a given paper ID.
        
        Args:
            paper_id: ID of the paper to check
            
        Returns:
            bool indicating if the embedding exists
        """
        key = f"{self.prefix}{paper_id}"
        return self.client.exists(key) > 0