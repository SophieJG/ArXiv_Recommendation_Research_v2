import os
import numpy as np
import pytest
import shutil

def test_redis_embedding_database():
    """Test the RedisEmbeddingDatabase class with a local Redis server."""
    pytest.skip("RedisEmbeddingDatabase is not configured for G2 cluster at this time")
    from src.embedding_database import RedisEmbeddingDatabase
    
    # Skip test if Redis is not available
    try:
        import redis
        # Try to connect to Redis, with a short timeout
        r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=1)
        r.ping()
    except (ImportError, redis.exceptions.ConnectionError):
        pytest.skip("Redis server not available")
    
    # Create a test database directory
    test_db_dir = "tests/tests_data/redis_test_db"
    os.makedirs(test_db_dir, exist_ok=True)
    
    try:
        # Create a test database with a unique collection name
        collection_name = "test_collection_redis"
        
        # Initialize RedisEmbeddingDatabase
        redis_db = RedisEmbeddingDatabase(
            db_dir=test_db_dir,
            collection_name=collection_name,
            dim=3  # Using small embeddings for testing
        )
        
        # Test data
        paper_ids = ["test1", "test2"]
        embeddings = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32)
        ]
        
        # Store embeddings
        redis_db.store_embeddings(paper_ids, embeddings)
        
        # Create a new instance to ensure we're not just accessing in-memory data
        redis_db2 = RedisEmbeddingDatabase(
            db_dir=test_db_dir,
            collection_name=collection_name,
            dim=3
        )
        
        # Test has_embedding method
        assert redis_db2.has_embedding("test1")
        assert redis_db2.has_embedding("test2")
        assert not redis_db2.has_embedding("nonexistent")
        
        # Test get_embeddings
        retrieved_ids, retrieved_embeddings = redis_db2.get_embeddings(paper_ids)
        
        # Verify IDs are returned correctly
        assert set(retrieved_ids) == set(paper_ids)
        
        # Verify embeddings
        for i, pid in enumerate(retrieved_ids):
            original_idx = paper_ids.index(pid)
            original_emb = embeddings[original_idx]
            
            # Compare embeddings
            np.testing.assert_allclose(
                retrieved_embeddings[i],
                original_emb,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Embedding mismatch for paper {pid}"
            )
        
    finally:
        # Clean up - remove test directory
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)
