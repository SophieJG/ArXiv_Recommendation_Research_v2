#!/usr/bin/env python3
"""
Test script to verify that CosineSimilarityModel can handle different embedding dimensions
"""
# NOTE: This test script has not been tried yet.

import numpy as np
import sys
import os

# Add src to path (now from tests directory)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cosine_sim import CosineSimilarityModel

def test_embedding_dimension_robustness():
    """Test that the model works with different embedding dimensions"""
    
    # Mock params
    params = {
        'threshold': 0.5,
        'vector_db_dir': '/tmp/test_db',
        'vector_collection_name': 'test_collection'
    }
    
    # Create model
    model = CosineSimilarityModel(params)
    
    # Test 1: Check placeholder creation for different dimensions
    print("Test 1: Testing placeholder embedding creation")
    
    # Test various dimensions
    dimensions = [128, 256, 512, 768, 1024]
    
    for dim in dimensions:
        placeholder = model._get_placeholder_embedding(dim)
        print(f"  Dimension {dim}: Created placeholder of shape {placeholder.shape}")
        
        # Verify it's normalized (unit length)
        norm = np.linalg.norm(placeholder)
        print(f"  Dimension {dim}: Norm = {norm:.6f} (should be ~1.0)")
        
        # Verify dimension
        assert len(placeholder) == dim, f"Expected dimension {dim}, got {len(placeholder)}"
        assert 0.99 < norm < 1.01, f"Expected unit norm, got {norm}"
    
    # Test 2: Check dimension detection fallback
    print("\nTest 2: Testing dimension detection fallback")
    
    # Mock dimension detection when no embeddings are found
    detected_dim = model._detect_embedding_dimension([])
    print(f"  Detected dimension with empty list: {detected_dim}")
    assert detected_dim == 768, f"Expected fallback to 768, got {detected_dim}"
    
    print("\nAll tests passed! ✅")
    print("The CosineSimilarityModel is now robust to different embedding dimensions.")

if __name__ == "__main__":
    test_embedding_dimension_robustness() 