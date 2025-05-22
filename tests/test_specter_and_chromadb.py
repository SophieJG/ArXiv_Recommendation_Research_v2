import os
import shutil
import numpy as np
import json
import pytest
import torch
import copy

import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
# from src.models.specter2_basic import Specter2Basic

def test_chromadb_save_and_load():
    # Set up the data directory path
    db_dir = "tests/tests_data/chroma_test_path/chroma_db"
    
    try:
        # Initialize EmbeddingDatabase
        from src.embedding_database import EmbeddingDatabase
        embedding_db = EmbeddingDatabase(db_dir=db_dir, collection_name="test_coll")
        
        # Test data
        paper_ids = ["a", "b"]
        embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        # Store embeddings
        embedding_db.store_embeddings(paper_ids, embeddings)
        
        # Create a new instance to verify persistence
        embedding_db2 = EmbeddingDatabase(db_dir=db_dir, collection_name="test_coll")
        
        # Retrieve embeddings
        retrieved = embedding_db2.get_embeddings(paper_ids)
        
        # Test that the data was retrieved correctly
        for pid in paper_ids:
            np.testing.assert_allclose(
                retrieved[pid],
                embeddings[paper_ids.index(pid)],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Embedding mismatch for paper {pid}"
            )
        
        # Test has_embedding method
        assert embedding_db2.has_embedding("a")
        assert embedding_db2.has_embedding("b")
        assert not embedding_db2.has_embedding("c")
        
        # Verify the data is actually on disk
        assert os.path.exists(db_dir)
        assert len(os.listdir(db_dir)) > 0
        print(f"ChromaDB data successfully written to {os.path.abspath(db_dir)}")
        
    finally:
        # Clean up if needed - uncomment this if you want to remove the test data after test
        # If you want to keep the data, leave this commented out
        shutil.rmtree("tests/tests_data/chroma_test_path", ignore_errors=True)


def read_sems_data():
    filename = "tests/tests_data/papers_10.json"
    with open(filename, "r") as f:
        papers = json.load(f)
    return papers


def load_specter2_model_and_tokenizer_from_hf():
    base_model_name = "allenai/specter2_base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load base model
    model = AutoAdapterModel.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load the adapter and activate it
    adapter_name = 'allenai/specter2'
    print(f"Loading Specter2 adapter: {adapter_name}")
    adapter_name = model.load_adapter(adapter_name, source="hf", load_as="specter2")
    model.set_active_adapters(adapter_name)
    
    # Verify adapter activation
    if hasattr(model, 'active_adapters'):
        print(f"Active adapters: {model.active_adapters}")

    return model, tokenizer

def test_specter2_embedding():
    # load official HF Specter2
    model, tokenizer = load_specter2_model_and_tokenizer_from_hf()
    papers = read_sems_data()
    
    for id, paper in papers.items():
        if "abstract" not in paper or not paper["abstract"]:
            # for now, skip papers without abstract.
            # TODO: It is unclear right now how Semantic Scholar computes the embeddings for papers without abstract.
            #       It would be good to find out.
            continue
            
        text = paper["title"] + tokenizer.sep_token + paper["abstract"]
        
        # Tokenize the input
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Take the first token embedding (CLS token) as the document embedding
            model_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
        # ground‑truth from a prior run – replace with real numbers
        sems_emb = paper['embedding']
        l2_diff = np.linalg.norm(model_emb - sems_emb)
        
        # Check if L2 difference is within acceptable threshold
        assert l2_diff < 1e-04, f"L2 difference {l2_diff} is too large for paper {id}"

def test_specter2basic_save_and_load():
    # Load model and tokenizer from Hugging Face
    model, tokenizer = load_specter2_model_and_tokenizer_from_hf()
    
    save_path = "tests/tests_data/tmp_model"
    
    try:
        # Initialize Specter2Basic with the loaded model
        from src.models.specter2_basic import Specter2Basic
        specter_model = Specter2Basic({
            'load_path': None,  # Force loading from HF
            'batch_size': 16,
            'scaling_factor': 5.0,
            'vector_db_dir': "tests/tests_data/chroma_db"
        })
        
        # Save the model to a directory
        specter_model._save(save_path)
        
        # Load the model from the saved directory
        loaded_model = Specter2Basic({
            'load_path': save_path,
            'batch_size': 16,
            'scaling_factor': 5.0,
            'vector_db_dir': "tests/tests_data/chroma_db"
        })
        
        # Make sure that both models are not using ChromaDB to generate embeddings
        specter_model.use_chroma_db = False
        loaded_model.use_chroma_db = False
        
        # Test with papers from the test data
        papers = read_sems_data()
        tested_paper = False
        
        # Convert papers to list format for _encode_batch
        paper_list = []
        for paper_id, paper in papers.items():
            if "abstract" in paper and paper["abstract"]:
                paper["paper_id"] = int(paper_id)
                paper_list.append(paper)
        
        if paper_list:
            # Get embeddings from both models using _encode_batch
            orig_embeddings = specter_model._encode_batch(paper_list)
            loaded_embeddings = loaded_model._encode_batch(paper_list)
            
            # Compare embeddings
            np.testing.assert_allclose(
                loaded_embeddings,
                orig_embeddings,
                rtol=1e-5,
                atol=1e-8,
                err_msg="Paper embeddings from saved and loaded model do not match original model"
            )
            
            # Also compare with the ground truth from the papers
            for i, paper in enumerate(paper_list):
                l2_diff = np.linalg.norm(loaded_embeddings[i] - paper['embedding'])
                # Check if L2 difference is within acceptable threshold
                assert l2_diff < 1e-04, f"L2 difference {l2_diff} is too large for paper {paper['paper_id']}"
        
        # Ensure we tested at least one paper
        assert len(paper_list) > 0, "No papers with abstracts were found in the test data"
        
    finally:
        # Clean up - this will run even if the test fails
        try:
            # Clear any cached files
            if 'specter_model' in locals() and hasattr(specter_model.model, 'clear_cache'):
                specter_model.model.clear_cache()
            if 'loaded_model' in locals() and hasattr(loaded_model.model, 'clear_cache'):
                loaded_model.model.clear_cache()
                
            # Delete the saved model directory
            if os.path.exists(save_path):
                shutil.rmtree(save_path, ignore_errors=True)
                
            # Clean up ChromaDB directory
            if os.path.exists("tests/tests_data/chroma_db"):
                shutil.rmtree("tests/tests_data/chroma_db", ignore_errors=True)
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")
