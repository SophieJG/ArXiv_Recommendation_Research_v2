from models.base_model import BaseModel
import numpy as np
import os
import joblib
import torch
import chromadb
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from tqdm import tqdm

def euclidean_distance(a, b):
    # Compute Euclidean distance between 1-D arrays a and b
    return np.linalg.norm(a - b)

def batch_euclidean_distance(a, b_matrix):
    """
    Compute Euclidean distance between a vector and a matrix of vectors efficiently
    
    Args:
        a: Single vector of shape (d,)
        b_matrix: Matrix of vectors of shape (n, d)
    
    Returns:
        Array of distances of shape (n,)
    """
    # Compute pairwise distances in a vectorized way
    # Using the formula ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    a_squared = np.sum(a**2)
    b_squared = np.sum(b_matrix**2, axis=1)
    dot_product = np.dot(b_matrix, a)
    
    # Calculate distances
    distances = np.sqrt(a_squared + b_squared - 2 * dot_product)
    return distances

def euclidean_similarity(distances, scaling_factor=5.0):
    """
    Convert Euclidean distances to similarity scores between 0 and 1
    using a negative exponential transformation
    
    Args:
        distances: Array of Euclidean distances
        scaling_factor: Controls how quickly similarity decreases with distance
    
    Returns:
        Array of similarities in range [0, 1]
    """
    return np.exp(-distances / scaling_factor)

class Specter2Basic(BaseModel):
    def __init__(self, params: dict) -> None:
        # Load specter2 model properly using adapters following Hugging Face example
        base_model_name = 'allenai/specter2_base'
        adapter_name = 'allenai/specter2'

        load_path = params.get('load_path')
        
        # Check if we should load from local path or from Hugging Face
        if load_path and os.path.exists(load_path):
            # Check for model files
            has_model_file = (os.path.exists(os.path.join(load_path, "config.json")) and 
                             (os.path.exists(os.path.join(load_path, "model.safetensors")) or 
                              os.path.exists(os.path.join(load_path, "pytorch_model.bin"))))
            
            # Check for tokenizer files
            has_tokenizer_file = os.path.exists(os.path.join(load_path, "tokenizer_config.json"))
            
            # Check for adapter directory
            specter2_dir = os.path.join(load_path, "specter2")
            has_adapter = (os.path.exists(specter2_dir) and 
                          os.path.exists(os.path.join(specter2_dir, "adapter_config.json")) and
                          os.path.exists(os.path.join(specter2_dir, "pytorch_adapter.bin")))
            
            if has_model_file and has_tokenizer_file:
                print(f"Loading saved model from {load_path}")
                try:
                    # Use the _load method for consistency
                    self._load(load_path)
                    
                    # Verify adapter status
                    if hasattr(self.model, 'active_adapters'):
                        print(f"Active adapters after loading: {self.model.active_adapters}")
                except Exception as e:
                    print(f"Error loading saved model: {e}")
                    print("Falling back to loading model from Hugging Face")
                    has_model_file = False
            else:
                missing = []
                if not has_model_file:
                    missing.append("model files")
                if not has_tokenizer_file:
                    missing.append("tokenizer files")
                if not has_adapter:
                    missing.append("adapter files")
                    
                print(f"Path {load_path} exists but is missing required {', '.join(missing)}")
                has_model_file = False
        else:
            has_model_file = False
                
        # If we couldn't load from local path, load from Hugging Face
        if not has_model_file:
            print(f"Loading Specter2 base model: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # Load base model
            self.model = AutoAdapterModel.from_pretrained(base_model_name)
            
            # Load the adapter and activate it
            print(f"Loading Specter2 adapter: {adapter_name}")
            adapter_name = self.model.load_adapter(adapter_name, source="hf", load_as="specter2")
            self.model.set_active_adapters(adapter_name)
            print(f"Activated adapter: {adapter_name}")
            
            # Verify adapter activation
            if hasattr(self.model, 'active_adapters'):
                print(f"Active adapters: {self.model.active_adapters}")
        
        self.model.eval()  # set to evaluation mode
        self.batch_size = params.get('batch_size', 16)
        # Scaling factor for euclidean similarity conversion
        self.scaling_factor = params.get('scaling_factor', 5.0)
        
        # Initialize ChromaDB for vector storage
        vector_db_dir = params.get('vector_db_dir')
        collection_name = params.get('vector_collection_name', 'specter_embeddings')
        if vector_db_dir:
            print(f"Initializing ChromaDB at {vector_db_dir}")
            try:
                os.makedirs(vector_db_dir, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=vector_db_dir)
                # Create or get the collection for paper embeddings
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=collection_name, 
                    metadata={"description": "Specter2 paper embeddings"}
                )
                self.use_vector_db = True
            except Exception as e:
                print(f"Error initializing ChromaDB: {e}")
                print("Continuing without vector database...")
                self.use_vector_db = False
        else:
            print("No vector database directory provided, embeddings will not be cached")
            self.use_vector_db = False

    def _extract_text(self, paper):
        """Format paper text following Specter2 conventions: title + sep_token + abstract"""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        # Concatenate title and abstract as shown in Hugging Face example
        return title + self.tokenizer.sep_token + abstract
            
    def _encode_batch(self, papers, batch_size=None):
        """Encode a batch of papers to embeddings"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # Prepare to track which papers need encoding and which can be retrieved from DB
        paper_ids = []
        texts = []
        for paper in papers:
            paper_id = str(paper["paper_id"])
            paper_ids.append(paper_id)
            
            if isinstance(paper, dict) and "embedding" in paper:
                # Skip if we'll use precomputed embedding
                texts.append(None)
            else:
                texts.append(self._extract_text(paper))
        
        # Initialize embeddings list
        embeddings = [None] * len(papers)
        
        # First try to get embeddings from ChromaDB for papers with IDs
        papers_to_encode_indices = []
        
        if self.use_vector_db:
            # Get valid paper IDs (non-empty)
            valid_indices = [i for i, pid in enumerate(paper_ids) if pid]
            valid_paper_ids = [paper_ids[i] for i in valid_indices]
            
            if valid_paper_ids:
                try:
                    # Query ChromaDB for existing embeddings
                    results = self.chroma_collection.get(
                        ids=valid_paper_ids,
                        include=["embeddings"]
                    )
                    
                    # Process the results
                    found_ids = results.get("ids", [])
                    found_embeddings = results.get("embeddings", [])
                    
                    # Map found embeddings to their positions
                    for idx, paper_id in enumerate(found_ids):
                        original_idx = paper_ids.index(paper_id)
                        embeddings[original_idx] = np.array(found_embeddings[idx])
                        
                    # print(f"Retrieved {len(found_ids)} embeddings from ChromaDB")
                except Exception as e:
                    print(f"Error retrieving embeddings from ChromaDB: {e}")
        
        # Identify which papers still need encoding
        for i, embedding in enumerate(embeddings):
            if embedding is None and texts[i] is not None:
                papers_to_encode_indices.append(i)
            elif isinstance(papers[i], dict) and "embedding" in papers[i]:
                # Use precomputed embedding from paper
                embeddings[i] = papers[i]["embedding"]
        
        # Encode papers that weren't found in ChromaDB
        if papers_to_encode_indices:
            papers_to_encode = [(i, texts[i]) for i in papers_to_encode_indices]
            
            for i in range(0, len(papers_to_encode), batch_size):
                batch_indices = [p[0] for p in papers_to_encode[i:i+batch_size]]
                batch_texts = [p[1] for p in papers_to_encode[i:i+batch_size]]
                
                # Preprocess input following Hugging Face example
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True,
                    return_tensors="pt", 
                    return_token_type_ids=False, 
                    max_length=512
                )
                
                # Move inputs to the same device as the model
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Ensure adapter is activated before inference
                if hasattr(self.model, 'active_adapters') and not self.model.active_adapters:
                    print("Re-activating adapter for inference")
                    self.model.set_active_adapters("specter2")
                    
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Take the first token embedding as in the Hugging Face example
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Put batch embeddings back in original order and store in ChromaDB
                for j, idx in enumerate(batch_indices):
                    embeddings[idx] = batch_embeddings[j]
                    
                    # Store in ChromaDB if we have a paper_id and vector_db is enabled
                    if self.use_vector_db and paper_ids[idx]:
                        try:
                            # Store new embedding in ChromaDB
                            self.chroma_collection.upsert(
                                ids=[paper_ids[idx]],
                                embeddings=[batch_embeddings[j].tolist()],
                                metadatas=[{"paper_id": paper_ids[idx]}]
                            )
                        except Exception as e:
                            print(f"Error storing embedding in ChromaDB: {e}")
        
        return np.array(embeddings)

    def fit(self, train_samples: list, validation_samples: list):
        """No training needed for SPECTER2 model - it's already trained"""
        print("Specter2 model is already trained - skipping fit step")
        return

    def predict_proba(self, samples: list):
        """
        Predict probabilities by computing max similarity (based on Euclidean distance)
        between target paper and author's papers
        """
        print(f"Predicting probabilities for {len(samples)} samples...")
        probabilities = []
        
        for sample in tqdm(samples, desc="Computing similarities"):
            # Get target paper embedding
            target_paper_embedding = self._encode_batch([sample])[0]
            
            # Get embeddings for author's papers
            author_papers = sample["author"]["papers"]
            if not author_papers:
                # No papers from author, default to low similarity
                print(f"Oh no! No papers from author {sample['author']['id']}")
                probabilities.append(0.0)
                continue
                
            author_paper_embeddings = self._encode_batch(author_papers)
            
            # Compute Euclidean distances
            distances = batch_euclidean_distance(target_paper_embedding, author_paper_embeddings)
            
            # Convert distances to similarities (smaller distance = higher similarity)
            similarities = euclidean_similarity(distances, self.scaling_factor)
            
            # Get max similarity
            max_similarity = np.max(similarities)
            
            probabilities.append(max_similarity)
        
        # Debug output
        if samples:
            print(f"Sample prediction: paper: {samples[0].get('title', '')} | "
                  f"Author papers: {len(samples[0]['author']['papers'])} | "
                  f"Probability: {probabilities[0]}")
            print(f"\nSample sample: \n{samples[0]}\n")
            
        return np.array(probabilities)

    def predict_proba_ranking(self, papers: list, authors: list):
        """Predict ranking matrix between papers and authors efficiently"""
        # print(f"Computing utility matrix for {len(authors)} authors...")
        paper_embeddings = self._encode_batch(papers)
        
        # Compute utility matrix
        utility = np.zeros((len(authors), len(papers)))
        
        for i, author in enumerate(tqdm(authors, desc="Processing authors")):
            # Get embeddings for all author's papers
            author_papers = author.get("papers", [])
            if not author_papers:
                # No papers from this author, leave similarities as 0
                continue
                
            author_paper_embeddings = self._encode_batch(author_papers)
            
            # For each paper embedding, compute similarity with all author paper embeddings at once
            for j, paper_emb in enumerate(paper_embeddings):
                # Compute Euclidean distances with all author papers
                distances = batch_euclidean_distance(paper_emb, author_paper_embeddings)
                
                # TODO: Figure out if ranking scores need to be in range [0, 1] or not
                # # Convert distances to similarities
                # similarities = euclidean_similarity(distances, self.scaling_factor)
                
                # Get maximum similarity
                max_similarity = np.max(-distances)
                
                utility[i, j] = max_similarity
        
        assert utility.shape == (len(authors), len(papers))
        return utility

    def _save(self, path: str):
        """Save the model parameters with adapters"""
        print(f"Saving Specter2 model to {path}")
        self.model.save_all_adapters(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _load(self, path: str):
        """Load the model parameters with adapters"""
        print(f"Loading Specter2 model from {path}")
        
        # Load the tokenizer from local path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load the model from local path
        self.model = AutoAdapterModel.from_pretrained(path)
        
        # Check for adapter
        specter2_dir = os.path.join(path, "specter2")
        if os.path.exists(specter2_dir):
            print(f"Found adapter at {specter2_dir}, activating it")
            adapter_name = self.model.load_adapter(specter2_dir, load_as="specter2")
            self.model.set_active_adapters(adapter_name)
            print(f"Activated adapter: {adapter_name}")
        else:
            print(f"No adapter found at {path}/specter2")
                
        self.model.eval()