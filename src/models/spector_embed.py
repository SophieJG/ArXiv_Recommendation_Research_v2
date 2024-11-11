import os
import pandas as pd
import numpy as np
import joblib
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from data import Data
from models.base_model import BaseModel
from concurrent.futures import ProcessPoolExecutor
import threading


class Specter2EmbeddingsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="allenai/specter_plus_plus", adapter_name="allenai/specter2"):
        # Load the tokenizer and model with the SPECTER2 adapter
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.model.load_adapter(adapter_name, set_active=True)
        self.embed_dict = {} 
        self.lock = threading.Lock()  # Initialize a lock

    def fit(self, X, y=None):
        # No fitting required for this transformer
        return self

    def get_abstract_hash(self, abstract):
        return hashlib.sha256(abstract.encode()).hexdigest()


    def transform(self, X, y=None):
        print("Calculating embeddings for SPECTER2 model")
        if isinstance(X, np.ndarray):
            X = X.flatten().tolist()
        elif hasattr(X, "tolist"):
            X = X.tolist()

        # Use ProcessPoolExecutor for multiprocessing
        max_workers = os.cpu_count() or 1
        print(f"Using {max_workers} processes for embedding computation")

        # Use ProcessPoolExecutor with individual memory space to parallelize the embedding computation
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(self.compute_embedding, X))

        print("Embedding calculation completed.")
        return np.array(embeddings)
    
    def compute_embedding(self, text):
        # Initialize model and tokenizer within each process
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoAdapterModel.from_pretrained(self.model_name)
        model.load_adapter(self.adapter_name, set_active=True)
        model.eval()
        embedding_size = model.config.hidden_size

        if isinstance(text, list):
            text = " ".join(text)
        text = text.strip()
        if not text:
            return np.zeros(embedding_size)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
        return embedding

    def get_params(self, deep=True):
        # Return the parameters to make it compatible with scikit-learn cloning
        return {"model_name": self.model_name, "adapter_name": self.adapter_name}

    def set_params(self, **params):
        # Update parameters for compatibility with scikit-learn cloning
        for param, value in params.items():
            setattr(self, param, value)
        # Reload the tokenizer and model when parameters change
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoAdapterModel.from_pretrained(self.model_name)
        self.model.load_adapter(self.adapter_name, set_active=True)
        return self