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


class Specter2EmbeddingsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="allenai/specter_plus_plus", adapter_name="allenai/specter2"):
        # Load the tokenizer and model with the SPECTER2 adapter
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.model.load_adapter(adapter_name, set_active=True)

    def fit(self, X, y=None):
        # No fitting required for this transformer
        return self

    def get_abstract_hash(self, abstract):
        return hashlib.md5(abstract.encode()).hexdigest()


    def transform(self, X, y=None):
        print("Calculating embeddings for SPECTER2 model")

        # Ensure X is converted to a list if it is a pandas Series or other non-list structure
        if isinstance(X, np.ndarray):
            X = X.flatten().tolist()  # Convert numpy arrays to a flat list
        elif hasattr(X, "tolist"):  # Handle pandas series/dataframes
            X = X.tolist()
            
        embed_dict = {}
        embeddings = []
        for text in X:
            if isinstance(text, list):
                # If text is still a list (in case a list of strings is passed), flatten it into a single string
                text = " ".join(text)

            abstract_hash = self.get_abstract_hash(text)

            if abstract_hash not in embed_dict:
                # Tokenize the input, truncating it to the max_length (512 tokens)
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # The output is a hidden state from the model, so we average over tokens
                embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
                embed_dict[abstract_hash] = embedding
            else:
                embedding = embed_dict[abstract_hash]

            embeddings.append(embedding)
            
        print("Embedding calculation completed.")
        return np.array(embeddings)

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