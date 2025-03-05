import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import numpy as np


class Specter2Embedder:
    def __init__(self, model_name="allenai/specter2_base", adapter_name="allenai/specter2", device=None):
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.max_length = 512
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.model.load_adapter(adapter_name, set_active=True)
        self.model.to(self.device)
        self.model.eval()

    def compute_embedding(self, text):
        """
        Compute SPECTER2 embedding for a given text
        Args:
            text (str): Input text/abstract to embed
        Returns:
            numpy.ndarray: Embedding vector
        """
        if isinstance(text, list):
            text = " ".join(text)
        text = text.strip()
        
        # Handle empty text
        if not text:
            return np.zeros(self.model.config.hidden_size)

        # Tokenize and move to device
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            
        # Move back to CPU and convert to numpy
        embedding = embedding.cpu().numpy()
        return embedding