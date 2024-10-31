import os
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from tqdm import tqdm
from data import Data
from models.base_model import BaseModel

class Specter2EmbeddingsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="allenai/specter_plus_plus", adapter_name="allenai/specter2"):
        # Initialize the SPECTER2 model with an adapter
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.model.load_adapter(adapter_name, set_active=True)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = X.flatten().tolist()
        elif hasattr(X, "tolist"):
            X = X.tolist()

        embeddings = []
        for text in X:
            if isinstance(text, list):
                text = " ".join(text)
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)
        return np.array(embeddings)

class RegressionModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = LogisticRegression(max_iter=params.get("max_iter", 1000))
        self.feature_processing_pipeline = Specter2EmbeddingsTransformer()

    def load_fold(self, data: Data, fold: str):
        samples = data.get_fold(fold)
        new_samples = []
        for sample in tqdm(samples, "Converting samples to dataframe"):
            new_sample = {"abstract": [sample["abstract"]], "label": sample["label"]}
            new_samples.append(new_sample)

        df = pd.DataFrame.from_records(new_samples)
        X = df["abstract"]
        y = df["label"]
        return X, y

    def fit(self, data: Data):
        X_train, y_train = self.load_fold(data, "train")
        X_train = self.feature_processing_pipeline.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict_proba(self, data: Data, fold: str):
        X, _ = self.load_fold(data, fold)
        X = self.feature_processing_pipeline.transform(X)
        return self.model.predict_proba(X)

    def save_(self, path: str):
        joblib.dump(self.model, os.path.join(path, "model.pkl"))
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.feature_processing_pipeline, f, protocol=5)

    def load_(self, path: str):
        self.model = joblib.load(os.path.join(path, "model.pkl"))
        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            self.feature_processing_pipeline = joblib.load(f)
