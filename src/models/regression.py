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
        print("Calculating embeddings for SPECTER2 model")
        # Convert X to a list if needed
        if isinstance(X, np.ndarray):
            X = X.flatten().tolist()
        elif hasattr(X, "tolist"):
            X = X.tolist()
        
        filtered_embeddings = []
        valid_indices = []  
        
        for index, row in X.iterrows():
            abstract_text = row['abstract']
            author_paper_abstract_text = row['author_paper_abstract']
            
            # Step 1: Filter out rows where 'author_paper_abstract' is empty
            if not author_paper_abstract_text:
                continue
            
            # Step 2: Concatenate list elements into a single string if needed
            if isinstance(abstract_text, list):
                abstract_text = " ".join(abstract_text)
            if isinstance(author_paper_abstract_text, list):
                author_paper_abstract_text = " ".join(author_paper_abstract_text)
            
            # Step 3: Get embeddings for both fields
            abstract_inputs = self.tokenizer(abstract_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            author_inputs = self.tokenizer(author_paper_abstract_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            
            with torch.no_grad():
                abstract_outputs = self.model(**abstract_inputs)
                author_outputs = self.model(**author_inputs)
            
            # Step 4: Calculate mean embeddings and concatenate them
            abstract_embedding = torch.mean(abstract_outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
            author_embedding = torch.mean(author_outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
            concatenated_embedding = np.concatenate((abstract_embedding, author_embedding))
            
            filtered_embeddings.append(concatenated_embedding)

            valid_indices.append(index)
        
        print("Embedding calculation complete.")
        # Return as a 2D array where each row is a concatenated embedding
        return np.array(filtered_embeddings), valid_indices

class RegressionModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = LogisticRegression(max_iter=params.get("max_iter", 1000))
        self.feature_processing_pipeline = Specter2EmbeddingsTransformer()

    def load_fold(self, data: Data, fold: str):
        samples = data.get_fold(fold)
        new_samples = []
        for sample in tqdm(samples, "Converting samples to dataframe"):
            new_sample = {key: sample[key] for key in ["label"]}
            new_sample["abstract"] = [sample["abstract"]]
            new_sample["author_paper_abstract"] = []
            for p in sample["author"]["papers"]:
                if len(new_sample["author_paper_abstract"])<1:
                    if p["abstract"]:
                        new_sample["author_paper_abstract"].append(p["abstract"])
            new_samples.append(new_sample)
        print("*****length of new_samples:", len(new_samples))
        df = pd.DataFrame.from_records(new_samples)
        # print(df.head())
        X = df[[col for col in df.columns if col != "label"]]
        y = df["label"]
        return X, y

    def concatenate_entries(self, X_train):
        concatenated_X = []
        for entry in X_train:
            print(len(entry))
            if isinstance(entry, list):
                entry = np.array(entry)
            if entry.ndim == 1:
                entry = np.expand_dims(entry, axis=1)
            concatenated_X.append(np.concatenate(entry, axis=1))
        return np.vstack(concatenated_X)

    def fit(self, data: Data):
        X_train, y_train = self.load_fold(data, "train")
        X_train, valid_indices = self.feature_processing_pipeline.fit_transform(X_train)
        # Filter y_train to keep only rows with valid embeddings
        y_train = y_train.iloc[valid_indices].reset_index(drop=True)

        print(f"length of X_train input: {len(X_train[0])}")
        # X_train = self.concatenate_entries(X_train)
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict_proba(self, data: Data, fold: str):
        X, y = self.load_fold(data, fold)
        X, valid_indices = self.feature_processing_pipeline.transform(X)
        y = y.iloc[valid_indices].reset_index(drop=True)
        # print("X", type(X), X.shape)
        # print("y", type(y), y.shape)
        # print("y pred", type(y_pred), y_pred.shape)
        return self.model.predict_proba(X)[:, 1] , y

    def save_(self, path: str):
        joblib.dump(self.model, os.path.join(path, "model.pkl"))
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.feature_processing_pipeline, f, protocol=5)

    def load_(self, path: str):
        self.model = joblib.load(os.path.join(path, "model.pkl"))
        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            self.feature_processing_pipeline = joblib.load(f)