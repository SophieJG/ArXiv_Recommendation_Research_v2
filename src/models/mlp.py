import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data import Data
from models.base_model import BaseModel
from util import passthrough_func


class MLPNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class MLPModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = None
        self.feature_processing_pipeline = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default MLP parameters if not specified
        self.mlp_params = {
            'hidden_sizes': (128, 64),
            'max_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 5,
            **params
        }

    @staticmethod
    def _process_author(new_sample: dict, sample: dict):
        new_sample["author_num_papers"] = len(sample["author"]["papers"])
        new_sample["author_fieldsOfStudy"] = []
        new_sample["author_s2fieldsofstudy"] = []
        new_sample["author_title"] = ""
        for p in sample["author"]["papers"]:
            for key in ["s2fieldsofstudy"]:
                if p[key] is not None:
                    new_sample[f"author_{key}"] += p[key]
            if p["title"] is not None:
                new_sample["author_title"] += " " + p["title"]
        return new_sample

    @staticmethod
    def _process_paper(new_sample: dict, sample: dict):
        for key in ["title", "categories"]:
            new_sample[key] = sample[key]
        return new_sample

    def _samples_to_dataframe(self, samples: list):
        new_samples = []
        for sample in tqdm(samples, "MLP: samples -> dataframe"):
            new_sample = {"label": sample["label"]}
            self._process_paper(new_sample, sample)
            self._process_author(new_sample, sample)
            new_samples.append(new_sample)
        df = pd.DataFrame.from_records(new_samples)
        X = df[[col for col in df.columns if col != "label"]]
        y = df["label"]
        return X, y

    def fit(self, train_samples: list, validation_samples: list):
        X_train, y_train = self._samples_to_dataframe(train_samples)
        
        # Create feature processing pipeline
        self.feature_processing_pipeline = ColumnTransformer([
            ('passthrough', 'passthrough', ["author_num_papers"]),
            ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
            ("title", CountVectorizer(), "title"),
            ("author_s2fieldsofstudy", CountVectorizer(analyzer=passthrough_func), "author_s2fieldsofstudy"),
        ])
        
        # Transform training data
        X_train = self.feature_processing_pipeline.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train.toarray())
        
        # Transform validation data
        X_val, y_val = self._samples_to_dataframe(validation_samples)
        X_val = self.feature_processing_pipeline.transform(X_val)
        X_val = self.scaler.transform(X_val.toarray())
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train.values).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val.values).to(self.device)
        
        print("Training data size:", X_train.shape)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = MLPNetwork(input_size, self.mlp_params['hidden_sizes']).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mlp_params['learning_rate'])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.mlp_params['max_epochs']):
            self.model.train()
            for i in range(0, len(X_train), self.mlp_params['batch_size']):
                batch_X = X_train[i:i + self.mlp_params['batch_size']]
                batch_y = y_train[i:i + self.mlp_params['batch_size']]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.mlp_params['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    def _predict_proba(self, X: pd.DataFrame):
        X = self.feature_processing_pipeline.transform(X)
        X = self.scaler.transform(X.toarray())
        X = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X).squeeze().cpu().numpy()
        return probs

    def predict_proba(self, samples: list):
        assert self.model is not None
        X, _ = self._samples_to_dataframe(samples)
        return self._predict_proba(X)

    def predict_proba_ranking(self, papers: list, authors: list):
        assert self.model is not None
        papers_df = pd.DataFrame.from_records([self._process_paper({}, p) for p in papers])
        authors_df = pd.DataFrame.from_records([self._process_author({}, a) for a in authors])
        X = pd.merge(authors_df, papers_df, how='cross')
        utility = np.array(self._predict_proba(X)).reshape((len(authors), len(papers)))
        return utility

    def _save(self, path: str):
        assert self.model is not None
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.feature_processing_pipeline, f, protocol=5)
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            joblib.dump(self.scaler, f, protocol=5)

    def _load(self, path: str):
        assert self.model is None
        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            self.feature_processing_pipeline = joblib.load(f)
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            self.scaler = joblib.load(f)
            
        # Need to initialize model with correct input size before loading weights
        dummy_X, _ = self._samples_to_dataframe([])  # Empty list to get feature size
        dummy_X = self.feature_processing_pipeline.transform(dummy_X)
        input_size = dummy_X.shape[1]
        
        self.model = MLPNetwork(input_size, self.mlp_params['hidden_sizes']).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt")))