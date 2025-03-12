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
from datetime import datetime
import json

from data import Data
from models.base_model import BaseModel
from util import passthrough_func


class MLPNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            linear = nn.Linear(prev_size, hidden_size)
            # Initialize weights properly
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            
            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_size = hidden_size
            
        final_layer = nn.Linear(prev_size, 1)
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        
        layers.append(final_layer)
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class MLPModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = None
        self.feature_processing_pipeline = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = None
        
        # Default MLP parameters if not specified
        self.mlp_params = {
            'hidden_sizes': (64, 32),
            'max_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 10,
            **params
        }

    @staticmethod
    def _process_author(new_sample: dict, sample: dict):
        # new_sample["author_num_papers"] = len(sample["author"]["papers"])
        # new_sample["author_fieldsOfStudy"] = []
        # new_sample["author_s2fieldsofstudy"] = []
        # new_sample["author_title"] = ""
        # for p in sample["author"]["papers"]:
        #     for key in ["s2fieldsofstudy"]:
        #         if p[key] is not None:
        #             new_sample[f"author_{key}"] += p[key]
        #     if p["title"] is not None:
        #         new_sample["author_title"] += " " + p["title"]
        new_sample["author_embedding"] = sample["author"]["embedding"]
        return new_sample

    @staticmethod
    def _process_paper(new_sample: dict, sample: dict):
        # for key in ["title", "categories"]:
        #     new_sample[key] = sample[key]
        new_sample["paper_embedding"] = sample["embedding"]
        return new_sample

    def _samples_to_dataframe(self, samples: list):
        new_samples = []
        labels = []

        for sample in tqdm(samples, desc="Processing Samples"):
            # Store embeddings separately
            # paper_embeddings.append(sample["embedding"])
            # author_embeddings.append(sample["author"]["embedding"])
            labels.append(sample["label"])

            # Create feature dict excluding embeddings
            new_sample = {}
            self._process_paper(new_sample, sample)
            self._process_author(new_sample, sample)
            new_samples.append(new_sample)

        # Convert non-embedding features into a Pandas DataFrame
        df = pd.DataFrame.from_records(new_samples)

        # # Convert embeddings and labels to PyTorch tensors
        # paper_embeddings = torch.FloatTensor(paper_embeddings)
        # author_embeddings = torch.FloatTensor(author_embeddings)
        
        # embeddings = torch.cat((paper_embeddings, author_embeddings), dim=1).to(self.device)

        return df, labels

    def _process_dataframe_embeddings(self, df: pd.DataFrame):
        paper_embeddings = np.vstack(df["paper_embedding"].values)
        author_embeddings = np.vstack(df["author_embedding"].values)
        embeddings = np.hstack((paper_embeddings, author_embeddings))
        return embeddings

    def fit(self, train_samples: list, validation_samples: list):
        X_train, y_train = self._samples_to_dataframe(train_samples)
        X_train = self._process_dataframe_embeddings(X_train)
        # print("X_train:", X_train.head())
        X_train = torch.FloatTensor(X_train).to(self.device)

        print("X_train mean:", torch.mean(X_train))
        print("X_train std:", torch.std(X_train))

        # Create feature processing pipeline
        self.feature_processing_pipeline = ColumnTransformer([
            ('passthrough', 'passthrough', ["author_num_papers"]),
            ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
            ("title", CountVectorizer(), "title"),
            ("author_s2fieldsofstudy", CountVectorizer(analyzer=passthrough_func), "author_s2fieldsofstudy"),
        ])
        
        # # Transform training data
        # X_train_sparse = self.feature_processing_pipeline.fit_transform(X_train)
        # X_train_sparse = self.scaler.fit_transform(X_train.toarray())
        
        # # Transform validation data
        X_val, y_val = self._samples_to_dataframe(validation_samples)
        X_val = self._process_dataframe_embeddings(X_val)
        X_val = torch.FloatTensor(X_val).to(self.device)
        # X_val_sparse = self.feature_processing_pipeline.transform(X_val)
        # X_val_sparse = self.scaler.transform(X_val.toarray())
        
        # Convert to PyTorch tensors
        # X_train_sparse = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        # X_val_sparse = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        # Train with embddings only
        # X_train = train_embeddings
        # X_val = val_embeddings

        print("X_train:", X_train)
        print("X_val:", X_val)
        print("y_train:", y_train)
        print("y_val:", y_val)

        
        print("Training data size:", X_train.shape)
        
        # Initialize model
        self.input_size = X_train.shape[1]
        self.model = MLPNetwork(self.input_size, self.mlp_params['hidden_sizes']).to(self.device)
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mlp_params['learning_rate'])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.mlp_params['max_epochs']):
            self.model.train()
            total_train_loss = 0
            num_batches = 0
            
            # Training phase
            for i in range(0, len(X_train), self.mlp_params['batch_size']):
                batch_X = X_train[i:i + self.mlp_params['batch_size']]
                batch_y = y_train[i:i + self.mlp_params['batch_size']]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Add gradient checking
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Gradient stats for {name}:")
                #         print(f"Mean: {param.grad.mean():.6f}")
                #         print(f"Std: {param.grad.std():.6f}")
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
                
                # Inside training loop, after forward pass
                # print("Predictions distribution:", 
                #       torch.histc(outputs, bins=2, min=0, max=1))
            
            # Calculate average training loss
            avg_train_loss = total_train_loss / num_batches
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val)
                
            # Print progress
            print(f'Epoch [{epoch+1}/{self.mlp_params["max_epochs"]}] '
                  f'Training Loss: {avg_train_loss:.4f}, '
                  f'Validation Loss: {val_loss:.4f}')
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.mlp_params['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def _predict_proba(self, X: pd.DataFrame):
        # X = self.feature_processing_pipeline.transform(X)
        # X = self.scaler.transform(X.toarray())
        X = self._process_dataframe_embeddings(X)
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
        os.makedirs(path, exist_ok=True)
    
        # Save PyTorch model with metadata
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.mlp_params,  # Save model configuration
            'input_size': self.input_size,
            'version': '1.0',  # Version tracking
        }
        torch.save(checkpoint, os.path.join(path, "model.pt"))
        
        # Save preprocessing components
        try:
            joblib.dump(self.feature_processing_pipeline, 
                    os.path.join(path, "pipeline.pkl"), protocol=5)
            joblib.dump(self.scaler, 
                    os.path.join(path, "scaler.pkl"), protocol=5)
        except Exception as e:
            print(f"Error saving preprocessing components: {e}")
            
        # Optionally save a metadata file
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "date_saved": str(datetime.now()),
                "model_type": "MLP",
                # Any other metadata
            }, f)
        # torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        # with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
        #     joblib.dump(self.feature_processing_pipeline, f, protocol=5)
        # with open(os.path.join(path, "scaler.pkl"), "wb") as f:
        #     joblib.dump(self.scaler, f, protocol=5)

    def _load(self, path: str):
        assert self.model is None

        try:
            # Load PyTorch model and metadata
            checkpoint = torch.load(os.path.join(path, "model.pt"))
            
            # Initialize model with saved config if needed
            input_size = checkpoint.get('input_size')
            model_config = checkpoint.get('model_config')
            if self.model is None and input_size is not None:
                self.model = MLPNetwork(input_size, model_config['hidden_sizes']).to(self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode
            
            # Load preprocessing components
            try:
                self.feature_processing_pipeline = joblib.load(
                    os.path.join(path, "pipeline.pkl"))
                self.scaler = joblib.load(
                    os.path.join(path, "scaler.pkl"))
            except Exception as e:
                print(f"Error loading preprocessing components: {e}")
            
            # Optionally load and verify metadata
            try:
                with open(os.path.join(path, "metadata.json"), "r") as f:
                    self.metadata = json.load(f)
                    print(f"Model saved on: {self.metadata['date_saved']}")
                    print(f"Model type: {self.metadata['model_type']}")
            except FileNotFoundError:
                print("No metadata file found")
                
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            self.feature_processing_pipeline = joblib.load(f)
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            self.scaler = joblib.load(f)
            
        # Need to initialize model with correct input size before loading weights
        # dummy_X, paper_embeddings, author_embeddings, _ = self._samples_to_dataframe([])  # Empty list to get feature size
        # # dummy_X = self.feature_processing_pipeline.transform(dummy_X)
        # # input_size = paper_embeddings.shape[0] + author_embeddings.shape
        # input_size = paper_embeddings.shape[0][1] + author_embeddings.shape[0][1]
        # self.model = MLPNetwork(input_size, self.mlp_params['hidden_sizes']).to(self.device)
        # self.model.load_state_dict(torch.load(os.path.join(path, "model.pt")))