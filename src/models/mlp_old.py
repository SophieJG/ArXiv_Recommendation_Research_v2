import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import Data
from models.base_model import BaseModel
from util import passthrough_func
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from data import Data
from models.base_model import BaseModel
from models.spector_embed import Specter2EmbeddingsTransformer
from util import *

def passthrough_func(x):
    return x

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class MLPClassifier(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = None
        self.feature_processing_pipeline = None
        self.input_size = None  # To be dynamically set during training
        self.params = params

        # Define the preprocessing pipeline
        self.feature_processing_pipeline = ColumnTransformer([
            # ('passthrough', 'passthrough', ["referenceCount", "author_num_papers", "is_cited"]),
            # ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
            # ("title", CountVectorizer(), "title"),
            # ("author_fieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_fieldsOfStudy"),
            # ("author_s2FieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_s2FieldsOfStudy"),
            ("abstract_embedding", Specter2EmbeddingsTransformer(), "abstract"),
            ("author_paper_embedding", Specter2EmbeddingsTransformer(), "author_paper_abstract")
        ])


    def load_fold(self, data: Data, fold: str):
        """
        Loads a fold and converts it to a pandas dataframe.
        """
        samples = data.get_fold(fold)
        new_samples = []
        top_k = 1
        for sample in tqdm(samples, "Converting samples to dataframe"):
            new_sample = {key: sample[key] for key in ["label"]}
            # new_sample = {key: sample[key] for key in ["title", "categories", "label"]}
            new_sample["abstract"] = [sample["abstract"]]

            # new_sample["author_num_papers"] = len(sample["author"]["papers"])
            # new_sample["author_fieldsOfStudy"] = []
            # new_sample["author_s2FieldsOfStudy"] = []
            # new_sample["author_title"] = ""
            new_sample["author_paper_abstract"] = []
            for p in sample["author"]["papers"]:
                if len(new_sample["author_paper_abstract"])<top_k:
                    if p["abstract"]:
                        new_sample["author_paper_abstract"].append(p["abstract"])
                # for key in ["fieldsOfStudy", "s2FieldsOfStudy"]: #s2 field of study is always none empty
                #     if p[key] is not None:
                #         new_sample[f"author_{key}"] += p[key]
                # if p["title"] is not None:
                #     new_sample["author_title"] += " " + p["title"]
            # print("*****length: ", len(new_sample["author_paper_abstract"]))
            # new_sample["is_cited"] = int(sample["author"]["id"]) in sample["cited_authors"]
            new_samples.append(new_sample)
        print("*****length of new_samples:", len(new_samples))
        # print(new_samples)
        df = pd.DataFrame.from_records(new_samples)
        X = df[[col for col in df.columns if col != "label"]]
        y = df["label"]
        return X, y
    
    def get_data_transform(self, data: pd.DataFrame, fold, fit: bool):
        # abstract_transformer = Specter2EmbeddingsTransformer()

        # # Define the preprocessing pipeline
        # self.feature_processing_pipeline = ColumnTransformer([
        #     # ('passthrough', 'passthrough', ["referenceCount", "author_num_papers", "is_cited"]),
        #     # ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
        #     # ("title", CountVectorizer(), "title"),
        #     # ("author_fieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_fieldsOfStudy"),
        #     # ("author_s2FieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_s2FieldsOfStudy"),
        #     ("abstract_embedding", abstract_transformer, "abstract"),
        #     ("author_paper_embedding", abstract_transformer, "author_paper_abstract")
        # ])
        print("get_data params", fold, fit)
        embed_file_path = os.path.join(data_dir(self.params), f"{fold}_embed.csv")
        X, y = self.load_fold(data, fold)
        if os.path.exists(embed_file_path):
            print("Embedding already exist, reading existing file")
            X_embed = pd.read_csv(embed_file_path)
            if X_embed.shape[1] == X.shape[1]:
                return X_embed.values, y
            print("Existing embedding file shape does not match, re-embedding")
        print("Transforming dataframe")
        if fit:
            X = self.feature_processing_pipeline.fit_transform(X)
        else: 
            X = self.feature_processing_pipeline.transform(X)
        X_df = pd.DataFrame(X)
        X_df.to_csv(embed_file_path, index=False)
        print(f"Transformation Done, X_train saved to {embed_file_path}")
        return X, y

    def fit(self, data: Data):
        """
        1. Fit the preprocessing pipeline on the training data.
        2. Train the MLP model.
        """
        X_train, y_train = self.get_data_transform(data, "train", True)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        print("***** X train shape:", X_train.shape)
        # Set input size based on transformed data
        self.input_size = X_train.shape[1]

        # Initialize the MLP model
        self.model = MLPModel(input_size=self.input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        epochs = 50
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict_proba(self, data: Data, fold: str):
        """
        Run inference on a fold and return probabilities.
        """
        assert self.model is not None
        X, _ = self.get_data_transform(data, fold, False)

        # X = self.feature_processing_pipeline.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X)
        return outputs.squeeze().numpy()

    def save_(self, path: str):
        """
        Save only the trained model weights and necessary model info.
        """
        assert self.model is not None
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,  # Save input_size
        }, os.path.join(path, "model.pth"))
            # with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            #     joblib.dump(self.feature_processing_pipeline, f, protocol=5)

    def load_(self, path: str):
        """
        Load the trained model and the input size.
        """
        assert self.model is None
        # Load the saved model state dict and input_size
        checkpoint = torch.load(os.path.join(path, "model.pth"))
        
        # Retrieve input size
        self.input_size = checkpoint['input_size']
        
        # Initialize the MLP model with the correct input size
        self.model = MLPModel(input_size=self.input_size)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        