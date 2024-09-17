import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
from catboost import CatBoostClassifier

from data import Data
from models.base_model import BaseModel
from util import DATA_DIR


class CatboostModel(BaseModel):
    def __init__(self) -> None:
        self.model = None
        # self.paper_features_pipeline = ColumnTransformer([
        #     ('ss_referenceCount', 'passthrough', ['ss_referenceCount']),
        #     ("counts", FunctionTransformer(lambda df: df.map(len)), ["ss_authors", "ss_cited_authors", "arxiv_versions"])
        # ])
        
    def samples_to_dataframe(self, samples: list):
        new_samples = []
        for sample in tqdm(samples, "Converting samples to dataframe"):
            new_sample = {key: sample[key] for key in ["title", "referenceCount", "categories", "label"]}
            new_sample["author_num_papers"] = len(sample["author"]["papers"])
            new_sample["author_fieldsOfStudy"] = []
            new_sample["author_s2FieldsOfStudy"] = []
            for p in sample["author"]["papers"]:
                for key in ["fieldsOfStudy", "s2FieldsOfStudy"]:
                    if p[key] is not None:
                        new_sample[f"author_{key}"] += p[key]
            new_sample["is_cited"] = int(sample["author"]["id"]) in sample["cited_authors"]
            new_samples.append(new_sample)
        df = pd.DataFrame.from_records(new_samples)
        return df

    def process_fold(
        self,
        data: Data,
        fold: str
    ):
        samples = data.get_fold(fold)
        df = self.samples_to_dataframe(samples)
        X = df[["referenceCount", "author_num_papers", "is_cited"]]
        y = df["label"]
        return X, y

    def fit(
        self,
        data: Data
    ):
        X_train, y_train = self.process_fold(data, "train")
        X_val, y_val = self.process_fold(data, "validation")
        self.model = CatBoostClassifier()
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, use_best_model=True)

    def predict_proba(self, data: Data, fold: str):
        assert self.model is not None
        X, _ = self.process_fold(data, fold)
        return self.model.predict_proba(X)[:, 1]

    def save(
        self,
        path: str
    ):
        p = os.path.join(path, "catboost")
        print(f"Saving Catboost model to {p}")
        os.makedirs(p, exist_ok=True)
        self.model.save_model(os.path.join(p, "model.cbm"), format="cbm")

    def load(
        self,
        path: str
    ):
        assert self.model is None
        p = os.path.join(path, "catboost")
        print(f"Loading Catboost model from {p}")
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(p, "model.cbm"), format="cbm")
        

