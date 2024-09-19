import os
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from catboost import CatBoostClassifier

from data import Data
from models.base_model import BaseModel
from util import DATA_DIR


def passthrough_func(x):
    return x


class CatboostModel(BaseModel):
    def __init__(self) -> None:
        self.model = None
        self.feature_processing_pipeline = None
        
    def load_fold(
        self,
        data: Data,
        fold: str
    ):
        samples = data.get_fold(fold)
        new_samples = []
        for sample in tqdm(samples, "Converting samples to dataframe"):
            new_sample = {key: sample[key] for key in ["title", "referenceCount", "categories", "label"]}
            new_sample["author_num_papers"] = len(sample["author"]["papers"])
            new_sample["author_fieldsOfStudy"] = []
            new_sample["author_s2FieldsOfStudy"] = []
            new_sample["author_title"] = ""
            for p in sample["author"]["papers"]:
                for key in ["fieldsOfStudy", "s2FieldsOfStudy"]:
                    if p[key] is not None:
                        new_sample[f"author_{key}"] += p[key]
                if p["title"] is not None:
                    new_sample["author_title"] += " " + p["title"]
            new_sample["is_cited"] = int(sample["author"]["id"]) in sample["cited_authors"]
            new_samples.append(new_sample)
        df = pd.DataFrame.from_records(new_samples)
        X = df[[col for col in df.columns if col != "label"]]
        y = df["label"]
        return X, y

    def fit(
        self,
        data: Data
    ):
        X_train, y_train = self.load_fold(data, "train")
        self.feature_processing_pipeline = ColumnTransformer([
            ('passthrough', 'passthrough', ["referenceCount", "author_num_papers", "is_cited"]),
            ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
            ("title", CountVectorizer(), "title"),
            ("author_fieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_fieldsOfStudy"),
            ("author_s2FieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_s2FieldsOfStudy"),
        ])
        X_train = self.feature_processing_pipeline.fit_transform(X_train)
        print("Training data size:", X_train.shape, " type:", type(X_train))
        X_val, y_val = self.load_fold(data, "validation")
        X_val = self.feature_processing_pipeline.transform(X_val)
        self.model = CatBoostClassifier()
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, use_best_model=True)

    def predict_proba(self, data: Data, fold: str):
        assert self.model is not None
        X, _ = self.load_fold(data, fold)
        X = self.feature_processing_pipeline.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def save(
        self,
        path: str
    ):
        p = os.path.join(path, "catboost")
        print(f"Saving Catboost model to {p}")
        os.makedirs(p, exist_ok=True)
        self.model.save_model(os.path.join(p, "model.cbm"), format="cbm")
        with open(os.path.join(p, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.feature_processing_pipeline, f, protocol=5)        

    def load(
        self,
        path: str
    ):
        assert self.model is None
        p = os.path.join(path, "catboost")
        print(f"Loading Catboost model from {p}")
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(p, "model.cbm"), format="cbm")
        with open(os.path.join(p, "pipeline.pkl"), "rb") as f:
            self.feature_processing_pipeline = joblib.load(f)

