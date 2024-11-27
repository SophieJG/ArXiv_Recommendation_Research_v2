import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from catboost import CatBoostClassifier

from data import Data
from models.base_model import BaseModel
from util import passthrough_func


class CatboostModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = None
        self.feature_processing_pipeline = None

    @staticmethod
    def _process_author(new_sample: dict, sample: dict):
        new_sample["author_num_papers"] = len(sample["author"]["papers"])
        # Go over the author papers and collect the fiefieldsOfStudy and s2fieldsofstudy into two lists
        # Additionally, the papers' titles are concatenated into a long string
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

    def _samples_to_dataframe(
        self,
        samples: list
    ):
        """
Loads a fold and converts it to pandas dataframe. Some non-trivial data processing is required to convert the paper and author
dictionaries to rows in a dataframe
"""
        new_samples = []
        for sample in tqdm(samples, "Catboost: samples -> dataframe"):
            # Copy fields from the data dictionaries 
            new_sample = {"label": sample["label"]}
            self._process_paper(new_sample, sample)
            self._process_author(new_sample, sample)
            new_samples.append(new_sample)
        df = pd.DataFrame.from_records(new_samples)
        X = df[[col for col in df.columns if col != "label"]]
        y = df["label"]
        return X, y

    def fit(
        self,
        train_samples: list,
        validation_samples: list
    ):
        """
1. Fit the preprocessing pipeline on the training data
2. Convert the training and validation data using the preprocessing pipeline
3. Train the model on the processed training and validation data
"""
        X_train, y_train = self._samples_to_dataframe(train_samples)
        self.feature_processing_pipeline = ColumnTransformer([
            ('passthrough', 'passthrough', ["author_num_papers"]),
            ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
            ("title", CountVectorizer(), "title"),
            ("author_s2fieldsofstudy", CountVectorizer(analyzer=passthrough_func), "author_s2fieldsofstudy"),
        ])
        X_train = self.feature_processing_pipeline.fit_transform(X_train)
        print("Training data size:", X_train.shape, " type:", type(X_train))
        X_val, y_val = self._samples_to_dataframe(validation_samples)
        X_val = self.feature_processing_pipeline.transform(X_val)
        self.model = CatBoostClassifier().fit(
            X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, use_best_model=True
            )

    def _predict_proba(self, X: pd.DataFrame):
        X = self.feature_processing_pipeline.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, samples: list):
        """
Run inference on a list of samples
"""
        assert self.model is not None
        X, _ = self._samples_to_dataframe(samples)
        return self._predict_proba(X)

    def predict_proba_ranking(self, papers: list, authors: list):
        """
Run inference on the cartesian product between all papers and all authors
"""
        assert self.model is not None
        papers_df = pd.DataFrame.from_records([self._process_paper({}, p) for p in papers])
        authors_df = pd.DataFrame.from_records([self._process_author({}, a) for a in authors])
        X = pd.merge(authors_df, papers_df, how='cross')
        utility = np.array(self._predict_proba(X)).reshape((len(authors), len(papers)))
        return utility

    def _save(
        self,
        path: str
    ):
        assert self.model is not None
        self.model.save_model(os.path.join(path, "model.cbm"), format="cbm")
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.feature_processing_pipeline, f, protocol=5)        

    def _load(
        self,
        path: str
    ):
        assert self.model is None
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(path, "model.cbm"), format="cbm")
        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            self.feature_processing_pipeline = joblib.load(f)


