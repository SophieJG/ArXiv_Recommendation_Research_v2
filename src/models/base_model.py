import os
from data import Data


class BaseModel:
    def __init__(
        self,
        params: dict
    ):
        pass

    def save(
        self,
        path: str,
        model: str,
        version: str
    ):
        path = os.path.join(path, f"{model}.{version}")
        print(f"Saving {model} model to {path}")
        os.makedirs(path, exist_ok=True)
        self._save(path)

    def load(
        self,
        path: str,
        model: str,
        version: str
    ):
        path = os.path.join(path, f"{model}.{version}")
        print(f"Loading {model} model from {path}")
        self._load(path)

    def fit(
        self,
        train_samples: list,
        validation_samples: list
    ):
        raise NotImplementedError("fit for base_model must be overloaded")

    def predict_proba(self, samples: list):
        """
Run inference on a list of samples
"""
        raise NotImplementedError("predict_proba for base_model must be overloaded")

    def predict_proba_ranking(self, papers: list, authors: list):
        """
Run inference on the cartesian product between all papers and all authors
"""
        raise NotImplementedError("predict_proba_ranking for base_model must be overloaded")

    def _save(
        self,
        path: str
    ):
        raise NotImplementedError("save_ for base_model must be overloaded")

    def _load(
        self,
        path: str
    ):
        raise NotImplementedError("load_ for base_model must be overloaded")
