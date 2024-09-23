import os
import pandas as pd
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
        self.save_(path)

    def load(
        self,
        path: str,
        model: str,
        version: str
    ):
        path = os.path.join(path, f"{model}.{version}")
        print(f"Loading {model} model from {path}")
        self.load_(path)

    def fit(
        self,
        data: Data
    ):
        raise NotImplementedError("fit for base_model must be overloaded")

    def predict_proba(self, data: Data, fold: str):
        raise NotImplementedError("predict_proba for base_model must be overloaded")

    def save_(
        self,
        path: str
    ):
        raise NotImplementedError("save_ for base_model must be overloaded")

    def load_(
        self,
        path: str
    ):
        raise NotImplementedError("load_ for base_model must be overloaded")
