import pandas as pd
from data import Data


class BaseModel:
    def fit(
        self,
        data: Data
    ):
        raise NotImplementedError("fit for base_model should not be called")

    def predict_proba(self, data: Data, fold: str):
        raise NotImplementedError("predict_proba for base_model should not be called")

    def save(
        self,
        path: str
    ):
        raise NotImplementedError("save for base_model should not be called")

    def load(
        self,
        path: str
    ):
        raise NotImplementedError("load for base_model should not be called")
