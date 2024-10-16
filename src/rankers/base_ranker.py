import pandas as pd


class BaseRanker:
    def __init__(
        self,
        params: dict
    ):
        pass

    def rank(
        self,
        proba: pd.DataFrame,
        paper_embeddings: dict
    ):
        raise NotImplementedError("rank for BaseRanker must be overloaded")
