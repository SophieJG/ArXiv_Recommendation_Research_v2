import pandas as pd


class BaseRanker:
    def __init__(
        self,
        items_to_rank: int,
        params: dict
    ):
        pass

    def rank(
        self,
        proba: pd.DataFrame,
        paper_embeddings: dict
    ):
        raise NotImplementedError("rank for BaseRanker must be overloaded")
