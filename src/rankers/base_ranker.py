class BaseRanker:
    def __init__(
        self,
        params: dict
    ):
        pass

    def rank(
        self
    ):
        raise NotImplementedError("rank for BaseRanker must be overloaded")
        pass