import os


class BaseEmbedder:
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
        print(f"Saving {model} embedder to {path}")
        os.makedirs(path, exist_ok=True)
        self.save_(path)

    def load(
        self,
        path: str,
        model: str,
        version: str
    ):
        path = os.path.join(path, f"{model}.{version}")
        print(f"Loading {model} embedder from {path}")
        self.load_(path)

    def fit(
        self,
        papers: list
    ):
        raise NotImplementedError("fit for BaseEmbedder must be overloaded")

    def embed(
        self,
        papers: list
    ):
        raise NotImplementedError("embed for BaseEmbedder must be overloaded")

    def save_(
        self,
        path: str
    ):
        raise NotImplementedError("save_ for BaseEmbedder must be overloaded")

    def load_(
        self,
        path: str
    ):
        raise NotImplementedError("load_ for BaseEmbedder must be overloaded")
