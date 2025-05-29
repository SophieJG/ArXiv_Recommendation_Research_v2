import os
from util import model_version_path


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
        path = model_version_path(path, model, version)
        print(f"Saving {model} embedder to {path}")
        os.makedirs(path, exist_ok=True)
        self._save(path)

    def load(
        self,
        path: str,
        model: str,
        version: str
    ):
        path = model_version_path(path, model, version)
        print(f"Loading {model} embedder from {path}")
        self._load(path)

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

    def _save(
        self,
        path: str
    ):
        raise NotImplementedError("save_ for BaseEmbedder must be overloaded")

    def _load(
        self,
        path: str
    ):
        raise NotImplementedError("load_ for BaseEmbedder must be overloaded")
