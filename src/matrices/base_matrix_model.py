import os
from models.base_model import BaseModel

class BaseMatrixModel(BaseModel):
    """
    Base class for matrix models. This class is not meant to be used directly, but rather as a base class for other matrix models.
    """
    def __init__(self, config: dict):
        pass

    def load_matrix(self, matrix_path: str) -> None:
        """
        Load the citation matrix from the given path.
        """
        raise NotImplementedError("load_matrix for base_matrix_model must be overloaded")