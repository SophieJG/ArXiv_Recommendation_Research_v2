import json
import os
import numpy as np
import pandas as pd


from data import Data
from models.base_model import BaseModel
from util import DATA_DIR


class CatboostModel(BaseModel):
    def __init__(self) -> None:
        pass
        
    def train(
        self,
        data: Data
    ):
        train_papers = data.get_papers("train")
        self.paper_features(train_papers)
        
    @staticmethod
    def paper_features(papers: pd.DataFrame):
        print(papers.iloc[0])

