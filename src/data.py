import json
import os

import pandas as pd
from util import DATA_DIR, KAGGLE_DATA_PATH, AUTHORS_PATH, PAPERS_PATH

class Data:
    def __init__(self) -> None:
        print("Loading Arxiv Kaggle data")
        with open(KAGGLE_DATA_PATH) as f:
            self.kaggle_data = pd.read_parquet(KAGGLE_DATA_PATH)
        print("Loading papers data")
        with open(PAPERS_PATH) as f:
            self.papers = json.load(f)
        print("Loading authors data")
        with open(AUTHORS_PATH) as f:
            self.authors = json.load(f)
        print("loading folds")
        self.train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        self.validation = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
        self.test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        print(self.train)