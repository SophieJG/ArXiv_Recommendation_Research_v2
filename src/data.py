import json
import os

import pandas as pd
from util import DATA_DIR, KAGGLE_DATA_PATH, AUTHORS_PATH, PAPERS_PATH

class Data:
    def __init__(self) -> None:
        print("Loading Arxiv Kaggle data")
        with open(KAGGLE_DATA_PATH) as f:
            self.kaggle_data = pd.read_parquet(KAGGLE_DATA_PATH)
        self.kaggle_data["index"] = [f"ARXIV:{id}" for id in self.kaggle_data["id"]]
        self.kaggle_data = self.kaggle_data.set_index("index", drop=True)
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

    def get_papers(
            self,
            fold: str
    ):
        # Refactors the paper raw data into vector of the relevant papers for the fold
        assert fold in ["train", "validation", "test"]
        fold = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test
        }[fold]

        paper_list = []
        paper_ids = set(fold["paper"])
        for paper_id, paper_data in self.papers.items():
            if paper_id in paper_ids:
                paper_list.append({
                    # No other fields are allowed to take from Semantic Scholar - won't be available for nightly email
                    "id": paper_id,
                    "referenceCount" : paper_data["referenceCount"],
                    "authors" : paper_data["authors"],
                    "cited_authors" : paper_data["cited_authors"]
                })
        df = pd.DataFrame.from_records(paper_list).set_index("id", drop=True)
        # join with kaggle data
        df = df.add_prefix('ss_').join(self.kaggle_data.add_prefix("arxiv_"))
        return df
