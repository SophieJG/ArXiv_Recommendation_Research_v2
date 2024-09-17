import json
import os

import pandas as pd
from tqdm import tqdm
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

    def parse_fold(self, fold: str):
        assert fold in ["train", "validation", "test"]
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test
        }[fold]
    
    def get_fold(self, fold: str):
        fold = self.parse_fold(fold)
        samples = []
        for _, row in tqdm(fold.iterrows(), total=len(fold), desc="Generating samples"):
            paper_id = row["paper"]
            author_id = str(row["author"])
            kaggle_paper_data = self.kaggle_data.loc[paper_id]
            paper_year = kaggle_paper_data["year_updated"]
            samples.append({
                **{key: value for key, value in self.papers[paper_id].items() if key not in ["year", "citing_authors"]},
                "title": kaggle_paper_data["title"],
                "categories": list(kaggle_paper_data["categories"]),
                "author": {
                    "id": author_id,
                    "papers": [
                        p for p in self.authors[author_id]["papers"] if p["year"] < paper_year  # Take only papers that precede the recommended paper publication year
                    ]
                },
                "label": row["label"]
            })
        return samples
