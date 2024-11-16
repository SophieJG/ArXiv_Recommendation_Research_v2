import json
import os

import pandas as pd
from tqdm import tqdm
from util import data_dir, kaggle_data_path, authors_path, papers_path


class Data:
    """
Main class used to store data for training and evaluation purpose. See the readme for more details.
"""
    def __init__(self, config: dict) -> None:
        print("Data: loading Arxiv Kaggle data")
        self.kaggle_data = pd.read_parquet(kaggle_data_path(config))
        self.kaggle_data["index"] = [str(id) for id in self.kaggle_data["id"]]
        self.kaggle_data = self.kaggle_data.set_index("index", drop=True)
        print("Data: loading papers data")
        self.papers = json.load(open(papers_path(config)))
        print("Data: loading authors data")
        self.authors = json.load(open(authors_path(config)))
        print("Data: loading folds")
        self.train = pd.read_csv(os.path.join(data_dir(config), "train.csv"))
        self.validation = pd.read_csv(os.path.join(data_dir(config), "validation.csv"))
        self.test = pd.read_csv(os.path.join(data_dir(config), "test.csv"))
        ranking_fold_path = os.path.join(data_dir(config), "ranking.csv")
        self.ranking = pd.read_csv(ranking_fold_path) if os.path.exists(ranking_fold_path) else None

    def parse_fold(self, fold: str):
        """
Convert the fold string to the fold data
"""
        if "ranking" in fold:
            _, idx0, idx1 = fold.split("_")
            return self.ranking[int(idx0): int(idx1)]
        assert fold in ["train", "validation", "test"]
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test
        }[fold]
    
    def get_fold(self, fold: str):
        """
This function returns the unstrucuted fold data as a list of samples. Each sample includes:
(1) paper info as a dictionary
(2) author info as dictionary
(3) a boolean label

- In order to guarantee consistency, the author info is "shifted" in time to the year the paper was published. In practice,
that implies removing all publications by the author that proceed (are after) the paper.
"""
        fold = self.parse_fold(fold)
        samples = []
        for _, row in tqdm(fold.iterrows(), total=len(fold), desc="Generating samples"):
            paper_id = str(row["paper"])
            author_id = str(row["author"])
            kaggle_paper_data = self.kaggle_data.loc[str(self.papers[paper_id]["arxiv_id"])]
            paper_year = kaggle_paper_data["year_updated"]
            author_papers = []
            for id in self.authors[author_id]:
                # Filter the author's published papers by year
                if str(id) not in self.papers:
                    continue
                paper = self.papers[str(id)]
                if paper["year"] < paper_year:
                    author_papers.append(paper)
            samples.append({
                # copy all fields from Semantic Scholar except `year` and `citing_authors`
                **{key: value for key, value in self.papers[paper_id].items() if key not in ["year", "citing_authors"]},
                # title and categories are taken from the kaggle dataset
                "title": kaggle_paper_data["title"],
                "categories": list(kaggle_paper_data["categories"]),
                "author": {
                    "id": author_id,
                    "papers": author_papers
                },
                "label": row["label"]
            })
        return samples
