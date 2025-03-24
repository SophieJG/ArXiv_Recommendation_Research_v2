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
        ranking_fold_path = os.path.join(data_dir(config), "ranking.json")
        self.ranking = json.load(open(ranking_fold_path)) if os.path.exists(ranking_fold_path) else None

    def _process_author(self, new_sample: dict, author_id: str, paper_year: int):
        author_papers = []
        for id in self.authors[author_id]:
            # Filter the author's published papers by year
            if str(id) not in self.papers:
                continue
            paper = self.papers[str(id)]
            if paper["year"] < paper_year:
                author_papers.append(paper)
        new_sample["author"] = {
            "id": author_id,
            "papers": author_papers
        }
        return new_sample

    def _process_paper(self, new_sample: dict, paper_id: str):
        kaggle_paper_data = self.kaggle_data.loc[str(self.papers[paper_id]["arxiv_id"])]
        # copy all fields from Semantic Scholar except `year` and `citing_authors`
        for key, value in self.papers[paper_id].items():
            if key in ["year", "citing_authors"]:
                continue
            new_sample[key] = value
        # title, categories and year are taken from the kaggle dataset
        new_sample["title"] = kaggle_paper_data["title"]
        new_sample["categories"] = list(kaggle_paper_data["categories"])
        new_sample["year"] = kaggle_paper_data["year_updated"]
        return new_sample

    def _parse_fold(self, fold_str: str):
        """
Convert the fold string to the fold data
"""
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test
        }[fold_str]
    
    def get_fold(self, fold_str: str):
        """
This function returns the unstrucuted fold data as a list of samples. Each sample includes:
(1) paper info as a dictionary
(2) author info as dictionary
(3) a boolean label

- In order to guarantee consistency, the author info is "shifted" in time to the year the paper was published. In practice,
that implies removing all publications by the author that proceed (are after) the paper.
"""
        fold = self._parse_fold(fold_str)
        samples = []
        for row in tqdm(fold.itertuples(), total=len(fold), desc=f"Data: loading ({fold_str})"):
            new_sample = {
                "label": row.label
            }
            paper_id = str(row.paper)
            author_id = str(row.author)
            self._process_paper(new_sample, paper_id)
            self._process_author(new_sample, author_id, new_sample["year"])
            samples.append(new_sample)
        # print(samples)
        return samples

    def get_ranking_papers(self):
        return [
            self._process_paper({}, str(paper_id))
            for paper_id in tqdm(self.ranking["papers"], "Data: loading papers")
        ]
    
    def get_ranking_authors(self, paper_year: int, start_idx: int, end_idx: int):
        return [
            self._process_author({}, str(author_id), paper_year)
            for author_id in self.ranking["authors"][start_idx: end_idx]
        ]
