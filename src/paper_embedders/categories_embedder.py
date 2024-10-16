import os
import joblib
import pandas as pd
from tqdm import tqdm
from paper_embedders.base_embedder import BaseEmbedder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from util import passthrough_func


class CategoriesEmbedder(BaseEmbedder):
    def __init__(
        self,
        params: dict
    ):
        self.pipeline = None

    def papers_to_dataframe(
        self,
        papers: list
    ):
        samples = []
        for paper in tqdm(papers, "Converting papers to dataframe"):
            samples.append({
                key: paper[key] for key in ["title", "abstract", "categories"]
            })
        return pd.DataFrame.from_records(samples)
        
    def fit(
        self,
        papers: list
    ):
        df = self.papers_to_dataframe(papers)
        self.pipeline = ColumnTransformer([
            ("categories", CountVectorizer(analyzer=passthrough_func), "categories")
        ])
        self.pipeline.fit(df)

    def embed(
        self,
        papers: list
    ):
        return self.pipeline.transform(self.papers_to_dataframe(papers))

    def save_(
        self,
        path: str
    ):
        assert self.pipeline is not None
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.pipeline, f, protocol=5)        

    def load_(
        self,
        path: str
    ):
        assert self.pipeline is None
        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            self.pipeline = joblib.load(f)
