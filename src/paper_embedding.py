import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import Data
from paper_embedders.categories_embedder import CategoriesEmbedder
from util import data_dir, models_dir


def get_paper_embedder(config):
    assert config["embedder"] is not None, "Embedder config is required"
    return {
        "category": CategoriesEmbedder(config["embedder"]["params"])
    }[config["embedder"]["embedder"]]


def get_papers(data: Data, fold: pd.DataFrame):
    papers = []
    for paper_id in tqdm(fold["paper"].unique(), "Loading papers"):
        paper_id = str(paper_id)
        paper_data = data.papers[paper_id]
        arxiv_id = str(paper_data["arxiv_id"])
        kaggle_data = data.kaggle_data.loc[arxiv_id]
        papers.append(
            {
                "id": paper_id,
                **{key: kaggle_data[key] for key in ["title", "abstract"]},
                "categories": kaggle_data["categories"].tolist()
            }
        )
    return papers

def fit_paper_embedding(config: dict):
    """
Fit a paper embedder on the train data and store to disk
"""
    print("\nFitting paper embeddings")
    data = Data(config)
    train_papers = get_papers(data, data.train)
    embedder = get_paper_embedder(config)
    embedder.fit(train_papers)
    embedder.save(models_dir(config), config["embedder"]["embedder"], config["embedder"]["version"])


def generate_paper_embeddings(config: dict):
    """
Generate paper embeddings for all papers in the test set
"""
    print("\nGenerating paper embeddings")
    output_path = os.path.join(data_dir(config), "ranking_papers.npz")
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        return
    embedder = get_paper_embedder(config)
    embedder.load(models_dir(config), config["embedder"]["embedder"], config["embedder"]["version"])
    data = Data(config)
    papers = get_papers(data, data.test)
    embeddings = embedder.embed(papers).toarray()
    # Normalize embeddings
    embeddings = embeddings / np.sqrt(np.square(embeddings).sum(axis=1))[:, np.newaxis]
    paper_ids = np.array([paper["id"] for paper in papers])
    print(f"Saving to {output_path}")
    np.savez(os.path.join(data_dir(config), "ranking_papers.npz"), paper_ids=paper_ids, embeddings=embeddings)
