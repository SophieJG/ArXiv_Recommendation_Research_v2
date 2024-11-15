import os

import numpy as np


def data_dir(config: dict):
    return os.path.join(config["data"]["base_path"], "data")


def tmp_data_dir(config: dict):
    return os.path.join(config["data"]["base_path"], "data", "tmp")


def models_dir(config: dict):
    return os.path.join(config["data"]["base_path"], "models")


def papers_path(config: dict):
    return os.path.join(data_dir(config), "papers.json")


def authors_path(config: dict):
    return os.path.join(data_dir(config), "authors.json")


def kaggle_data_path(config: dict):
    return os.path.join(data_dir(config), "kaggle_data.parquet")


def passthrough_func(x):
    return x


def mean_consine_distance(embedding: list):
    """
Calculate average cosine distance between all embedding vectors in a list. It is assumed that the 
vectors are normalized to have an l2 norm of 1.
"""
    embeddings = np.vstack(embedding)
    return np.mean(np.matmul(embeddings, embeddings.transpose()))
