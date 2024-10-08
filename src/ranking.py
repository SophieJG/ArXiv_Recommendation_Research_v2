import json
import os

import numpy as np
import pandas as pd
from data import Data
from rankers.random_ranker import RandomRanker
from rankers.sort_ranker import SortRanker
from train_eval import get_model
from util import data_dir, models_dir


def get_ranker(config):
    return {
        "random": RandomRanker(config["ranker"]["params"]),
        "sort": SortRanker(config["ranker"]["params"])
    }[config["ranker"]["ranker"]]


def generate_ranking_predictions(config: dict, batch_size: int = 500000):
    print("\nGenerating predictions for ranking")
    data = Data(config)
    model = get_model(config)
    model.load(models_dir(config), config["model"]["model"], config["model"]["version"])
    proba = []
    for idx0 in range(0, len(data.ranking), batch_size):
        print(f"Working on batch {int(idx0 / batch_size)} / {int(np.ceil(len(data.ranking) / batch_size))}")
        proba.append(model.predict_proba(data, f"ranking_{idx0}_{idx0 + batch_size}"))
    proba = np.hstack(proba)
    ranking = data.ranking.copy()
    ranking["proba"] = proba
    proba = ranking.pivot(index='author', columns='paper', values=['proba'])
    proba.columns = [c[1] for c in proba.columns]
    proba.to_parquet(os.path.join(data_dir(config), f"ranking_proba.parquet"))
    labels = ranking[ranking["label"]]
    labels = labels[["author", "paper"]]
    labels.to_parquet(os.path.join(data_dir(config), f"ranking_labels.parquet"))


def evaluate_ranker(config: dict):
    assert config["ranker"] is not None
    ranker = get_ranker(config)
    proba = pd.read_parquet(os.path.join(data_dir(config), f"ranking_proba.parquet"))
    ranked = ranker.rank(proba)
    labels = pd.read_parquet(os.path.join(data_dir(config), f"ranking_labels.parquet"))
    
    # Calculate standard ranking metrics e.g. perc top k
    paper_ranks = []
    for author, paper in zip(labels["author"], labels["paper"]):
        author_ranked = ranked[author]
        paper_ranks.append(author_ranked.index(paper))
    
    top_k = [1, 5, 10, 100]
    top_k_res = [[], [], [], []]
    mrr = []
    for r in paper_ranks:
        mrr.append(1 / (r + 1))
        for idx, k in enumerate(top_k):
            top_k_res[idx].append(r < k)
    metrics = {
        "mrr": np.mean(mrr),
        **{f"top_{k}": np.mean(top_k_res[idx]) for idx, k in enumerate(top_k)}
    }
    print(json.dumps(metrics, indent=4))
