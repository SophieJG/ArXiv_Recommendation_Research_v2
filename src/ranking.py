import json
import os

import numpy as np
import pandas as pd
from data import Data
from rankers.random_ranker import RandomRanker
from rankers.sort_ranker import SortRanker
from rankers.diversity_ranker import DiversityRanker
from train_eval import get_model
from util import data_dir, mean_consine_distance, models_dir


def get_ranker(config, items_to_rank: int):
    return {
        "random": RandomRanker,
        "sort": SortRanker,
        "diversity": DiversityRanker,
    }[config["ranker"]["ranker"]](items_to_rank, config["ranker"]["params"])


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


def load_embeddings(config: dict):
    tmp = np.load(os.path.join(data_dir(config), "ranking_papers.npz"))
    paper_ids = tmp["paper_ids"]
    embeddings = tmp["embeddings"]
    # Verify that embedding vectors are normalized
    for norm in np.square(embeddings).sum(axis=1):
        assert norm >= 0.999 and norm <= 1.001
    return {
        paper_ids[idx]: embeddings[idx, :] for idx in range(len(paper_ids))
    }


def safe_index(l: list, key: str):
    # Return the index of key in l. If key not in l, returns the length of l
    try:
        return l.index(key) 
    except ValueError:
        pass
    return len(l)


def evaluate_ranker(config: dict):
    assert config["ranker"] is not None
    top_k = config["data"]["top_k"]
    items_to_rank = max(top_k)

    ranker = get_ranker(config, items_to_rank)
    proba = pd.read_parquet(os.path.join(data_dir(config), f"ranking_proba.parquet"))
    paper_embeddings = load_embeddings(config)
    ranked = ranker.rank(proba, paper_embeddings)
    labels = pd.read_parquet(os.path.join(data_dir(config), f"ranking_labels.parquet"))
    
    # Preprocessing needed in order to calculate standard ranking metrics e.g. precision top k, hit top k
    paper_ranks = []
    for author, paper in zip(labels["author"], labels["paper"]):
        author_ranked = ranked[author]
        assert len(author_ranked) == items_to_rank
        paper_ranks.append(safe_index(author_ranked, paper))
    
    # Calculate precision top k
    top_k_prec = [[] for _ in range(len(top_k))]
    mrr = []
    for r in paper_ranks:
        mrr.append(1 / (r + 1))
        for idx, k in enumerate(top_k):
            top_k_prec[idx].append(r < k)

    # Calculate hit metrics. Defined as: for every author did we recommend a paper that was interacted with in the top k
    author_min_hit = {}
    for author, paper in zip(labels["author"], labels["paper"]):
        author_ranked = ranked[author]
        index = safe_index(author_ranked, paper)
        if author not in author_min_hit:
            author_min_hit[author] = index
        else:
            author_min_hit[author] = min(author_min_hit[author], index)

    top_k_hit = [0] * len(top_k)
    for _, min_hit in author_min_hit.items():
        for idx, k in enumerate(top_k):
            if min_hit < k:
                top_k_hit[idx] += 1
    
    diversity_k = top_k
    diversity = []
    for k in diversity_k:
        authors_diversity = []
        for ranked_papers in ranked.values():
            embeddings = [paper_embeddings[paper] for paper in ranked_papers[:k]]
            authors_diversity.append(mean_consine_distance(embeddings))
        diversity.append(1. - np.mean(authors_diversity))

    metrics = {
        f"MRR (clipped to {items_to_rank})": np.mean(mrr),
        **{f"Precision @ {k}": np.mean(top_k_prec[idx]) for idx, k in enumerate(top_k)},
        **{f"Hit rate @ {k}": top_k_hit[idx] / len(author_min_hit) for idx, k in enumerate(top_k)},
        **{f"Diversity @ {k}": diversity[idx] for idx, k in enumerate(diversity_k)},
    }
    print(json.dumps(metrics, indent=4))
