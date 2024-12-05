import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from data import Data
from rankers.random_ranker import RandomRanker
from rankers.utility_ranker import UtilityRanker
from rankers.diversity_ranker import DiversityRanker
from train_eval import get_model
from util import data_dir, mean_consine_distance, models_dir


def get_ranker(config, items_to_rank: int):
    assert config["ranker"] is not None, "Ranker config is required"
    return {
        "random": RandomRanker,
        "utility": UtilityRanker,
        "diversity": DiversityRanker,
    }[config["ranker"]["ranker"]](items_to_rank, config["ranker"]["params"])


def generate_utility_predictions(config: dict, batch_size: int = 500000):
    """
Generate utility predictions for all pairs of <paper, author> in the test set. Due to memory limitations
the predictions are calculated in batches of `batch_size` predictions
"""
    print("\nGenerating utility predictions for ranking")
    output_path = os.path.join(data_dir(config), f"ranking_utility.parquet")
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        return
    data = Data(config)
    model = get_model(config)
    model.load(models_dir(config), config["model"]["model"], config["model"]["version"])
    papers = data.get_ranking_papers()
    for p in papers:
        assert p["year"] == papers[0]["year"], "Ranking is only supported for the case where all papers are from the same year"
    num_authors_in_batch = int(np.ceil(batch_size / len(data.ranking["papers"])))
    print("num_authors_in_batch:", num_authors_in_batch)
    utility = []
    for author_idx in tqdm(range(0, len(data.ranking["authors"]), num_authors_in_batch), "Calculating utility matrix"):
        authors = data.get_ranking_authors(papers[0]["year"], author_idx, author_idx + num_authors_in_batch)
        utility.append(model.predict_proba_ranking(papers, authors))
    utility = pd.DataFrame(
        np.vstack(utility),
        index=data.ranking["authors"],
        columns=data.ranking["papers"]
    )
    print("Saving to", output_path)
    utility.to_parquet(output_path)


def load_embeddings(config: dict):
    """
Load pre-calculated paper embeddings
"""
    tmp = np.load(os.path.join(data_dir(config), "ranking_papers.npz"))
    paper_ids = tmp["paper_ids"]
    embeddings = tmp["embeddings"]
    # Verify that embedding vectors are normalized
    for norm in np.linalg.norm(embeddings, axis=1):
        assert norm >= 0.999 and norm <= 1.001, f"Embedding l2 norm is {norm}"
    return {
        paper_ids[idx]: embeddings[idx, :] for idx in range(len(paper_ids))
    }


def safe_index(l: list, key: str):
    """
Return the index of key in l. If key not in l, returns the length of l
"""
    try:
        return l.index(key) 
    except ValueError:
        pass
    return len(l)


def evaluate_ranker(config: dict):
    """
Evaluate the ranker on the test fold
"""
    top_k = config["data"]["top_k"]
    items_to_rank = max(top_k)

    ranker = get_ranker(config, items_to_rank)
    utility = pd.read_parquet(os.path.join(data_dir(config), f"ranking_utility.parquet"))
    paper_embeddings = load_embeddings(config)
    data = Data(config)
    labels = pd.DataFrame.from_records(data.ranking["pairs"], columns=["paper", "author"])

    # Use the ranker to rank the first `items_to_rank` papers for each author
    ranked = ranker.rank(utility, paper_embeddings)
    
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
    
    # Diversity is defined as 1 minus the mean cosine distance between the top-k recommended papers - higher diversity implies
    # a more diverse set.
    diversity_k = top_k
    diversity = []
    for k in diversity_k:
        authors_diversity = []
        for ranked_papers in ranked.values():
            embeddings = [paper_embeddings[str(paper)] for paper in ranked_papers[:k]]
            authors_diversity.append(mean_consine_distance(embeddings))
        diversity.append(1. - np.mean(authors_diversity))

    metrics = {
        f"MRR (clipped to {items_to_rank})": np.mean(mrr),
        **{f"Precision @ {k}": np.mean(top_k_prec[idx]) for idx, k in enumerate(top_k)},
        **{f"Hit rate @ {k}": top_k_hit[idx] / len(author_min_hit) for idx, k in enumerate(top_k)},
        **{f"Diversity @ {k}": diversity[idx] for idx, k in enumerate(diversity_k)},
    }
    print(json.dumps(metrics, indent=4))
