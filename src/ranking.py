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
from util import data_dir, mean_consine_distance, models_dir, model_version_path


def get_ranker(config, items_to_rank: int):
    assert config["ranker"] is not None, "Ranker config is required"
    return {
        "random": RandomRanker,
        "utility": UtilityRanker,
        "diversity": DiversityRanker,
    }[config["ranker"]["ranker"]](items_to_rank, config["ranker"]["params"])


def generate_utility_predictions_100_negatives(config: dict, batch_size: int = 500000):
    """
Generate utility predictions for all pairs of <paper, author> in the test set. For each author, we sample one positive
paper and num_random_negatives negative papers.
"""
    raise Exception("Are you sure you want to use this function? Ranking scores for 100 negatives are already additionally calculated in the default pipeline.")
    print("\nGenerating utility predictions for ranking")
    output_path = os.path.join(
        model_version_path(models_dir(config), config["model"]["model"], config["model"]["version"]), 
        f"ranking_utility.parquet"
        )
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        return
    data = Data(config)
    model = get_model(config)
    model.load(models_dir(config), config["model"]["model"], config["model"]["version"])
    papers = data.get_ranking_papers()
    for p in papers:
        assert p["year"] == papers[0]["year"], "Ranking is only supported for the case where all papers are from the same year"

    # Create a set of positive paper-author pairs for quick lookup
    positive_pairs = set((str(p), str(a)) for p, a in data.ranking["pairs"])
    
    # Create a mapping of author to their positive papers
    author_to_positive_papers = {}
    for paper, author in data.ranking["pairs"]:
        author = str(author)
        if author not in author_to_positive_papers:
            author_to_positive_papers[author] = []
        author_to_positive_papers[author].append(str(paper))

    # For each author, sample one positive paper and num_random_negatives negative papers
    author_samples = {}
    rng = np.random.default_rng(seed=42)
    num_random_negatives = config["data"]["num_negative_ranking"]
    for author in data.ranking["authors"]:
        author = str(author)
        # Sample one positive paper
        if author in author_to_positive_papers:
            positive_paper = rng.choice(author_to_positive_papers[author])
        else:
            continue  # Skip authors with no positive papers
            
        # Sample negative papers
        negative_papers = []
        available_papers = [str(p) for p in data.ranking["papers"] if p != positive_paper and (p, author) not in positive_pairs]
        if len(available_papers) >= num_random_negatives:
            negative_papers = rng.choice(available_papers, size=num_random_negatives, replace=False)
        else:
            negative_papers = available_papers  # Use all available papers if we don't have enough
            
        author_samples[author] = {
            "positive": positive_paper,
            "negatives": negative_papers
        }

    # Create a mapping of paper IDs to their indices in the papers list
    paper_id_to_idx = {str(p["paper_id"]): i for i, p in enumerate(papers)}

    # Initialize utility matrix with negative infinity
    utility = np.full((len(data.ranking["authors"]), len(papers)), -np.inf)
    
    # Process each author
    num_ranking_authors = len(data.ranking["authors"])
    for author_idx in tqdm(range(len(data.ranking["authors"])), "Calculating utility matrix", miniters=max(1, num_ranking_authors // 100)):
        author_obj = data.get_ranking_authors(papers[0]["year"], author_idx, author_idx + 1)[0]
        author_id = str(author_obj["author"]["id"])
        if author_id not in author_samples:
            continue
            
        # Get the sampled papers for this author
        sampled_papers = [author_samples[author_id]["positive"]] + list(author_samples[author_id]["negatives"])
        
        # Create a list of paper indices for the sampled papers
        paper_indices = [paper_id_to_idx[paper_id] for paper_id in sampled_papers]
        
        # Get the paper objects for the sampled papers
        sampled_paper_objects = [papers[idx] for idx in paper_indices]
        
        # Calculate utility for just these papers
        author_utility = model.predict_proba_ranking(sampled_paper_objects, [author_obj])
        
        # Fill in the utility matrix for this author
        for paper_idx, paper_id in zip(paper_indices, sampled_papers):
            utility[author_idx, paper_idx] = author_utility[0, paper_indices.index(paper_idx)]

    # Convert to DataFrame
    utility = pd.DataFrame(
        utility,
        index=data.ranking["authors"],
        columns=data.ranking["papers"]
    )
    print("Saving to", output_path)
    utility.to_parquet(output_path)


def generate_utility_predictions(config: dict, batch_size: int = 500000):
    """
Generate utility predictions for all pairs of <paper, author> in the test set. Due to memory limitations
the predictions are calculated in batches of `batch_size` predictions
"""
    print("\nGenerating utility predictions for ranking")
    output_path = os.path.join(
        model_version_path(models_dir(config), config["model"]["model"], config["model"]["version"]), 
        f"ranking_utility.parquet"
        )
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
    
    # Calculate total iterations and mininterval for ~100 updates
    total_iterations = len(range(0, len(data.ranking["authors"]), num_authors_in_batch))
    mininterval = max(1, total_iterations / 100)
    
    for author_idx in tqdm(range(0, len(data.ranking["authors"]), num_authors_in_batch), 
                          desc="Calculating utility matrix",
                          mininterval=mininterval):
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
    tmp = np.load(os.path.join(
        model_version_path(models_dir(config), config["embedder"]["embedder"], config["embedder"]["version"]), 
        f"ranking_papers.npz"
    ))
    paper_ids = tmp["paper_ids"]
    embeddings = tmp["embeddings"]
    # Verify that embedding vectors are normalized
    for norm in np.square(embeddings).sum(axis=1):
        assert norm >= 0.999 and norm <= 1.001
    return {
        paper_ids[idx]: embeddings[idx, :] for idx in range(len(paper_ids))
    }


def safe_index(l: list, key: str):
    """
Return the index of key in l. If key not in l, returns negative infinity.
"""
    try:
        return l.index(key)
    except ValueError:
        pass
    return len(l)


def evaluate_ranker(config: dict):
    """
    Evaluate the ranker on the test fold, both with all papers and with num_negative_ranking random negatives per author
    """
    top_k = config["data"]["top_k"]
    items_to_rank = max(top_k)
    num_negatives = config["data"]["num_negative_ranking"]

    ranker = get_ranker(config, items_to_rank)
    utility = pd.read_parquet(os.path.join(
        model_version_path(models_dir(config), config["model"]["model"], config["model"]["version"]), 
        f"ranking_utility.parquet"
        ))
    # Convert utility index and columns to strings for consistent keying
    utility.index = utility.index.map(str)
    utility.columns = utility.columns.map(str)
        
    paper_embeddings = load_embeddings(config) if config["runner"]["ranking"]["generate_paper_embeddings"] else None
    data = Data(config)
    labels = pd.DataFrame.from_records(data.ranking["pairs"], columns=["paper", "author"])

    # Create a mapping of author to their positive papers for quick lookup
    author_to_positive_papers = {}
    for paper, author in data.ranking["pairs"]: # paper, author are ints
        author_str = str(author)
        if author_str not in author_to_positive_papers:
            author_to_positive_papers[author_str] = set()
        author_to_positive_papers[author_str].add(str(paper))

    print(f"\nFound {len(author_to_positive_papers)} unique authors with positive papers from {len(labels)} pairs.")

    # Sample num_negatives random negatives for each author
    rng = np.random.default_rng(seed=42)
    author_to_sampled_negatives = {}
    all_papers_str = set(str(p) for p in data.ranking["papers"]) # set of string paper IDs
    
    for author_int_id in data.ranking["authors"]: # author_int_id is an int
        author_str = str(author_int_id)
        positive_papers_for_author_str = author_to_positive_papers.get(author_str, set()) # set of string paper IDs
        # Ensure available_negatives are strings
        available_negatives_str = list(all_papers_str - positive_papers_for_author_str)
        
        if len(available_negatives_str) >= num_negatives:
            author_to_sampled_negatives[author_str] = rng.choice(available_negatives_str, size=num_negatives, replace=False)
        else:
            author_to_sampled_negatives[author_str] = np.array(available_negatives_str) if available_negatives_str else np.array([])

    print(f"Sampled negatives for {len(author_to_sampled_negatives)} authors (authors from data.ranking['authors']).")

    # Use the ranker to rank the first `items_to_rank` papers for each author
    # `utility` now has string index (author_str) and string columns (paper_str)
    # `ranker.rank` should return `ranked` with string author_str keys and list of string paper_str values
    ranked = ranker.rank(utility, paper_embeddings) 
    
    # Preprocessing needed in order to calculate standard ranking metrics e.g. precision top k, hit top k
    paper_ranks = []
    author_to_paper_ranks = {} # keys: author_str, values: list of ranks
    for author_int, paper_int in zip(labels["author"], labels["paper"]):
        author_str_loop = str(author_int)
        paper_str_loop = str(paper_int)
        
        if author_str_loop not in ranked:
            print(f"Warning (Global Metrics): Author {author_str_loop} not found in ranked output. Skipping.")
            continue
            
        author_ranked_list_str = ranked[author_str_loop] # list of string paper IDs
        assert len(author_ranked_list_str) == items_to_rank, \
            f"Author {author_str_loop} ranked list length {len(author_ranked_list_str)} != items_to_rank {items_to_rank}"
        
        rank = safe_index(author_ranked_list_str, paper_str_loop) # paper_str_loop is str
        paper_ranks.append(rank)
        
        if author_str_loop not in author_to_paper_ranks:
            author_to_paper_ranks[author_str_loop] = []
        author_to_paper_ranks[author_str_loop].append(rank)
    
    # Calculate precision top k
    top_k_prec = [[] for _ in range(len(top_k))]
    for r in paper_ranks:
        for idx, k_val in enumerate(top_k):
            top_k_prec[idx].append(r < k_val)
    
    # Calculate MRR properly - use the first relevant item for each author
    mrr_values = []
    for author_str_loop, ranks_list in author_to_paper_ranks.items(): # author_str_loop is str
        min_rank = min(ranks_list) if ranks_list else float('inf')
        if min_rank < items_to_rank: # items_to_rank is the max possible rank considered relevant
            mrr_values.append(1 / (min_rank + 1))
        else:
            mrr_values.append(0)
    
    # Calculate hit metrics. Defined as: for every author did we recommend a paper that was interacted with in the top k
    author_min_hit = {} # keys: author_str, values: min_rank
    for author_int, paper_int in zip(labels["author"], labels["paper"]):
        author_str_loop = str(author_int)
        paper_str_loop = str(paper_int)

        if author_str_loop not in ranked:
            # Already warned above
            continue
            
        author_ranked_list_str = ranked[author_str_loop] # list of string paper IDs
        index = safe_index(author_ranked_list_str, paper_str_loop)
        
        if author_str_loop not in author_min_hit:
            author_min_hit[author_str_loop] = index
        else:
            author_min_hit[author_str_loop] = min(author_min_hit[author_str_loop], index)

    top_k_hit = [0] * len(top_k)
    if author_min_hit: # Ensure author_min_hit is not empty before iterating / division
        for _, min_hit_val in author_min_hit.items():
            for idx, k_val in enumerate(top_k):
                if min_hit_val < k_val:
                    top_k_hit[idx] += 1
    
    # Diversity is defined as 1 minus the mean similarity score between the top-k recommended papers
    if paper_embeddings:
        # TODO: improve diversity score to calculate a mean similarity score, where the
        # similarity score depends on the embedding model used
        # TODO: implement this in a way where not all the embeddings have to be loaded at once
        diversity_k = top_k
        diversity = []
        for k_val in diversity_k:
            authors_diversity_scores = []
            for author_str_key in ranked: # iterate over authors present in ranked output
                ranked_papers_list_str = ranked[author_str_key][:k_val]
                if len(ranked_papers_list_str) == k_val : # Ensure we have enough papers for this k
                    # paper_embeddings keys are str, ranked_papers_list_str contains str
                    embeddings = [paper_embeddings[p_str] for p_str in ranked_papers_list_str if p_str in paper_embeddings]
                    if len(embeddings) == k_val: # Check if all embeddings were found
                         authors_diversity_scores.append(mean_consine_distance(embeddings))
            if authors_diversity_scores:
                diversity.append(1. - np.mean(authors_diversity_scores))
            else:
                diversity.append(0.0) # Or np.nan, depending on desired behavior for missing data

    metrics = {
        f"MRR @ {items_to_rank} (clipped to 0)": np.mean(mrr_values) if mrr_values else 0.0,
        **{f"Precision @ {k_val}": np.mean(top_k_prec[idx]) if top_k_prec[idx] else 0.0 for idx, k_val in enumerate(top_k)},
        **{f"Hit rate @ {k_val}": top_k_hit[idx] / len(author_min_hit) if author_min_hit else 0.0 for idx, k_val in enumerate(top_k)},
    }
    if paper_embeddings:
        metrics.update({
            **{f"Diversity @ {k_val}": diversity[idx] if idx < len(diversity) else 0.0 for idx, k_val in enumerate(diversity_k)},
        })

    print(f"Ranking results using all samples (Total number of papers: {len(data.ranking['papers'])}):")
    print(json.dumps(metrics, indent=4))

    ##### Now compute metrics with num_negatives random negatives #######

    # Create a mapping of author to their positive papers for evaluation
    author_to_positive_papers = {}
    for paper, author in data.ranking["pairs"]:
        author_str = str(author)
        paper_str = str(paper)
        if author_str not in author_to_positive_papers:
            author_to_positive_papers[author_str] = set()
        author_to_positive_papers[author_str].add(paper_str)
    
    # Use our new method to rank with sampled negatives
    local_ranked, sampled_positives = ranker.rank_with_sampled_negatives(
        utility,
        author_to_positive_papers,
        num_negatives,
        paper_embeddings
    )
    
    # Calculate metrics on the results of sampled negative ranking
    paper_ranks_neg = []
    author_to_paper_ranks_neg = {}
    
    # For each author and their ranked papers
    for author_str, ranked_papers in local_ranked.items():
        # Get the positive paper that was sampled for this author
        positive_paper = sampled_positives[author_str]
        
        # Find its rank in the ranked list, or use a penalty value if not found
        if positive_paper in ranked_papers:
            rank = ranked_papers.index(positive_paper)
        else:
            # If positive paper isn't in the ranked list, assign a penalty rank
            # Use num_negatives + 1 as the penalty (worst possible rank + 1)
            rank = num_negatives + 1
            
        paper_ranks_neg.append(rank)
        
        if author_str not in author_to_paper_ranks_neg:
            author_to_paper_ranks_neg[author_str] = []
        author_to_paper_ranks_neg[author_str].append(rank)
    
    # Calculate precision @ k
    top_k_prec_neg = [[] for _ in range(len(top_k))]
    for rank in paper_ranks_neg:
        for idx, k_val in enumerate(top_k):
            top_k_prec_neg[idx].append(rank < k_val)
    
    # Calculate MRR
    mrr_values_neg = []
    for author_str, ranks in author_to_paper_ranks_neg.items():
        min_rank = min(ranks) if ranks else float('inf')
        if min_rank < items_to_rank:  # Use items_to_rank as the threshold, consistent with ranking length
            mrr_values_neg.append(1 / (min_rank + 1))
        else:
            mrr_values_neg.append(0)
    
    # Calculate hit rate
    author_min_hit_neg = {}
    for author_str, ranked_papers in local_ranked.items():
        positive_paper = sampled_positives[author_str]
        if positive_paper in ranked_papers:
            rank = ranked_papers.index(positive_paper)
        else:
            # If positive paper isn't in the ranked list, assign a penalty rank
            rank = float('inf')
        author_min_hit_neg[author_str] = rank
    
    top_k_hit_neg = [0] * len(top_k)
    for min_hit in author_min_hit_neg.values():
        for idx, k_val in enumerate(top_k):
            if min_hit < k_val:
                top_k_hit_neg[idx] += 1
    
    # Calculate diversity if paper embeddings are available
    diversity_neg = []
    if paper_embeddings:
        for k_val in top_k:
            authors_diversity_scores = []
            for author_str, ranked_papers in local_ranked.items():
                top_k_papers = ranked_papers[:k_val]
                if len(top_k_papers) == k_val:
                    embeddings = [paper_embeddings[p] for p in top_k_papers if p in paper_embeddings]
                    if len(embeddings) == k_val:
                        authors_diversity_scores.append(mean_consine_distance(embeddings))
            if authors_diversity_scores:
                diversity_neg.append(1. - np.mean(authors_diversity_scores))
            else:
                diversity_neg.append(0.0)
    
    num_authors_evaluated_neg = len(author_min_hit_neg)

    metrics_neg = {
        f"MRR @ {items_to_rank} (clipped to 0) ({num_negatives} neg)": np.mean(mrr_values_neg) if mrr_values_neg else 0.0,
        **{f"Precision @ {k_val} ({num_negatives} neg)": np.mean(top_k_prec_neg[idx]) if top_k_prec_neg[idx] else 0.0 for idx, k_val in enumerate(top_k)},
        **{f"Hit rate @ {k_val} ({num_negatives} neg)": (top_k_hit_neg[idx] / num_authors_evaluated_neg) if num_authors_evaluated_neg > 0 else 0.0 for idx, k_val in enumerate(top_k)},
    }
    if paper_embeddings:
        metrics_neg.update({
            **{f"Diversity @ {k_val} ({num_negatives} neg)": diversity_neg[idx] if idx < len(diversity_neg) else 0.0 for idx, k_val in enumerate(diversity_k)},
        })
    
    print(f"\nRanking results using {num_negatives} random negative samples (evaluated on {num_authors_evaluated_neg} authors):")
    print(json.dumps(metrics_neg, indent=4))
