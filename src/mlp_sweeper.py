import argparse
import json
import yaml
import itertools
from ranking import evaluate_ranker, generate_utility_predictions
from paper_embedding import fit_paper_embedding, generate_paper_embeddings
from semantic_scholar_data import process_papers, process_citations, process_citing_papers, process_authors, \
    kaggle_json_to_parquet, generate_ranking_sample, generate_samples, unify_papers, get_abstracts, process_references, process_paper_embedding, process_author_paper_embedding, process_author_embedding
from train_eval import train, eval


def load_if_exists(path: str):
    if path is not None:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    return None


def load_config():
    mlp_config = load_if_exists("/home/mz572/ArXiv_Recommendation_Research_v2/configs/mlp.yml")
    params = mlp_config["params"]
    all_combinations = [
        dict(zip(params.keys(), values))
        for values in itertools.product(*params.values())
    ]
    for combination in all_combinations:
        print(combination)
        print("--------------------------------")
    mlp_config["params"] = all_combinations
    return {
        "runner": load_if_exists("/home/mz572/ArXiv_Recommendation_Research_v2/configs/runner.yml"),
        "data": load_if_exists("/home/mz572/ArXiv_Recommendation_Research_v2/configs/data_1k.yml"),
        "embedder": load_if_exists("/home/mz572/ArXiv_Recommendation_Research_v2/configs/category_embedder.yml"),
        "model": mlp_config,
        "ranker": load_if_exists("/home/mz572/ArXiv_Recommendation_Research_v2/configs/utility_ranker.yml")
    }


def runner(config: dict):
    if config["runner"]["data"]["kaggle_json_to_parquet"]:
        kaggle_json_to_parquet(config)
    if config["runner"]["data"]["process_papers"]:
        process_papers(config)
    if config["runner"]["data"]["process_citations"]:
        process_citations(config)
    if config["runner"]["data"]["process_citing_papers"]:
        process_citing_papers(config)
    if config["runner"]["data"]["process_authors"]:
        process_authors(config)
    if config["runner"]["data"]["get_abstracts"]:
        get_abstracts(config)
    if config["runner"]["data"]["unify_papers"]:
        unify_papers(config)
    if config["runner"]["data"]["process_references"]:
        process_references(config)
    if config["runner"]["data"]["process_paper_embedding"]:
        process_paper_embedding(config)
    if config["runner"]["data"]["process_author_paper_embedding"]:
        process_author_paper_embedding(config)
    if config["runner"]["data"]["process_author_embedding"]:
        process_author_embedding(config)
    if config["runner"]["data"]["generate_samples"]:
        generate_samples(config)
        

    # if config["runner"]["paper_embedding"]["fit"]:
    #     fit_paper_embedding(config)
    for param_comb in config["model"]["params"]:
        print(param_comb)
        if config["runner"]["model"]["train"]:
            train(config, param_comb)
        if config["runner"]["model"]["eval"]:
            eval(config, param_comb)

    # if config["runner"]["ranking"]["generate_samples"]:
    #     generate_ranking_sample(config)
    # if config["runner"]["ranking"]["generate_predictions"]:
    #     generate_utility_predictions(config)
    # if config["runner"]["ranking"]["generate_paper_embeddings"]:
    #     generate_paper_embeddings(config)
    # if config["runner"]["ranking"]["evaluate"]:
    #     evaluate_ranker(config)


if __name__ == '__main__':
    config = load_config()

    # print("Running with config:\n", json.dumps(config, indent=4))
    runner(config)

