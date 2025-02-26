import argparse
import json
import yaml

from ranking import evaluate_ranker, generate_utility_predictions
from paper_embedding import fit_paper_embedding, generate_paper_embeddings
from semantic_scholar_data import process_papers, process_citations, process_citing_papers, process_authors, \
    kaggle_json_to_parquet, generate_ranking_sample, generate_samples, unify_papers, get_abstracts, process_references, process_embedding
from train_eval import train, eval


def load_if_exists(path: str):
    if path is not None:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    return None


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('runner_config', help='Path of the runner configuration file')
    parser.add_argument('data_config', help='Path of the data configuration file')
    parser.add_argument('--embedder-config', help='Path of the embedder configuration file')
    parser.add_argument('--model-config', help='Path of the model configuration file')
    parser.add_argument('--ranker-config', help='Path of the ranker configuration file')
    args = parser.parse_args()
    return {
        "runner": load_if_exists(args.runner_config),
        "data": load_if_exists(args.data_config),
        "embedder": load_if_exists(args.embedder_config),
        "model": load_if_exists(args.model_config),
        "ranker": load_if_exists(args.ranker_config)
    }


def runner(config: dict):
    if config["runner"]["data"]["kaggle_json_to_parquet"]:
        kaggle_json_to_parquet(config)
    if config["runner"]["data"]["process_papers"]:
        process_papers(config)
    if config["runner"]["data"]["process_embedding"]:
        process_embedding(config)
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
    if config["runner"]["data"]["generate_samples"]:
        generate_samples(config)
        

    # if config["runner"]["paper_embedding"]["fit"]:
    #     fit_paper_embedding(config)

    # if config["runner"]["model"]["train"]:
    #     train(config)
    # if config["runner"]["model"]["eval"]:
    #     eval(config)
        
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

