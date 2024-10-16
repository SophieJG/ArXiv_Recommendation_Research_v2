import argparse
import json
import yaml

from data_queries import generate_ranking_sample, generate_samples, kaggle_json_to_parquet, query_authors, query_papers
from ranking import evaluate_ranker, generate_ranking_predictions
from paper_embedding import fit_paper_embedding, generate_paper_embeddings
from train_eval import train, eval


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_config', help='Path of the data configuration file')
    parser.add_argument('model_config', help='Path of the model configuration file')
    parser.add_argument('runner_config', help='Path of the runner configuration file')
    parser.add_argument('ranker_config', help='Path of the ranker configuration file', nargs='?', default=None)
    args = parser.parse_args()
    with open(args.data_config, 'r') as file:
        data_config = yaml.safe_load(file)
    with open(args.model_config, 'r') as file:
        model_config = yaml.safe_load(file)
    with open(args.runner_config, 'r') as file:
        runner_config = yaml.safe_load(file)
    if args.ranker_config is not None:
        with open(args.ranker_config, 'r') as file:
            ranker_config = yaml.safe_load(file)
    else:
        ranker_config = None
    return {
        "data": data_config,
        "model": model_config,
        "runner": runner_config,
        "ranker": ranker_config
    }


def runner(config: dict):
    if config["runner"]["data_queries"]["kaggle_json_to_parquet"]:
        kaggle_json_to_parquet(config)
    if config["runner"]["data_queries"]["query_papers"]:
        query_papers(config)
    if config["runner"]["data_queries"]["query_authors"]:
        # For unknown reason, Semantic Scholar consistently fail on some authors. Thus, we skip batches which return
        # errors. Since we use large batches, a single errorneous author can cause us to miss the data of many authors.
        # In order to avoid that, we query the authors several times with decreasing batch sizes, according to
        # config["data"]["prepare_authors_data_batch_sizes"]
        for batch_size in config["data"]["prepare_authors_data_batch_sizes"]:
            query_authors(config, batch_size)
    if config["runner"]["data_queries"]["generate_samples"]:
        generate_samples(config)
    if config["runner"]["paper_embedding"]["fit"]:
        fit_paper_embedding(config)
    if config["runner"]["train"]:
        train(config)
    if config["runner"]["eval"]:
        eval(config)
    if config["runner"]["ranking"]["generate_samples"]:
        generate_ranking_sample(config)
    if config["runner"]["ranking"]["generate_predictions"]:
        generate_ranking_predictions(config)
    if config["runner"]["ranking"]["generate_paper_embeddings"]:
        generate_paper_embeddings(config)
    if config["runner"]["ranking"]["evaluate"]:
        evaluate_ranker(config)


if __name__ == '__main__':
    config = load_config()
    print("Running with config:\n", json.dumps(config, indent=4))
    runner(config)
