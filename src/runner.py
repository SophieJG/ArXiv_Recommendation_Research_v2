import argparse
import json
import yaml

from data_queries import generate_samples, kaggle_json_to_parquet, query_authors, query_papers
from train_eval import train, eval


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_config', help='Path of the data configuration file')
    parser.add_argument('model_config', help='Path of the model configuration file')
    parser.add_argument('runner_config', help='Path of the runner configuration file')
    args = parser.parse_args()
    with open(args.data_config, 'r') as file:
        data_config = yaml.safe_load(file)
    with open(args.model_config, 'r') as file:
        model_config = yaml.safe_load(file)
    with open(args.runner_config, 'r') as file:
        runner_config = yaml.safe_load(file)
    return {
        "data": data_config,
        "model": model_config,
        "runner": runner_config
    }


def runner(config: dict):
    if config["runner"]["data_queries"]["kaggle_json_to_parquet"]:
        kaggle_json_to_parquet(config)
    if config["runner"]["data_queries"]["query_papers"]:
        query_papers(config)
    if config["runner"]["data_queries"]["query_authors"]:
        for batch_size in config["data"]["prepare_authors_data_batch_sizes"]:
            query_authors(config, batch_size)
    if config["runner"]["data_queries"]["generate_samples"]:
        generate_samples(config)
    if config["runner"]["train"]:
        train(config)
    if config["runner"]["eval"]:
        eval(config)
            

if __name__ == '__main__':
    config = load_config()
    print("Running with config:\n", json.dumps(config, indent=4))
    runner(config)
