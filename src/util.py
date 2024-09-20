import os

def data_dir(config: dict):
    return os.path.join(config["data"]["base_path"], "data")

def models_dir(config: dict):
    return os.path.join(config["data"]["base_path"], "models")

def papers_path(config: dict):
    return os.path.join(data_dir(config), "papers.json")

def authors_path(config: dict):
    return os.path.join(data_dir(config), "authors.json")

def kaggle_data_path(config: dict):
    return os.path.join(data_dir(config), "kaggle_data.parquet")
