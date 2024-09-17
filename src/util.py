import os


NUM_NEGATIVE = 50
DATASET_START_YEAR = 2011
DATASET_END_YEAR = 2021
CITATION_YEARS = 3

DATA_DIR = "/home/loaner/workspace/data"
PAPERS_PATH = os.path.join(DATA_DIR, "papers.json")
AUTHORS_PATH = os.path.join(DATA_DIR, "authors.json")
KAGGLE_DATA_PATH = os.path.join(DATA_DIR, "kaggle_data.parquet")
