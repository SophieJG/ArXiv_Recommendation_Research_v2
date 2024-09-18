import os


NUM_NEGATIVE = 20
DATASET_START_YEAR = 2011
DATASET_END_YEAR = 2021
CITATION_YEARS = 3
NUM_PAPERS = 100

DATA_DIR = "/home/loaner/workspace/data"
PAPERS_PATH = os.path.join(DATA_DIR, "papers.json")
AUTHORS_PATH = os.path.join(DATA_DIR, "authors.json")
KAGGLE_DATA_PATH = os.path.join(DATA_DIR, "kaggle_data.parquet")
