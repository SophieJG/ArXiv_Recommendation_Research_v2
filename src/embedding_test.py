from collections import defaultdict
import glob
import gzip
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import copy

from util import authors_path, data_dir, kaggle_data_path, papers_path, tmp_data_dir


def multi_file_query(
    files_path: str,
    processing_func: callable,
    n_jobs: int,
    *args,
    **kwargs
):
    """
This is a utility function facilitating parallel load and processing of data from disk. All files 
in `files_path` are processed by applying `processing_func` on them. The number of files processed
in parallel is given by `n_jobs`. *args and **kwargs are passed to `processing_func`

Arguments:
    files_path: the path to the list of files to load and process. We run glob(files_path) to get the list
        of files
    processing_func: the function used to process the chunk files
    n_jobs: how many processes to run in parallel
    *args: passed to processing_func
    **kwargs: passed to processing_func

Returns:
    a list containing the outputs of processing_func
"""
    files = glob.glob(files_path)
    return Parallel(n_jobs=n_jobs)(delayed(processing_func)(f, *args, **kwargs) for f in tqdm(files, f"n_jobs={n_jobs}"))

def find_paper_by_id(id: int):
    """
    Find a paper by its id
    """
    # files_path = os.path.join("/share/garg/semantic_scholar_Nov2024/2024-11-05", "embeddings-specter_v2", "*.gz")
    # embeddings =  multi_file_query(files_path, _process_paper_data, 50, target = id)
    abstract_files_path = os.path.join("/share/garg/semantic_scholar_Nov2024/2024-11-05", "abstracts", "*.gz")
    abstracts = multi_file_query(abstract_files_path, _process_abstract_inner, 50, paper_ids = [id])
    abstract_dict = {}
    for d in abstracts:
        abstract_dict.update(d)
    return abstract_dict[id]


def _process_paper_data(path: str, target: int):
    """
    Process a paper data from the `papers` table
    """
    with gzip.open(path, "rt", encoding="UTF-8") as fin:
        for line in fin:
            j = json.loads(line)
            if j["corpusid"] == target:
                return j['vector']
    return None

def _process_abstract_inner(path: str, paper_ids: list):
    paper_ids = set([int(id) for id in paper_ids])

    abstracts = {}
    with gzip.open(path, "rt", encoding="UTF-8") as fin:
        for l in fin:
            j = json.loads(l)
            if int(j["corpusid"]) not in paper_ids:
                continue
            abstracts[j["corpusid"]] = j["abstract"]
    return abstracts

if __name__ == "__main__":
    target = 90191874
    abstract = find_paper_by_id(target)
    print(abstract)