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
import ast
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

def find_paper_by_id(id: list):
    """
    Find a paper by its id
    """
    files_path = os.path.join("/share/garg/semantic_scholar_Nov2024/2024-11-05", "embeddings-specter_v2", "*.gz")
    embeddings =  multi_file_query(files_path, _process_embedding_papers_inner, 50, ids = id)
    print(embeddings)
    abstract_files_path = os.path.join("/share/garg/semantic_scholar_Nov2024/2024-11-05", "abstracts", "*.gz")
    abstracts = multi_file_query(abstract_files_path, _process_abstract_inner, 50, paper_ids = id)
    print(abstracts)
    return


def _process_embedding_papers_inner(path: str, ids: list):
    """
A helper function to process a single SemS papers chunk file. Should only be called through process_papers. Go
over all papers in the file and returns the info of all papers with id in `ids`

Arguments:
    path: the file to load and process
    ids: a list of strings/integers specifying which papers to load
    id_type: either CorpusId or ArXiv
"""
    # _load_papers_inner could be called in parallel so it's important to copy the inputs to avoid locks
    # the corpus id in the embedding file is an integer
    ids = set(int(id) for id in ids)
    paper_embeddings = {}
    with gzip.open(path, "rt", encoding="UTF-8") as fin:
        for l in fin:
            j = json.loads(l)
            if j["corpusid"] not in ids:
                # If the paper id is not in the set of required ids - ignore the paper
                continue
            paper_embeddings[str(j["corpusid"])] = ast.literal_eval(j["vector"])
    return paper_embeddings


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
    target = [231740816, 61225984, 123973632, 244786968, 18146311, 215878656]
    find_paper_by_id(target)
