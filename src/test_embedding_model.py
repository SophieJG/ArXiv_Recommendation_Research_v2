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
import math
from util import authors_path, data_dir, kaggle_data_path, papers_path, tmp_data_dir
from models.spector_embed import Specter2Embedder

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



def _generate_paper_embeddings_inner(paper_id_abstract_dict: dict, n_jobs: int):
    """
    A helper function to generate embeddings for a single paper. Should only be called through process_paper_embedding.
    Takes a paper ID and returns a dictionary with the paper ID and its generated embedding.

    Arguments:
        paper_id: string ID of the paper to generate embedding for
        paper_info: dictionary containing paper information including abstracts
    Returns:
        Dictionary mapping paper ID to its embedding vector
    """

    def _generate_single_embedding(pid, abstract):
        print(f"Generating embedding for paper {pid}")
        paper_embeddings = {}
        try:
            # Initialize embedder
            embedder = Specter2Embedder()
                
            # Generate embedding
            embedding = embedder.compute_embedding(abstract)
            print(type(embedding))
            # Store result
            paper_embeddings[str(pid)] = embedding.tolist()
            
        except Exception as e:
            print(f"Error generating embedding for paper {pid}: {str(e)}")
            
        return paper_embeddings

    # Convert dictionary items to list and then chunk
    chunk_size = math.ceil(len(paper_id_abstract_dict) / n_jobs)
    items = list(paper_id_abstract_dict.items())
    chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]

    # Process chunks in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_generate_single_embedding)(pid, abstract) 
        for chunk in tqdm(chunks, desc="Generating embeddings") 
        for pid, abstract in chunk.items()
    )

    # Merge results
    final_embeddings = {}
    for r in results:
        final_embeddings.update(r)

    return final_embeddings

def test_embedding_model():
    """
    Using the list of corpus ids of the Arxiv papers we query the `embedding` table to get spector2 embedding for each paper.
    """
    # multi_file_query returns a list of dicts. Merge it to a single dict. Note that a single paper can have citations in multiple
    # res files so the dictionaries needs to be "deep" merged
    
    missing_paperid_abstract_dict = {}
    missing_paper_count = 0

    embedding_papers = {}

    if missing_paper_count > 0:
        print(f"Generating embeddings for {missing_paper_count} papers")
        missing_paper_embeddings = _generate_paper_embeddings_inner(missing_paperid_abstract_dict, 1)
        print(type(missing_paper_embeddings))
        print(missing_paper_embeddings)
        embedding_papers.update(missing_paper_embeddings)
    
    # print(f"embedding_papers:\n", embedding_papers)

if __name__ == "__main__":
    test_embedding_model()