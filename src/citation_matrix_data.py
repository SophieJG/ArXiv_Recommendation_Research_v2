from collections import defaultdict
import glob
import gzip
import json
import os
import pickle
from joblib import Parallel, delayed
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

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

def _process_nodes_edges_inner(path: str, cited_ids: list):
    """
A helper function to process a single SemS citations chunk file. Should only be called through process_citations
return a list of pairs <cited id, citing id>. Cited id is in the set of `cited_ids`

Arguments:
    path: the file to load and process
    cited_ids: a list paper corpus ids
"""
    raise RuntimeError("Are you sure you want to run this? It shouldn't be necessary due to processing in semantic_scholar_data.py")
    # _process_citations_inner could be called in parallel so it's important to copy the ids list to avoid locks
    cited_ids = set([int(id) for id in cited_ids])
    node_set = set()
    citing_cited_edges = []
    with gzip.open(path, "rt", encoding="UTF-8") as fin:
        for l in fin:
            j = json.loads(l)
            if j["citingcorpusid"] is None or j["citedcorpusid"] is None:
                continue
            if j["citedcorpusid"] in cited_ids:
                node_set.add(citing)
                node_set.add(cited)
                citing_cited_edges.append((int(j["citingcorpusid"]), int(j["citedcorpusid"])))
    return citing_cited_edges, node_set

def _process_edges_inner(path: str, node_ids: list):
    """
A helper function to process a single SemS citations chunk file. Should only be called through process_citations
return a list of pairs <cited id, citing id>. Cited id is in the set of `cited_ids`

Arguments:
    path: the file to load and process
    cited_ids: a list paper corpus ids
"""
    # TODO: update docstring
    # _process_citations_inner could be called in parallel so it's important to copy the ids list to avoid locks
    node_ids = set([int(id) for id in node_ids])
    citing_cited_edges = []
    with gzip.open(path, "rt", encoding="UTF-8") as fin:
        for l in fin:
            j = json.loads(l)
            if j["citingcorpusid"] is None or j["citedcorpusid"] is None:
                continue
            if j["citedcorpusid"] in node_ids and j["citingcorpusid"] in node_ids:
                # only add edges between nodes in the node_ids list
                citing_cited_edges.append((int(j["citingcorpusid"]), int(j["citedcorpusid"])))
    return citing_cited_edges

def _process_edges(config, node_ids: list):
    """"""
    # TODO: references will often not make it into node_list right now, might want to rethink pipeline
    res = multi_file_query(
        os.path.join(config["data"]["semantic_scholar_path"], "citations", "*.gz"),
        _process_edges_inner,
        config["data"]["n_jobs"],
        node_ids=node_ids
    )

    # multi_file_query returns a list of dicts. Merge it to a single dict. Note that a single paper can have citations in multiple
    # res files so the dictionaries needs to be "deep" merged
    edges = []
    for citing_cited_pairs in tqdm(res, "merging files"):
        for pair in citing_cited_pairs:
            edges.append(pair)

    return edges

def save_citation_matrix(matrix_path, matrix, node_list, node_to_index):
    """
    Save the citation matrix and metadata to a file using pickle.

    Args:
        matrix_path (str): The path to the file where data will be saved.
        matrix (scipy.sparse.csr_matrix): The citation matrix.
        node_list (list): Sorted list of node (paper) IDs.
        node_to_index (dict): Mapping from paper IDs to matrix indices.
    """
    with open(matrix_path, 'wb') as f:
        pickle.dump({
            'matrix': matrix,
            'node_list': node_list,
            'node_to_index': node_to_index,
        }, f)
    print(f"Citation matrix and metadata saved to {matrix_path}")

def load_citation_matrix(matrix_path):
    """
    Load the citation matrix and metadata from a pickle file.

    Args:
        matrix_path (str): The path to the file where data is saved.

    Returns:
        tuple: A tuple (matrix, node_list, node_to_index)
    """
    with open(matrix_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Citation matrix and metadata loaded from {matrix_path}")
    return data['matrix'], data['node_list'], data['node_to_index']

def process_citation_matrix(config: dict):
    """
Using the list of corpus ids of the Arxiv papers we query the `citations` table to get corpus ids for all
citing papers. This includes all citing papers, disregarding publication year
"""
    print("\nLoading Citation Matrix") # TODO: update comment

    matrix_path = os.path.join(data_dir(config), "citation_matrix.npz") # TODO: update save path for csr matrix
    if os.path.exists(matrix_path):
        print(f"{matrix_path} exists - Skipping")
        return
    
    # load the unified papers
    unified_papers_path = papers_path(config)
    unified_papers = json.load(open(unified_papers_path))
    

    if config["data"]["test_is_2020"]:
        # Test set is the year 2020
        cutoff_year = 2020
    else:
        # test_is_2020 needs to be true for matrix-based methods to prevent leakage during unsupervised training
        # TODO: consider ways to handle a randomized split (is it possible?)
        raise ValueError("test_is_2020 in your data config must be set to true for matrix-based methods to prevent leakage during unsupervised training")

    paper_ids_pubdates = [(int(corpus_id), info['publicationdate']) for corpus_id, info in unified_papers.items() if info['year'] < cutoff_year] # assumes each paper has publication date info

    # Create an ordered node list and mapping from paper ID to an index
    # node_list = [corpus_id for corpus_id, pubdate in sorted(paper_ids_pubdates, key=lambda x: x[1])] # TODO: decide how to handle missing publication dates
    node_list = [corpus_id for corpus_id, _ in np.random.permutation(paper_ids_pubdates)] # NOTE: shuffling for now to eliminate unknown bias
    node_to_index = {node: i for i, node in enumerate(node_list)}
    n_nodes = len(node_list)
    print(f"Number of nodes (papers): {n_nodes}")

    # create edge list
    edges = _process_edges(config, node_list)
    print(f"Number of edges: {len(edges)}")

    # --- Step 2: Build the sparse citation matrix ---
    # We treat rows as "citing" papers and columns as "cited" papers.
    rows = []
    cols = []
    data = []

    for citing, cited in edges:
        i = node_to_index[citing]
        j = node_to_index[cited]
        rows.append(i)
        cols.append(j)
        data.append(1)  # Unweighted edge

    # Construct a CSR (Compressed Sparse Row) matrix
    M = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Save the sparse matrix along with node list and index
    save_citation_matrix(matrix_path, M, node_list, node_to_index)
    