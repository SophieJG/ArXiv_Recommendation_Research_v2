import pandas as pd
import json
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import copy
import gzip
import os
all_paper_path = "/share/garg/arxiv_runs/bryan_all/data/tmp/paper_info.json"
ArxivId_to_paperId_path = "/share/garg/scholar_inbox_datasets/data/arxivId_to_paperId.json"
scholar_inbox_data_path = "/share/garg/scholar_inbox_datasets/data/rated_papers.csv"
semantic_scholar_ArxivId_to_paperId_path = "/share/garg/scholar_inbox_datasets/data/semantic_scholar_arxivId_to_paperId.json"

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

def generate_arxiv_id_to_paper_id_mapping(all_paper_path: str, paperId_to_ArxivId_path: str):
    if os.path.exists(paperId_to_ArxivId_path):
        print(f"{paperId_to_ArxivId_path} exists - Skipping")
        return
    with open(all_paper_path, 'r') as f:
        all_papers = json.load(f)
    paperId_to_ArxivId = {}
    for index, row in all_papers.items():
        paperId_to_ArxivId[row['arxiv_id']] = index
    with open(paperId_to_ArxivId_path, 'w') as f:
        json.dump(paperId_to_ArxivId, f)
    return paperId_to_ArxivId



def _process_papers_inner(path: str, ids: list, id_type: str):
    """
A helper function to process a single SemS papers chunk file. Should only be called through process_papers. Go
over all papers in the file and returns the info of all papers with id in `ids`

Arguments:
    path: the file to load and process
    ids: a list of strings/integers specifying which papers to load
    id_type: either CorpusId or ArXiv
"""
    assert id_type in ["CorpusId", "ArXiv"]
    # _load_papers_inner could be called in parallel so it's important to copy the inputs to avoid locks
    ids = set([str(id) for id in ids])
    id_type = copy.deepcopy(id_type)
    papers = {}
    with gzip.open(path, "rt", encoding="UTF-8") as fin:
        for l in fin:
            j = json.loads(l)
            if j["externalids"][id_type] not in ids:
                # If the paper id is not in the set of required ids - ignore the paper
                continue
            papers[str(j["externalids"]["ArXiv"])] = str(j["corpusid"])
    return papers


def add_paperId_to_scholar_inbox_data(scholar_inbox_data_path: str, ArxivId_to_paperId_path: str, semantic_scholar_ArxivId_to_paperId_path: str):
    with open(ArxivId_to_paperId_path, 'r') as f:
        ArxivId_to_paperId = json.load(f)
    scholar_inbox_data = pd.read_csv(scholar_inbox_data_path)
    
    scholar_inbox_data['arxiv_id'] = scholar_inbox_data['arxiv_id'].astype('string')

    # Initialize paperId column as string type (object in pandas)
    if 'paperId' not in scholar_inbox_data.columns:
        scholar_inbox_data['paperId'] = pd.NA
    scholar_inbox_data['paperId'] = scholar_inbox_data['paperId'].astype('string')
    
    print("\nNumber of null paperIds before:", scholar_inbox_data['paperId'].isna().sum())
    
    # First mapping attempt with local dataset
    scholar_inbox_data['paperId'] = scholar_inbox_data['arxiv_id'].map(ArxivId_to_paperId)
    
    # Find missing IDs
    missing_arxiv_ids = set(scholar_inbox_data[scholar_inbox_data['paperId'].isna()]['arxiv_id'])
    print(f"\nNumber of missing arxiv ids after local lookup: {len(missing_arxiv_ids)}")
    
    if os.path.exists(semantic_scholar_ArxivId_to_paperId_path):
        with open(semantic_scholar_ArxivId_to_paperId_path, 'r') as f:
            semantic_scholar_ArxivId_to_paperId = json.load(f)
        missing_arxiv_ids = missing_arxiv_ids - set(semantic_scholar_ArxivId_to_paperId.keys())
        print(f"\nNumber of missing arxiv ids after semantic scholar lookup: {len(missing_arxiv_ids)}")
    else:
        # Search semantic scholar data for missing IDs
        dict_list = multi_file_query(
            "/share/garg/semantic_scholar_Nov2024/2024-11-05/papers/*.gz",
            _process_papers_inner,
            1,
            ids=missing_arxiv_ids,
            id_type="ArXiv"
        )

        # Merge additional mappings
        arxiv_id_to_paper_id = {}
        for d in tqdm(dict_list, "merging paperid into scholar inbox data"):
            arxiv_id_to_paper_id.update(d)

        with open(semantic_scholar_ArxivId_to_paperId_path, 'w') as f:
            json.dump(arxiv_id_to_paper_id, f)
    


    print("Number of additional papers found:", len(arxiv_id_to_paper_id))
    
    # Update missing values with semantic scholar data
    mask = scholar_inbox_data['paperId'].isna()
    scholar_inbox_data.loc[mask, 'paperId'] = scholar_inbox_data.loc[mask, 'arxiv_id'].map(arxiv_id_to_paper_id)
    
    print("\nNumber of null paperIds after semantic scholar lookup:", scholar_inbox_data['paperId'].isna().sum())
    # Count unique arxiv IDs where paperId is NA
    print("\nNumber of unique arxiv IDs with missing paperIds:", scholar_inbox_data[scholar_inbox_data['paperId'].isna()]['arxiv_id'].nunique())
    
    scholar_inbox_data.to_csv(scholar_inbox_data_path, index=False)
    return

if __name__ == "__main__":
    generate_arxiv_id_to_paper_id_mapping(all_paper_path, ArxivId_to_paperId_path)
    add_paperId_to_scholar_inbox_data(scholar_inbox_data_path, ArxivId_to_paperId_path, semantic_scholar_ArxivId_to_paperId_path)