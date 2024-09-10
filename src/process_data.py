from collections import defaultdict
import json
import os
import time
import pandas as pd
from tqdm import tqdm
import requests

DATA_DIR = "/home/loaner/workspace/data"
DATASET_START_YEAR = 2011
DATASET_END_YEAR = 2021
CITATION_YEARS = 3
QUERY_BASE_DELAY = 0.1
QUERY_MULT_DELAY = 1.2
QUERY_BATCH_SIZE = 20
QUERY_DUMP_INTERVAL = 10
# https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_get_paper
QUERY_PAPER_FIELDS = "year,referenceCount,isOpenAccess,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,authors,citations.year,citations.authors,references.authors"

def kaggle_json_to_parquet():
    with open(os.path.join(DATA_DIR, "arxiv-metadata-oai-snapshot.json")) as f:
        kaggle_data = []
        for l in tqdm(f.readlines(), "Parsing json"):
            l = json.loads(l)
            if "cs." in l["categories"]:
                categories = l["categories"].split()
                for c in categories:
                    if c.startswith("cs"):
                        l["categories"] = categories
                        kaggle_data.append(l)
                        break
        kaggle_data = pd.DataFrame(kaggle_data)
        kaggle_data.to_parquet(os.path.join(DATA_DIR, "kaggle_data.parquet"))
        print(kaggle_data)


def get_citing_authors(l: list, paper_year: int, citation_years: int = CITATION_YEARS):
    authors = set()
    for citation in l:
        try:
            year = int(citation["year"])
            if year < paper_year + citation_years:
                for author in citation["authors"]:
                    authors.add(int(author["authorId"]))
        except TypeError:
            # catch year is null
            continue
    return list(authors)


def normalize_paper_response(j: json):
    j["s2FieldsOfStudy"] = list(set([tmp["category"] for tmp in j["s2FieldsOfStudy"] if tmp["category"] is not None]))
    j["authors"] = list(set([int(tmp["authorId"]) for tmp in j["authors"] if tmp["authorId"] is not None]))
    j["citing_authors"] = get_citing_authors(j["citations"], int(j["year"]))
    del j["citations"]
    j["cited_authors"] = list(set([int(author["authorId"]) for ref in j["references"] for author in ref["authors"] if author["authorId"] is not None]))
    del j["references"]
    del j["paperId"]
    return j


def query_papers_dataset():
    kaggle_data = pd.read_parquet(os.path.join(DATA_DIR, "kaggle_data.parquet"))
    print(f"Filtering relevant years (year < {DATASET_END_YEAR}) & (year >= {DATASET_START_YEAR})")
    kaggle_data['update_date'] = pd.to_datetime(kaggle_data['update_date'])
    kaggle_data['year_updated'] = kaggle_data['update_date'].dt.year
    kaggle_data = kaggle_data[(kaggle_data["year_updated"] < DATASET_END_YEAR) & (kaggle_data["year_updated"] >= DATASET_START_YEAR)]

    # Work on few papers
    kaggle_data = kaggle_data.sample(frac=1., random_state=0)
    kaggle_data = kaggle_data[:100]
    
    papers_path = os.path.join(DATA_DIR, "papers.json")
    if os.path.exists(papers_path):
        with open(papers_path) as f:
            papers = json.load(f)
    else:
        papers = {}

    print(f"kaggle_data size {len(kaggle_data)}")
    paper_ids = [id for id in kaggle_data["id"] if id not in papers.keys()]
    print(f"Papers without info: {len(paper_ids)}")

    delay = QUERY_BASE_DELAY
    pbar = tqdm(total=len(kaggle_data))
    pbar.update(len(kaggle_data) - len(paper_ids))
    idx = 0
    while idx < len(paper_ids):
        batch_ids = paper_ids[idx: idx + QUERY_BATCH_SIZE]
        response = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': QUERY_PAPER_FIELDS},
            json={"ids": [f"ARXIV:{paper_id}" for paper_id in batch_ids]}
        )
        if response.status_code != 200:
            if "error" in response.text:
                print(response.text)
                lkslsk
            delay *= QUERY_MULT_DELAY
            print(f" - Sleeping for {delay} seconds")
            if "Too Many Requests" not in response.text:
                print(json.loads(response.text))
            time.sleep(delay)
            continue

        for arxiv_id, paper_response in zip(batch_ids, response.json()):
            if paper_response is None:
                print(f"{arxiv_id} returned None")
                papers[arxiv_id] = None
                continue
            if paper_response["year"] is None:
                print(f"{arxiv_id} year is None")
                papers[arxiv_id] = None
                continue
            papers[arxiv_id] = normalize_paper_response(paper_response)
        
        idx += QUERY_BATCH_SIZE
        pbar.update(QUERY_BATCH_SIZE)
        delay = max(QUERY_BASE_DELAY, delay / QUERY_MULT_DELAY)

        if idx % (QUERY_BATCH_SIZE * QUERY_DUMP_INTERVAL) == 0:
            print("Dumping citations to file")
            with open(papers_path, 'w') as f:
                json.dump(papers, f, indent=4)
        time.sleep(delay)
    pbar.close()

    with open(papers_path, 'w') as f:
        json.dump(papers, f, indent=4)


query_papers_dataset()
# query_citations()