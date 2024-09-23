import json
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from sklearn.model_selection import train_test_split


from util import data_dir, kaggle_data_path, papers_path, authors_path


QUERY_FAIL_DELAY = 1
QUERY_FAIL_MULT_DELAY = 2
QUERY_DUMP_INTERVAL = 10


def kaggle_json_to_parquet(config: dict):
    """
Converts the json downloaded from kaggle to parquet and filters not-relevant papers. Currently only CS papers are used.
"""
    if os.path.exists(kaggle_data_path(config)):
        print("Kaggle data already converted to parquet - Skipping")
        return
    print("\nFiltering and converting Arxiv Kaggle data to Pandas")
    with open(config["data"]["arxiv_json_path"]) as f:
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
        print(f"Filtering relevant years (year < {config["data"]["end_year"]}) & (year >= {config["data"]["start_year"]})")
        kaggle_data['update_date'] = pd.to_datetime(kaggle_data['update_date'])
        kaggle_data['year_updated'] = kaggle_data['update_date'].dt.year
        kaggle_data = kaggle_data[(kaggle_data["year_updated"] < config["data"]["end_year"]) & (kaggle_data["year_updated"] >= config["data"]["start_year"])]
        os.makedirs(data_dir(config), exist_ok=True)
        kaggle_data.to_parquet(kaggle_data_path(config))
        print(kaggle_data)


def get_citing_authors(l: list, paper_year: int, citation_years: int):
    authors = set()
    for citation in l:
        try:
            year = int(citation["year"])
            if year < paper_year + citation_years:
                for author in citation["authors"]:
                    authors.add(int(author["authorId"]))
        except TypeError:
            # year is null
            continue
    return list(authors)


def batch_query(
    json_save_path,
    query_ids,
    batch_size,
    query_fields,
    query_url,
    process_response_f,
    **kwargs
):
    if os.path.exists(json_save_path):
        with open(json_save_path) as f:
            data = json.load(f)
    else:
        data = {}
    print(f"Number of query ids: {len(query_ids)}")
    filtered_query_ids = [id for id in query_ids if str(id) not in data.keys()]
    print(f"Ids without info: {len(filtered_query_ids)}")
    api_key = os.environ.get('API_KEY')
    print("Using api_key:", api_key)

    fail_delay = 0
    pbar = tqdm(total=len(query_ids))
    pbar.update(len(query_ids) - len(filtered_query_ids))
    idx = 0
    prev_request_time = 0
    while idx < len(filtered_query_ids):
        batch_ids = filtered_query_ids[idx: idx + batch_size]
        current_time = round(time.time() * 1000)  # time in milisecond
        throughput_delay = prev_request_time + 1000 - current_time
        throughput_delay = max(0, min(throughput_delay, 1000)) / 1000
        if throughput_delay > 0:
            time.sleep(throughput_delay)
        prev_request_time = current_time
        response = requests.post(
            query_url,
            headers={'x-api-key': api_key} if api_key is not None else {},
            params={'fields': query_fields},
            json={"ids": batch_ids}
        )
        if response.status_code != 200:
            if "error" in response.text:
                print(response.text)
                # sys.exit(1)
            fail_delay = max(QUERY_FAIL_DELAY, fail_delay * QUERY_FAIL_MULT_DELAY)
            print(f" - Sleeping for {fail_delay} seconds")
            if "Too Many Requests" not in response.text:
                print(json.loads(response.text))
            time.sleep(fail_delay)
            print("Skipping batch")
            idx += batch_size
            pbar.update(batch_size)
            continue

        for id, response in zip(batch_ids, response.json()):
            if response is None:
                print(f"{id} returned None")
                data[id] = None
                continue
            data[id] = process_response_f(response, **kwargs)
        
        idx += batch_size
        pbar.update(batch_size)
        fail_delay = 0

        if idx % (batch_size * QUERY_DUMP_INTERVAL) == 0:
            # print(f"Dumping data to {json_save_path}")
            with open(json_save_path, 'w') as f:
                json.dump(data, f, indent=4)
    pbar.close()

    with open(json_save_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully queried {len(data)} out of {len(query_ids)}")


def query_papers(config: dict):
    """
Query Semantic Scholar to get additional info about all papers
"""
    print("\nQuering Semantic Scholar for papers info")
    kaggle_data = pd.read_parquet(kaggle_data_path(config))

    # Work on few papers
    kaggle_data = kaggle_data.sample(frac=1., random_state=0)
    if config["data"]["num_papers"] > 0:
        kaggle_data = kaggle_data[:config["data"]["num_papers"]]
    
    def process_paper_response(j: dict, **kwargs):
        if j["year"] is None:
            return None
        j["authors"] = list(set([int(tmp["authorId"]) for tmp in j["authors"] if tmp["authorId"] is not None]))
        j["citing_authors"] = get_citing_authors(j["citations"], int(j["year"]), **kwargs)
        del j["citations"]
        j["cited_authors"] = list(set([int(author["authorId"]) for ref in j["references"] for author in ref["authors"] if author["authorId"] is not None]))
        del j["references"]
        del j["paperId"]
        return j

    # See Semantic Scholar API docs: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
    batch_query(
        json_save_path=papers_path(config),
        query_ids=[f"ARXIV:{id}" for id in kaggle_data["id"]],
        batch_size=100,
        query_fields="year,authors,referenceCount,references.authors,citations.year,citations.authors",
        query_url="https://api.semanticscholar.org/graph/v1/paper/batch",
        process_response_f=process_paper_response,
        citation_years=config["data"]["cication_years"]  # Additional arg for process_paper_response - passed using **kwargs
    )

    # Verify that no papers are missing from the dataset (usually due to failed api calls)
    with open(papers_path(config)) as f:
        papers = json.load(f)
    assert len(papers) == len(kaggle_data)



def query_authors(
    config: dict,
    batch_size: int
):
    """
Query Semantic Scholar for all authores who cited a paper from the dataset
"""
    print(f"\nQuering Semantic Scholar for authors info (batch size = {batch_size})")
    with open(papers_path(config)) as f:
        papers = json.load(f)
    citing_authors = set()
    for paper in papers.values():
        if paper is None:
            continue
        for citing_author in paper["citing_authors"]:
            citing_authors.add(citing_author)
    print(f"{len(papers)} papers have {len(citing_authors)} citing authors")

    def process_author_response(j: json):
        papers = [
            {
                **{key: paper[key] for key in ["year", "title", "fieldsOfStudy"]},
                "s2FieldsOfStudy": list(set([tmp["category"] for tmp in paper["s2FieldsOfStudy"] if tmp["category"] is not None]))
            }
            for paper in j["papers"] if paper["year"] is not None
        ]
        return {
            "papers": papers
        }

    # See Semantic Scholar API docs: https://api.semanticscholar.org/api-docs/graph#tag/Author-Data/operation/post_graph_get_authors
    batch_query(
        json_save_path=authors_path(config),
        query_ids=list(citing_authors),
        batch_size=batch_size,
        query_fields="papers.year,papers.title,papers.fieldsOfStudy,papers.s2FieldsOfStudy",
        query_url="https://api.semanticscholar.org/graph/v1/author/batch",
        process_response_f=process_author_response
    )


def split_by_paper(df: pd.DataFrame, test_size: float, random_state: int = 42):
    papers = df.groupby("paper").agg({"year": {"first"}})
    papers_train, papers_test = train_test_split(papers, test_size=test_size, random_state=random_state)
    df_train = df[df["paper"].isin(papers_train.index)]
    df_test = df[df["paper"].isin(papers_test.index)]
    assert len(df_train) + len(df_test) == len(df)
    return df_train, df_test


def generate_samples(config: dict):
    """
Generates a list of sample triplets: `<paper_id, author_id, label>`. The label can be true (the author cited the paper) or false.
"""
    print("\nGenerating train, validation and test folds")
    rng = np.random.default_rng(seed=42)
    with open(papers_path(config)) as f:
        papers = json.load(f)
    with open(authors_path(config)) as f:
        authors = json.load(f)
    valid_authors = [int(key) for key, value in authors.items() if value is not None]
    print(f"Valid authors: {len(valid_authors)} / {len(authors)}")

    num_citing_authors_not_found = 0
    num_null_papers = 0
    samples = []
    for paper_id, paper in papers.items():
        if paper is None:
            num_null_papers += 1
            continue
        # Positive samples
        for citing_author in paper["citing_authors"]:
            if citing_author not in valid_authors:
                num_citing_authors_not_found += 1
                continue
            samples.append(
                {
                    "paper": paper_id,
                    "year": paper["year"],
                    "author": citing_author,
                    "label": True
                }
            )
        # Negative samples
        for author_idx in rng.integers(low=0, high=len(valid_authors), size=config["data"]["num_negative"]):
            if valid_authors[author_idx] in paper["citing_authors"]:
                # Actually a positive sample
                continue
            samples.append(
                {
                    "paper": paper_id,
                    "year": paper["year"],
                    "author": valid_authors[author_idx],
                    "label": False
                }
            )
    samples = pd.DataFrame.from_records(samples)
    
    if config["data"]["test_is_2020"]:
        # Test set is the year 2020
        test = samples[samples["year"] == 2020]
        train_validation_samples = samples[samples["year"] < 2020]
    else:
        # Test set is randomly selected from all years
        train_validation_samples, test = split_by_paper(samples, test_size=0.2)
    train, validation = split_by_paper(train_validation_samples, test_size=0.2)
    for d, name in [(train, "train"), (validation, "validation"), (test, "test")]:
        print(f"{name}:", len(d))
        d = d.drop("year", axis=1)
        d.to_csv(os.path.join(data_dir(config), f"{name}.csv"), index=False)
    print("Citing authors not found in database:", num_citing_authors_not_found)
    print("num_null_papers:", num_null_papers)
