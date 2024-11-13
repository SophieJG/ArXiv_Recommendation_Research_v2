from collections import defaultdict
import glob
import gzip
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from util import data_dir, kaggle_data_path


def parse_s2fieldsofstudy(fieldsofstudy_list: list):
    if fieldsofstudy_list is None:
        return []
    return list(set([fieldsofstudy["category"] for fieldsofstudy in fieldsofstudy_list if fieldsofstudy["category"] is not None]))


def process_paper_data(j: dict, allow_none_year: bool):
    # Process Semantic Scholar data for a paper
    if not allow_none_year and j["year"] is None:
        return None
    return {
        "authors": list(set([int(tmp["authorId"]) for tmp in j["authors"] if tmp["authorId"] is not None])),
        "s2fieldsofstudy": parse_s2fieldsofstudy(j["s2fieldsofstudy"]),
        **{k: j[k] for k in ["title", "year", "referencecount", "publicationdate"]}
    }


def load_papers(config: dict, ids: list, id_type: str):
    ids = [str(id) for id in ids]
    assert id_type in ["CorpusId", "ArXiv"]
    files = glob.glob(os.path.join(config["data"]["semantic_scholar_path"], "papers", "*.gz"))
    papers = {}
    for idx, f in enumerate(files):
        if idx >= 5:
            break
        with gzip.open(f, "rt", encoding="UTF-8") as fin:
            for l in tqdm(fin, f"Processing file {idx} / {len(files)}"):
                j = json.loads(l)
                if j["externalids"][id_type] not in ids:
                    continue
                processed_paper = process_paper_data(j, allow_none_year=False)
                if processed_paper is None:
                    continue
                papers[j["corpusid"]] = processed_paper
        print(f"#papers = {len(papers)} / {len(ids)}")
    return papers


def load_citations(config: dict, ids: list):
    ids = [int(id) for id in ids]
    files = glob.glob(os.path.join(config["data"]["semantic_scholar_path"], "citations", "*.gz"))
    citing_papers = defaultdict(lambda: [])
    cited_papers = defaultdict(lambda: [])
    for idx, f in enumerate(files):
        if idx >= 5:
            break
        with gzip.open(f, "rt", encoding="UTF-8") as fin:
            for l in tqdm(fin, f"Processing file {idx} / {len(files)}"):
                j = json.loads(l)
                if j["citingcorpusid"] is None or j["citedcorpusid"] is None:
                    continue
                if j["citingcorpusid"] in ids:
                    citing_papers[j["citingcorpusid"]].append(j["citedcorpusid"])
                if j["citedcorpusid"] in ids:
                    cited_papers[j["citedcorpusid"]].append(j["citingcorpusid"])
        print(f"#citing_papers = {len(citing_papers)}, #cited_papers = {len(cited_papers)}")
    return citing_papers, cited_papers


def load_author_papers(config: dict, ids: list):
    ids = [str(id) for id in ids]
    files = glob.glob(os.path.join(config["data"]["semantic_scholar_path"], "papers", "*.gz"))
    author_papers = defaultdict(lambda: [])
    for idx, f in enumerate(files):
        if idx >= 5:
            break
        with gzip.open(f, "rt", encoding="UTF-8") as fin:
            for l in tqdm(fin, f"Processing file {idx} / {len(files)}"):
                j = json.loads(l)
                for author in j["authors"]:
                    if author["authorId"] in ids:
                        author_papers[author["authorId"]].append(process_paper_data(j, allow_none_year=True))
        print(f"#author_papers = {len(author_papers)} / {len(ids)}")
    return author_papers


def process_papers(config: dict):
    print("\nLoading papers info from Semantic Scholar")
    output_path = os.path.join(data_dir(config), "paper_info.json")
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        return
    kaggle_data = pd.read_parquet(kaggle_data_path(config))
    papers = load_papers(config, kaggle_data["id"], "ArXiv")
    print(f"Saving to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(papers, f, indent=4)


def process_citations(config: dict):
    print("\nLoading citation info from Semantic Scholar")
    citing_papers_path = os.path.join(data_dir(config), "citing_papers.json")
    cited_papers_path = os.path.join(data_dir(config), "cited_papers.json")
    if os.path.exists(cited_papers_path):
        print(f"{cited_papers_path} exists - Skipping")
        return
    with open(os.path.join(data_dir(config), "paper_info.json")) as f:
        paper_info = json.load(f)
    citing_papers, cited_papers = load_citations(config, list(paper_info.keys()))
    for path, data in zip([citing_papers_path, cited_papers_path], [citing_papers, cited_papers]):
        print(f"Saving to {path}")
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)


def process_citing_papers(config: dict):
    print("\nLoading citing papers info from Semantic Scholar")
    output_path = os.path.join(data_dir(config), "citing_paper_info.json")
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        return
    with open(os.path.join(data_dir(config), "citing_papers.json")) as f:
        citing_papers = json.load(f)
    paper_ids = [id for tmp in citing_papers.values() for id in tmp]
    papers = load_papers(config, paper_ids, "CorpusId")
    print(f"Saving to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(papers, f, indent=4)


def process_authors(config: dict):
    print("\nLoading citing authors info from Semantic Scholar")
    output_path = os.path.join(data_dir(config), "author_info.json")
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        return
    with open(os.path.join(data_dir(config), "citing_paper_info.json")) as f:
        citing_papers_info = json.load(f)
    author_ids = list(set([author_id for citing_paper in citing_papers_info.values() for author_id in citing_paper["authors"]]))
    authors = load_author_papers(config, author_ids)
    print(f"Saving to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(authors, f, indent=4)


def kaggle_json_to_parquet(config: dict):
    """
Converts the json downloaded from kaggle to parquet and filters not-relevant papers. Currently only CS papers are used.
Additionally, takes a random sample of config["data"]["num_papers"] papers
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
    if config["data"]["num_papers"] > 0:
        # Use only num_papers. If num_papers is 0 - use all papers
        kaggle_data = kaggle_data.sample(frac=1., random_state=0) # Suffle the papers
        kaggle_data = kaggle_data[:config["data"]["num_papers"]]
    os.makedirs(data_dir(config), exist_ok=True)
    kaggle_data.to_parquet(kaggle_data_path(config))
    print(kaggle_data)


def generate_samples(config: dict):
    """
Generate three lists of samples that will be used for training, validation and evaluation. Each sample is a triplet: `<paper_id, author_id, label>`. The
label can be true (the author cited the paper) or false.

Positive samples - all cases in which an author cited a paper (in the alloted time)
Negative samples - a random author who did not cite the paper. For each paper the number of negative samples generated is
    set by config["data"]["num_negative"]

The samples are split by paper id i.e. all the samples of a paper will be placed in the same data fold. The test set could be either the year 2020, or
a random set of papers. See config["data"]["test_is_2020"]
"""
    print("\nGenerating train, validation and test folds")
    rng = np.random.default_rng(seed=42)
    with open(os.path.join(data_dir(config), "paper_info.json")) as f:
        papers = json.load(f)
    with open(os.path.join(data_dir(config), "author_info.json")) as f:
        authors = json.load(f)
    with open(os.path.join(data_dir(config), "citing_paper_info.json")) as f:
        citing_paper_info = json.load(f)
    max_author_papers = config["data"]["max_author_papers"]
    valid_authors = [int(key) for key, value in authors.items() if value is not None and len(value) < max_author_papers]
    print(f"Removed {len(authors) - len(valid_authors)} authors with more than {max_author_papers} papers")
    print(f"Valid authors: {len(valid_authors)} / {len(authors)}")

    # Create the list of samples
    num_invalid_citing_authors = 0
    num_null_papers = 0
    samples = []
    for paper_id, paper in papers.items(): # Go over all papers
        if paper is None or str(paper_id) not in citing_paper_info:
            num_null_papers += 1
            print(paper_id)
            continue
        # Positive samples - all authors who cited the paper
        for citing_author in paper["citing_authors"]:
            if citing_author not in valid_authors:
                num_invalid_citing_authors += 1
                continue
            samples.append(
                {
                    "paper": paper_id,
                    "year": paper["year"],
                    "author": citing_author,
                    "label": True
                }
            )
        # Negative samples - a random author who did not cite the paper
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
    print(samples)
    return
    
    # Split the list of samples to train, validation and test folds
    if config["data"]["test_is_2020"]:
        # Test set is the year 2020
        test = samples[samples["year"] == 2020]
        validation = samples[(samples["year"] >= 2019) & (samples["year"] < 2020)]
        train = samples[samples["year"] < 2019]
    else:
        # Test set is randomly selected from all years
        train_validation_samples, test = split_by_paper(samples, test_size=0.2)
        train, validation = split_by_paper(train_validation_samples, test_size=0.2)
    for d, name in [(train, "train"), (validation, "validation"), (test, "test")]:
        print(f"{name}:", len(d))
        d = d.drop("year", axis=1)
        d.to_csv(os.path.join(data_dir(config), f"{name}.csv"), index=False)
    print("Invalid citing authors:", num_invalid_citing_authors)
    print("num_null_papers:", num_null_papers)


def generate_ranking_sample(config: dict):
    pass
#     """
# Generate the data sample used for ranking. These are all papers in the test fold and all authors who 
# interacted with at least one paper in the test fold
# """
#     print("\nGenerating ranking fold")
#     test_papers = pd.read_csv(os.path.join(data_dir(config), "test.csv"))["paper"].unique()
#     print(f"#test papers {len(test_papers)}")
#     with open(papers_path(config)) as f:
#         papers = json.load(f)
#     with open(authors_path(config)) as f:
#         authors = json.load(f)
#     valid_authors = [int(key) for key, value in authors.items() if value is not None]
#     print(f"Valid authors: {len(valid_authors)} / {len(authors)}")
    
#     test_authors = set()
#     for paper_id in test_papers:  # Go over all papers
#         if paper_id not in papers or papers[paper_id] is None:
#             continue
#         test_authors.update(set(papers[paper_id]["citing_authors"]))
#     print(f"Citing authors: {len(test_authors)}")
#     test_authors = test_authors.intersection(set(valid_authors))
#     print(f"Valid citing authors: {len(test_authors)}")

#     samples = []
#     for paper_id in test_papers:  # Go over all papers
#         if paper_id not in papers or papers[paper_id] is None:
#             continue
#         paper = papers[paper_id]
#         for author in test_authors:
#             samples.append(
#                 {
#                     "paper": paper_id,
#                     "author": author,
#                     "label": author in paper["citing_authors"]
#                 }
#             )
#     samples = pd.DataFrame.from_records(samples)
#     print(f"#ranking samples: {len(samples)}")
#     print(f"ranking pos ratio: {samples["label"].mean()}")
#     samples.to_csv(os.path.join(data_dir(config), f"ranking.csv"), index=False)


def split_by_paper(
    df: pd.DataFrame,
    test_size: float,
    random_state: int = 42
):
    """
Split a Pandas DataFrame into two parts. The size of the parts are (1 - test_size) * number_of_samples and test_size * number_of_samples.
It is assumed that the dataframe includes a "paper" column containing a paper id. There could be many samples with the same paper id. The
data is split so if a paper id exists in one split, it would not be put in the other. This is achieved by first splitting the paper ids
and then placing the samples according to the paper id.

Arguments:
df: Pandas DataFrame to be split. Must include "year" and "paper" columns.
test_size: The size of the second fold. 0 < test_size < 1
random_state: int
"""
    papers = df.groupby("paper").agg({"year": {"first"}})
    papers_train, papers_test = train_test_split(papers, test_size=test_size, random_state=random_state)
    df_train = df[df["paper"].isin(papers_train.index)]
    df_test = df[df["paper"].isin(papers_test.index)]
    assert len(df_train) + len(df_test) == len(df)
    return df_train, df_test
