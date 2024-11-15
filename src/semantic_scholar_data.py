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

from util import authors_path, data_dir, kaggle_data_path, papers_path


def multi_file_query(
    files_path: str,
    processing_func: callable,
    n_jobs: int,
    *args,
    **kwargs
):
    files = glob.glob(files_path)
    res = Parallel(n_jobs=n_jobs)(delayed(processing_func)(f, *args, **kwargs) for f in tqdm(files, f"n_jobs={n_jobs}"))

    # Result of processing_func is either a single dict or a list of dicts
    # In any case, these needs to be merged
    if isinstance(res[0], dict):
        # Each element of res is a single dict
        d_all = {}
        for d in res:
            d_all.update(d)
        return d_all

    # Each element of res is a list of dicts
    num_dicts = len(res[0])
    dicts = [{} for _ in range(num_dicts)]
    for d in res:
        for idx in range(num_dicts):
            dicts[idx].update(d[idx])
    if num_dicts == 1:
        return dicts[0]
    return dicts


def parse_s2fieldsofstudy_(fieldsofstudy_list: list):
    if fieldsofstudy_list is None:
        return []
    return list(set([fieldsofstudy["category"] for fieldsofstudy in fieldsofstudy_list if fieldsofstudy["category"] is not None]))


def process_paper_data_(j: dict, allow_none_year: bool):
    # Process Semantic Scholar data for a paper
    if not allow_none_year and j["year"] is None:
        return None
    return {
        "authors": list(set([int(tmp["authorId"]) for tmp in j["authors"] if tmp["authorId"] is not None])),
        "s2fieldsofstudy": parse_s2fieldsofstudy_(j["s2fieldsofstudy"]),
        **{k: j[k] for k in ["title", "year", "referencecount", "publicationdate"]}
    }


def load_papers(config, ids: list, id_type: str):
    def process_papers_(path: str, ids: list, id_type: str):
        ids = set([str(id) for id in ids])
        id_type = copy.deepcopy(id_type)
        assert id_type in ["CorpusId", "ArXiv"]
        papers = {}
        with gzip.open(path, "rt", encoding="UTF-8") as fin:
            for l in fin:
                j = json.loads(l)
                if j["externalids"][id_type] not in ids:
                    continue
                processed_paper = process_paper_data_(j, allow_none_year=False)
                if processed_paper is None:
                    continue
                if id_type == "ArXiv":
                    processed_paper["arxiv_id"] = j["externalids"]["ArXiv"]
                papers[j["corpusid"]] = processed_paper
        return papers

    return multi_file_query(
        os.path.join(config["data"]["semantic_scholar_path"], "papers", "*.gz"),
        process_papers_,
        config["data"]["n_jobs"],
        ids=ids,
        id_type=id_type
    )


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

    def process_citations_(path: str, ids: list):
        ids = set([int(id) for id in ids])
        citing_papers = defaultdict(lambda: [])
        cited_papers = defaultdict(lambda: [])
        with gzip.open(path, "rt", encoding="UTF-8") as fin:
            for l in fin:
                j = json.loads(l)
                if j["citingcorpusid"] is None or j["citedcorpusid"] is None:
                    continue
                if j["citingcorpusid"] in ids:
                    cited_papers[j["citingcorpusid"]].append(j["citedcorpusid"])
                if j["citedcorpusid"] in ids:
                    citing_papers[j["citedcorpusid"]].append(j["citingcorpusid"])
        # print(f"#citing_papers = {len(citing_papers)}, #cited_papers = {len(cited_papers)}")
        return citing_papers, cited_papers
    
    res = multi_file_query(
        os.path.join(config["data"]["semantic_scholar_path"], "citations", "*.gz"),
        process_citations_,
        config["data"]["n_jobs"],
        ids=[int(id) for id in paper_info.keys()]
    )
    
    for path, data in zip([citing_papers_path, cited_papers_path], res):
        print(f"Saving to {path}")
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)


def process_citing_papers(config: dict):
    print("\nLoading citing papers info from Semantic Scholar")
    output_path = os.path.join(data_dir(config), "citing_paper_info.json")
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        return
    citing_papers = json.load(open(os.path.join(data_dir(config), "citing_papers.json")))
    cited_papers = json.load(open(os.path.join(data_dir(config), "cited_papers.json")))
    paper_ids = [id for tmp in citing_papers.values() for id in tmp] + [id for tmp in cited_papers.values() for id in tmp]
    papers = load_papers(config, paper_ids, "CorpusId")
    print(f"Saving to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(papers, f, indent=4)


def unify_papers(config: dict):
    print("\nUnifying paper data")
    
    output_path = papers_path(config)
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        # return
    
    papers = json.load(open(os.path.join(data_dir(config), "paper_info.json")))
    citing_papers = json.load(open(os.path.join(data_dir(config), "citing_papers.json")))
    cited_papers = json.load(open(os.path.join(data_dir(config), "cited_papers.json")))
    citing_paper_info = json.load(open(os.path.join(data_dir(config), "citing_paper_info.json")))

    all_papers = {paper_id: paper for paper_id, paper in citing_paper_info.items()}
    for paper_id, paper in papers.items():
        all_papers[paper_id] = paper

    for paper_id, paper in all_papers.items():
        if "arxiv_id" not in paper:
            continue
        paper["citing_papers"] = list(set(citing_papers[paper_id])) if paper_id in citing_papers else []
        paper["cited_papers"] = list(set(cited_papers[paper_id])) if paper_id in cited_papers else []

    # Calculate some statistics
    cntrs = {
        "papers": 0,
        "arxiv papers": 0,
        "non-arxiv papers": 0,
        **{f"papers with {key}" : 0 for key in ["cited_papers", "citing_papers"]}
    }
    for paper in all_papers.values():
        cntrs["papers"] += 1
        cntrs["arxiv papers"] += "arxiv_id" in paper
        cntrs["non-arxiv papers"] += "arxiv_id" not in paper
        if "arxiv_id" in paper:
            for key in ["cited_papers", "citing_papers"]:
                cntrs[f"papers with {key}"] += len(paper[key]) > 0
    print(json.dumps(cntrs, indent=4))

    print(f"Saving to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_papers, f, indent=4)



def process_authors(config: dict):
    print("\nLoading citing authors info from Semantic Scholar")
    output_path = authors_path(config)
    if os.path.exists(output_path):
        print(f"{output_path} exists - Skipping")
        return
    papers = json.load(open(papers_path(config)))
    ids = set([str(author_id) for paper in papers.values() for author_id in paper["authors"]])

    def process_authors_(f: str, ids: list):
        ids = set([id for id in ids])
        author_papers = defaultdict(lambda: [])
        with gzip.open(f, "rt", encoding="UTF-8") as fin:
            for l in fin:
                j = json.loads(l)
                if j["authors"] is None:
                    continue
                for author in j["authors"]:
                    if author["authorId"] in ids:
                        author_papers[author["authorId"]].append(process_paper_data_(j, allow_none_year=True))
        return author_papers
    
    authors = multi_file_query(
        os.path.join(config["data"]["semantic_scholar_path"], "papers", "*.gz"),
        process_authors_,
        config["data"]["n_jobs"],
        ids=ids
    )

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


def get_citing_authors(
    citing_papers: list,
    papers: dict,
    paper_year: int,
    citation_years: int
):
    """
Get the set of all authors who cited the paper in the `citation_years` years that follow the paper's publication

Arguments:
    l: list of citations as received from Semantic Scholar
    paper_year: the year of the paper being cited
    citation_years: hom many years after the paper publication to consider as citations
"""
    authors = set()
    for citing_paper_id in citing_papers:
        citing_paper_id = str(citing_paper_id)
        try:
            citing_paper = papers[citing_paper_id]
            year = int(citing_paper["year"])
            if year < paper_year + citation_years:
                for author in citing_paper["authors"]:
                    authors.add(int(author))
        except (TypeError, KeyError):
            # no data for citing_paper_id or year is null
            continue
    return list(authors)


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
    papers = json.load(open(papers_path(config)))
    authors = json.load(open(authors_path(config)))
    max_author_papers = config["data"]["max_author_papers"]
    valid_authors = [int(key) for key, value in authors.items() if value is not None and len(value) < max_author_papers]
    print(f"Removed {len(authors) - len(valid_authors)} authors with more than {max_author_papers} papers")
    print(f"Valid authors: {len(valid_authors)} / {len(authors)}")

    # Create the list of samples
    num_invalid_citing_authors = 0
    num_null_papers = 0
    samples = []

    # Positive samples - all authors who cited the paper
    for paper_id, paper in papers.items():
        if "arxiv_id" not in paper:
            continue
        citing_authors = get_citing_authors(paper["citing_papers"], papers, paper["year"], config["data"]["cication_years"])
        for citing_author in citing_authors:
            citing_author = str(citing_author)
            if citing_author not in authors:
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

    # Negative samples - a random citing author
    # There's a very small chance that this author did cite the paper but what can you do...
    citing_authors = list(set([s["author"] for s in samples]))
    cited_papers = list(set([s["paper"] for s in samples]))
    for paper_id in cited_papers:
        paper = papers[paper_id]
        for author_idx in rng.integers(low=0, high=len(citing_authors), size=config["data"]["num_negative"]):
            samples.append(
                {
                    "paper": paper_id,
                    "year": paper["year"],
                    "author": citing_authors[author_idx],
                    "label": False
                }
            )

    samples = pd.DataFrame.from_records(samples)
    print(samples)
    
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
