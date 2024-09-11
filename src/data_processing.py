import json
import os
import numpy as np
import pandas as pd

from const import DATA_DIR
NUM_NEGATIVE = 10


def generate_sample(
        paper: dict,
        author: dict,
        label: bool
):
    return {
        "year": paper["year"],
        "referenceCount": paper["referenceCount"],
        "isOpenAccess": paper["isOpenAccess"],
        "label": label
    }


def generate_samples():
    rng = np.random.default_rng(seed=42)
    papers_path = os.path.join(DATA_DIR, "papers.json")
    with open(papers_path) as f:
        papers = json.load(f)
    authors_path = os.path.join(DATA_DIR, "authors.json")
    with open(authors_path) as f:
        authors = json.load(f)
    author_keys = list(authors.keys())

    samples = []
    for paper in papers.values():
        authors_and_label = []
        # Positive samples
        for citing_author in paper["citing_authors"]:
            authors_and_label.append((authors[str(citing_author)], True))
        # Negative samples
        for author_idx in rng.integers(low=0, high=len(authors), size=NUM_NEGATIVE):
            authors_and_label.append((
                authors[author_keys[author_idx]],
                False
                ))
        for author, label in authors_and_label:
            samples.append(generate_sample(paper, author, label))

    data = pd.DataFrame.from_records(samples)
    print(data)
