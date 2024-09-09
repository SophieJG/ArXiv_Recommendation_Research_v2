from collections import defaultdict
import json
import os
import time
import pandas as pd
from tqdm import tqdm
import requests

data_dir = "/home/loaner/workspace/data"
start_year = 2011
end_year = 2021


def kaggle_json_to_parquet():
    with open(os.path.join(data_dir, "arxiv-metadata-oai-snapshot.json")) as f:
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
        kaggle_data.to_parquet(os.path.join(data_dir, "kaggle_data.parquet"))
        print(kaggle_data)


def get_citing_authors(j: dict):
    authors = defaultdict(set)
    for citation in j["data"]:
        citation = citation["citingPaper"]
        try:
            year = int(citation["year"])
            for author in citation["authors"]:
                authors[year].add(int(author["authorId"]))
        except TypeError:
            # catch year is null
            continue
    return {year: list(authors) for year, authors in authors.items()}


BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{paper_id}/citations?fields=paperId,authors,title,year&limit=1000"
api_key = os.environ.get("API_KEY")

HEADERS = {
    # 'x-api-key': api_key,
    "Content-Type": "application/json"
}

kaggle_data = pd.read_parquet(os.path.join(data_dir, "kaggle_data.parquet"))
print(f"Filtering relevant years (year < {end_year}) & (year >= {start_year})")
kaggle_data['update_date'] = pd.to_datetime(kaggle_data['update_date'])
kaggle_data['year_updated'] = kaggle_data['update_date'].dt.year
kaggle_data = kaggle_data[(kaggle_data["year_updated"] < end_year) & (kaggle_data["year_updated"] >= start_year)]

citation_data_path = os.path.join(data_dir, "citations.json")
if os.path.exists(citation_data_path):
    with open(citation_data_path) as f:
        citations = json.load(f)
else:
    citations = {}

base_delay = 0.01
delay = base_delay
for idx, paper in tqdm(kaggle_data.iterrows(), "Querying citations"):
    url = BASE_URL.format(paper_id=paper["id"])
    paper_id = paper["id"]
    if paper_id in citations:
        continue
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        if "error" in response.text:
            citations[paper_id] = json.loads(response.text)
            continue
        delay *= 2
        print(f" - Sleeping for {delay} seconds")
        print(json.loads(response.text))
        time.sleep(delay)
        continue
    delay = max(base_delay, delay / 2)
    citations[paper_id] = get_citing_authors(json.loads(response.text))
    if idx % 100 == 0:
        print("Dumping citations to file")
        with open(citation_data_path, 'w') as f:
            json.dump(citations, f)
    time.sleep(base_delay)

with open(citation_data_path, 'w') as f:
    json.dump(citations, f)

print(f"Got citations for {len(citations)} papers out of {len(kaggle_data)}")