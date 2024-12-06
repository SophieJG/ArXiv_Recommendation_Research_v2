import json
from pathlib import Path
import requests
import time
from tqdm import tqdm
import os

QUERY_FAIL_DELAY = 1 
QUERY_FAIL_MULT_DELAY = 2
QUERY_DUMP_INTERVAL = 10

def fetch_papers_data(paper_ids: list, output_dir: str, batch_size: int = 10):
    """
    Fetch paper data from Semantic Scholar API and save in papers.json format
    """
    papers = {}  # paper_id -> paper data
    failed_ids = []  # Track failed papers
    
    # Load existing data if any
    output_path = os.path.join(output_dir, "data/ss_papers.json")
    if os.path.exists(output_path):
        with open(output_path) as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} existing papers")
    
    # Filter out already downloaded papers
    remaining_ids = [pid for pid in paper_ids if pid not in papers]
    print(f"Remaining papers to download: {len(remaining_ids)}")

    api_key = os.environ.get('API_KEY')
    
    def process_paper_response(response):
        if response is None:
            return None 
            
        # Extract expanded set of fields
        return {
            "year": response.get("year"),
            "publicationdate": response.get("publicationDate"),
            "authors": [
                {
                    "authorId": author.get("authorId"),
                    "name": author.get("name")
                } for author in response.get("authors", [])
            ],
            "fieldsOfStudy": response.get("fieldsOfStudy", []),
            "title": response.get("title"),
            "abstract": response.get("abstract"),
            "referenceCount": response.get("referenceCount"),
            "citationCount": response.get("citationCount"),
            "s2FieldsOfStudy": response.get("s2FieldsOfStudy", []),
        }

    fail_delay = 0
    pbar = tqdm(total=len(remaining_ids))
    prev_request_time = 0
    
    for idx in range(0, len(remaining_ids), batch_size):
        batch_ids = remaining_ids[idx:idx + batch_size]
        
        # Rate limiting
        current_time = round(time.time() * 1000)
        throughput_delay = prev_request_time + 1000 - current_time
        throughput_delay = max(0, min(throughput_delay, 1000)) / 1000
        if throughput_delay > 0:
            time.sleep(throughput_delay)
        prev_request_time = current_time

        # Make batch request with retries
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    headers={'x-api-key': api_key} if api_key else {},
                    params={'fields': 'paperId,abstract,publicationDate,authors,fieldsOfStudy,title,referenceCount,year,citationCount,s2FieldsOfStudy'},
                    json={"ids": batch_ids},
                    timeout=30
                )
                
                if response.status_code == 200:
                    break
                
                fail_delay = max(QUERY_FAIL_DELAY, fail_delay * QUERY_FAIL_MULT_DELAY)
                print(f"Retry {retry + 1}/{max_retries} - Sleeping for {fail_delay} seconds")
                time.sleep(fail_delay)
                
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                if retry == max_retries - 1:
                    failed_ids.extend(batch_ids)
                    break
                time.sleep(fail_delay)
                continue

        if response.status_code != 200:
            print(f"Failed to fetch batch starting with {batch_ids[0]}")
            failed_ids.extend(batch_ids)
            continue

        # Process responses
        for paper_id, paper_response in zip(batch_ids, response.json()):
            processed = process_paper_response(paper_response)
            if processed is None:
                failed_ids.append(paper_id)
            else:
                papers[paper_id] = processed

        pbar.update(batch_size)
        fail_delay = max(1, fail_delay // 2)  # Gradually reduce delay on success

        # Periodically save data and report progress
        if idx % (batch_size * QUERY_DUMP_INTERVAL) == 0:
            print(f"\nProgress: {len(papers)}/{len(paper_ids)} papers processed")
            print(f"Failed: {len(failed_ids)} papers")
            with open(output_path, "w") as f:
                json.dump(papers, f, indent=4)
            # Save failed IDs
            with open(os.path.join(output_dir, "data/tmp/failed_papers.json"), "w") as f:
                json.dump(failed_ids, f, indent=4)

    pbar.close()
    
    # Final save
    print(f"\nFinal count: {len(papers)}/{len(paper_ids)} papers processed")
    print(f"Failed papers: {len(failed_ids)}")
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=4)
    with open(os.path.join(output_dir, "data/tmp/failed_papers.json"), "w") as f:
        json.dump(failed_ids, f, indent=4)

def test_output_dir(output_dir: str):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Directory {output_dir} does not exist.")
    output_path = os.path.join(output_dir, "data/test.json")
    with open(output_path, "w") as f:
        f.write("test")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Path to save papers.json")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for API requests")
    args = parser.parse_args()

    test_output_dir(args.output_dir)

    OR_PAPERS_DIR = Path("/Users/thomaswiener/Arxiv-Recommender/ArXiv_Recommendation_Research_v2/Data/goldstandard-reviewer-paper-match/data/papers")
    if not OR_PAPERS_DIR.exists():
        raise FileNotFoundError(f"Directory {OR_PAPERS_DIR} does not exist.")

    ss_paper_ids = [f.stem for f in OR_PAPERS_DIR.glob("*.json")]
    print(f"Found {len(ss_paper_ids)} papers")
    print(f"Paper 1: {ss_paper_ids[0]}")
    assert len(ss_paper_ids) == 3374, "Expected 3374 papers"

    # Fetch and save paper data
    fetch_papers_data(ss_paper_ids, args.output_dir, args.batch_size)

    # Get length of papers.json
    output_path = os.path.join(args.output_dir, "data/ss_papers.json")
    with open(args.output_path, "r") as f:
        papers = json.load(f)
        print(f"Saved {len(papers)} papers")
