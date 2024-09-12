import json
import os
import numpy as np
import pandas as pd


from const import DATA_DIR


def calc_paper_features():
    papers_path = os.path.join(DATA_DIR, "papers.json")
    with open(papers_path) as f:
        papers = json.load(f)
    kaggle_data = pd.read_parquet(os.path.join(DATA_DIR, "kaggle_data.parquet"))
    kaggle_data = kaggle_data.set_index("id", drop=True)
    print(kaggle_data["categories"][:100])
    for key, paper in papers.items():
        arxiv_paper = kaggle_data.loc[key.replace("ARXIV:", "")]
        print(json.dumps(paper, indent=4))
        print(arxiv_paper)
        sys.exit(1)

    
if __name__ == '__main__':
    calc_paper_features()
