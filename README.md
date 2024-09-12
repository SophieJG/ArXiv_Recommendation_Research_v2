# User-Item Fairness Tradeoffs in Recommendations

'''bash
export API_KEY=<your key>
'''

## Overview
This repository houses the source code for a research project aimed at exploring user-item fairness tradeoffs in recommendation systems. The project encompasses both theoretical frameworks and empirical evaluations to assess and improve fairness in recommendations.

## Repository Structure
- **explore_data.py**: Script for getting main and sub categories information, and separate train and test datasets.
- **get_authors_papers.py**: Script to fetch all authors and their published papers from Semantic Scholar.
- **get_paper_citations.py**: Retrieves citation data for the recommended papers from Semantic Scholar.
- **get_paper_details.py**: Fetches information about papers such as semantic scholar ID etc for the citation/references.
- **get_references.py**: Collects references for the recommended papers from Semantic Scholar.
- **import_metadata.py**: Script for importing and processing metadata from Kaggle.
- **model_evaluation.py**: Contains functions to evaluate the recommendation model.
- **requirements.txt**: Lists all the dependencies required to run the scripts.
- **sentence_transformer_authors.py**: Get recommendations using Sentence Transformer and cosine similarity.
- **stopwords.txt**: Text file containing stopwords used in text processing.
- **tfidf_authors.py**: Get recommendations using TF-IDF embeddings and cosine similarity.
- **utils.py**: Utility functions used across the project.


## Installation

### Prerequisites
- Python 3.8 or newer
- Conda
- Semantic Scholar API Key

### Setup
Clone the repository and install the required dependencies:
```bash
conda create --name arxiv
conda activate arxiv
pip install -r requirements.txt
```

### Data preparation


### ArXiv Dataset
Download the public ArXiv Dataset available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data) and 

#### Step 1
```bash
python3 import_metadata.py
```
#### Step 2
```bash
python3 explore_data.py
```
#### Step 3
```bash
python3 get_authors_papers.py
python3 get_paper_citations.py
python3 get_paper_details.py
python3 get_references.py
```
#### Step 4
```bash
python3 tfidf_authors.py
```
