# User-Item Fairness Tradeoffs in Recommendations
In the basic recommendation paradigm, the most relevant item is recommended to each user. This may result in some items receiving lower exposure than they “should”; to counter this, several algorithmic approaches have been developed to ensure item fairness. These approaches necessarily degrade recommendations for some users to improve outcomes for items, leading to user fairness concerns. In turn, a recent line of work has focused on developing algorithms for multi-sided fairness, to jointly optimize user fairness, item fairness, and overall recommendation quality. This induces the question: what is the tradeoff between these objectives, and what are the characteristics of (multi-objective) optimal solutions? Theoretically, we develop a model of recommendations with user and item fairness objectives and characterize the solutions of fairness-constrained optimization. We identify two phenomena: (a) when user preferences are diverse, there is “free” item and user fairness; and (b) users whose preferences are misestimated can be especially disadvantaged by item fairness constraints. Empirically, we build a recommendation system for preprints on arXiv and implement our framework, measuring the phenomena in practice and showing how these phenomena inform the design of markets with recommendation systems-intermediated matching.

This package enables the execution of all simulations related to the paper.

## Installation
### Prerequisites
- Python 3.12.4 or newer
- Conda
- G2 cluster access, local run is not supported

### Code installation
Clone the repository, create a conda environment and install the required dependencies:
```bash
pip clone https://github.com/SophieJG/ArXiv_Recommendation_Research_v2
cd ArXiv_Recommendation_Research_v2
conda create --name arxiv python=3.12.4
conda activate arxiv
pip install -r requirements.txt
```

## How to use
The package has a single executable which is `src/runner.py`. Each call must include paths to two YAML configuration files:
1. runner config - specifies which parts of the pipeline to execute
2. data config - specifies dataset attributes such as the number of papers to include in the dataset and number of negative samples to consider for each paper

Additional configuration files may be required:
1. embedder config - required when training/testing/inferencing a paper embedder model - specifies which model to use and the model parameters
2. model config - required when training/testing/inferencing a paper-author utility model - specifies which model to use and the model parameters
3. ranker config - required when evaluating a ranker

For example, a run including all configuration files:
```bash
python src/runner.py configs/runner.yml configs/data_1k.yml --model-config configs/catboost.yml --embedder-config configs/category_embedder.yml --ranker-config configs/utility_ranker.yml
```

### Runner config
The complete pipeline from dataset generation to ranking evaluation comprises many steps:
- Dataset preparation
  - See the Data pipeline section below
- Paper embedding:
  - `train` Fit the paper embedder on the training data
- Paper-author utility model training and evaluation:
  - `train` Train a model and store the trained model to disk
  - `evaluate` Calculate binary classification metrics on the trained model and all data folds
- Ranking evaluation
  - `generate_samples` Prepare the data used by the rankers, this is the complete matrix of all papers in the test set and all authors that interacted with a paper in the test set
  - `generate_predictions` Run inference of the paper-author utility model between all authors and all papers
  - `generate_paper_embeddings` Generate paper embedding for all papers
  - `evaluate` Evaluate the ranker

The runner config file specifies which parts of the pipeline to run. See, for example, a runner config file that executes the paper-author utility model training and evaluation pipeline (`configs/runner.yml`):

```YAML
data:
  kaggle_json_to_parquet: True  # Prerequisite: None
  process_papers: True  # Prerequisite: data/kaggle_json_to_parquet
  process_citations: True  # Prerequisite: data/process_papers
  process_citing_papers: True  # Prerequisite: data/process_citations
  process_authors: True  # Prerequisite: data/process_citing_papers
  get_abstracts: True  # Prerequisite: data/process_citing_papers
  unify_papers: True  # Prerequisite: data/process_authors & data/get_abstracts
  generate_samples: True  # Prerequisite: data/unify_papers
paper_embedding:
  fit: False  # Prerequisite: data_queries
model:
  train: True  # Prerequisite: data_queries
  eval: True  # Prerequisite: model/train
ranking:
  generate_samples: False  # Prerequisite: data_queries
  generate_predictions: False  # Prerequisite: ranking/generate_samples & model/train
  generate_paper_embeddings: False  # Prerequisite: ranking/generate_samples & paper_embedding/fit
  evaluate: False  # Prerequisite: ranking/generate_predictions & ranking/generate_paper_embeddings
```

### Data config
For example (`configs/data_100.yml`):
```YAML
arxiv_json_path: "/home/loaner/workspace/data/arxiv-metadata-oai-snapshot.json"  # Path to the file downloaded from kaggle
base_path: "/tmp/arxiv/papers_100/"  # Path to store data and models
test_is_2020: True  # When this flag is set to true the test set is the year 2020, validation is 2019 and the rest is training. Otherwise the splits are random
start_year: 2011  # papers included in datasets are all papers where year >= `start_year`
end_year: 2021  # papers included in datasets are all papers where year < `end_year`
cication_years: 3  # In order to be considered a positive sample, a citation needs to be in the `cication_years` years after the paper publication
prepare_authors_data_batch_sizes: [100, 25, 5, 1]  # See src/data_queries.py:prepare_authors_data
num_papers: 100  # Number of papers to include in the dataset. Use 0 to include all papers
num_negative: 50  # Number of negative samples for each paper

# Ranking params
top_k: [1, 5, 10]  # Ks to consider for metrics which take top k. Note that the rankers only rank up to the largest k so this has an effect on ranking run time
```

### Model config
The model config file includes the name of the model and all model parameters. Model versions can be used to differ models running with different hyper-parameter sets.

For example (`configs/catboost.yml`):
```YAML
model: "catboost"  # Which model to train
version: "0.0.0"  # Should be used when changing parameters, preprocessing etc.
params:  # Additional model parameters. Catboost doesn't have any
  dummy: 0
```

### Embedder config
The embedder config file includes the name of the embedder and all parameters. Versions can be used to differ embedders running with different hyper-parameter sets.

For example (`configs/category_embedder.yml`):
```YAML
embedder: "category"  # Which embedder to use
version: "0.0.0"  # Should be used when changing parameters, preprocessing etc.
params:  # Additional embedder parameters, category embedder doesn't have any
  dummy: 0
```

### Ranker config
The ranker config file includes the name of the ranker and all parameters. Versions can be used to differ embedders running with different hyper-parameter sets.

For example (`configs/diversity_ranker.yml`):
```YAML
ranker: "diversity"  # Which ranker to use
version: "0.0.0"  # Should be used when changing parameters, preprocessing etc.
params:  # Additional ranker parameters
  lambda: 0.05
```

### Paper-author utility evaluation results
The output of the paper-author evaluation phase is a dictionary specifying the scores for the train, validation and test data folds. For example:
```json
{
    "train": {
        "average_precision_score": 0.822500982935828,
        "roc_auc_score": 0.8899631840083189,
        "accuracy_score": 0.8328675237562885
    },
    "validation": {
        "average_precision_score": 0.697259094647058,
        "roc_auc_score": 0.7709637147393296,
        "accuracy_score": 0.7441860465116279
    },
    "test": {
        "average_precision_score": 0.6106901983686241,
        "roc_auc_score": 0.6927708307018652,
        "accuracy_score": 0.674496644295302
    }
}
```

### Ranking evaluation results
The output of the ranking evaluation phase is a dictionary specifying the scores for the test data. For example:
```json
{
    "MRR (clipped to 10)": 0.16748175144556723,
    "Precision @ 1": 0.0734352773826458,
    "Precision @ 5": 0.10268492176386913,
    "Precision @ 10": 0.18483285917496445,
    "Hit rate @ 1": 0.07617817947062622,
    "Hit rate @ 5": 0.10532140551507885,
    "Hit rate @ 10": 0.1902609978788158,
    "Diversity @ 1": 0.0,
    "Diversity @ 5": 0.5493281259803502,
    "Diversity @ 10": 0.5972639481954042
}
```
Diversity is defined as 1 minus the mean cosine distance between the top-k recommended papers - higher diversity implies a more diverse set.

## Data pipeline
The system is built upon kaggle and Semantic Scholar (SemS) data files that were downloaded to directories specified in data config. The data already exists in the shared folder, so there's no need to download it again. In any case, download instructions can be found in the "Starting from scratch" section below.

Kaggle dataset is a list of arxiv papers and some info on each paper in json format. SemS data is much larger and includes several tables:
- `papers` includes info on all papers including corpus id, arxiv id, author id list, year, title and categories
- `citations` a list of pairs `<citing corpus id, cited corpus is>`
- `abstracts` abstracts for all papers indexed by corpus id
- `authors` (not used) little info for each author indexed by author id. Importantly, the papers written by each author are NOT included in this table

Because the data is split between several tables, we need to do some merge operations. Specifically, we need to query the `papers` table multiple times. High level description of the data pipeline:
1. `data/kaggle_json_to_parquet` - Filter out all non-CS papers from kaggle dataset and randomly select a subset of `num_papers` papers as specified in the data config. Note that the paper ids used in this stage are Arxiv ids
2. `data/process_papers` - Query the `papers` table for the info of all papers that were selected from Arxiv Kaggle dataset. Apart from getting their general info this function provide us with the corpus ids for these papers
3. `data/process_citations` - Using the list of corpus ids of the Arxiv papers we query the `citations` table to get corpus ids for all citing papers. This includes all citing papers, disregarding publication year
4. `data/process_citing_papers` - We are interested only in papers which cited Arxiv within `citation_years` of the Arxiv paper publication date. This filtering is done here by quering the `papers` table along with loading the info for all valid citing papers
5. `data/process_authors` - The step above provided as with a list of all valid citing papers and for each such paper the list of it's authors. For each author, we need to get the list of publications that preceded the publication date of the cited Arxiv paper. This is done here by quering the `papers` table again
6. `data/get_abstracts` - We query the `abstracts` table to get the abstracts of all papers: Arxiv papers, citing papers and papers written by a citing author
7. `data/unify_papers` - This stage unifies all previous paper queries into a single table including all papers, citing author ids and abstracts
8. `data/generate_samples` - Split the papers into training, validation and test folds. All valid citations are used as positive samples. Negative samples are generated by sampling

### Starting from scratch
Download the public ArXiv Dataset available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)

Set the Semantic Scholar API Key as environemnt variable
```bash
export API_KEY=<your key>
```

Run
```
python src/download_semantic_scholar_data.py
```



## Main classes

### The Data class

The `Data` class internally holds three data structures:

1. For each data fold (train/validation/test) a lists of triplets `<paper_id, author_id, label>`. A paper can be in at most one fold. An author can be in more than one fold. The test set could either be the year 2020 or from all years, see data config.
2. A dictionary of authors info. The key is `author_id`
3. A dictionary of papers info. The key is `paper_id`

The data should be accessed using the `Data:get_fold()` function which returns a list of samples. Each sample contains the unstrucuted paper and author info.

In order to guarantee consistency, the author info is "shifted" in time to the year the paper was published. In practice, that implies removing all publications by the author that proceed (are after) the paper.

### Model

All models must inherit the base class `src/models/base_model.py` and override the 4 following functions:

1. `fit(self, data: Data):` Fit the model on the data. In our case the model includes the preprocessing pipeline thus the fitting process entails (i) fitting the preprocessing pipeline; (ii) using the fitted preprocessing pipeline to transform the raw data into features and (iii) training the model on the processed data

2. `predict_proba(self, data: Data, fold: str):` Do inference which entails preprocessing the raw data into features and running the model. `fold` is a string which can be `train`, `validation` or `test`.

3. `save_(self, path: str):` Save the model as files in the folder `path`

4. `load_(self, path: str):` Load the model from the folder `path`

### Embedder

Similarly to model, must inherit the base class `src/paper_embedders/base_embedder.py` and override the 4 functions: `fit`, `embed`, `save_` and `load_`

### Ranker

There is no fit operation for the rankers, must inherit the base class `src/rankers/base_ranker.py` and override `rank(self, proba: pd.DataFrame paper_embeddings: dict)`
