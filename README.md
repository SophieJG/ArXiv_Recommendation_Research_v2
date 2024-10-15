# User-Item Fairness Tradeoffs in Recommendations
In the basic recommendation paradigm, the most relevant item is recommended to each user. This may result in some items receiving lower exposure than they “should”; to counter this, several algorithmic approaches have been developed to ensure item fairness. These approaches necessarily degrade recommendations for some users to improve outcomes for items, leading to user fairness concerns. In turn, a recent line of work has focused on developing algorithms for multi-sided fairness, to jointly optimize user fairness, item fairness, and overall recommendation quality. This induces the question: what is the tradeoff between these objectives, and what are the characteristics of (multi-objective) optimal solutions? Theoretically, we develop a model of recommendations with user and item fairness objectives and characterize the solutions of fairness-constrained optimization. We identify two phenomena: (a) when user preferences are diverse, there is “free” item and user fairness; and (b) users whose preferences are misestimated can be especially disadvantaged by item fairness constraints. Empirically, we build a recommendation system for preprints on arXiv and implement our framework, measuring the phenomena in practice and showing how these phenomena inform the design of markets with recommendation systems-intermediated matching.

This package enables the execution of all simulations related to the paper.

## Installation
### Prerequisites
- Python 3.8 or newer
- Conda
- Semantic Scholar API Key

### Code installation
Clone the repository, create a conda environment and install the required dependencies:
```bash
pip clone https://github.com/SophieJG/ArXiv_Recommendation_Research_v2
cd ArXiv_Recommendation_Research_v2
conda create --name arxiv
conda activate arxiv
pip install -r requirements.txt
```

### Dataset prerequisites
Download the public ArXiv Dataset available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)

Set the Semantic Scholar API Key as environemnt variable (as far as I could tell, the queries run fine without it)
```bash
export API_KEY=<your key>
```

## How to use the package
The package has a single executable which is `src/runner.py`. Each call must include paths to three YAML configuration files:
1. data config - specifies dataset attributes such as the number of papers to include in the dataset and number of negative samples to consider for each paper
2. model config - specifies which model to train/evaluate and the model parameters
3. runner config - specifies which parts of the pipeline to execute

For example:
```bash
python src/runner.py configs/data_1000.yml configs/catboost.yml configs/runner.yml
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
num_negative: 50 # Number of negative samples for each paper
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

### Runner config
The runner config specifies which parts of the pipeline to run. See, for example, a runner config file that executes the complete pipeline (`configs/runner.yml`):

```YAML
data_queries:
  kaggle_json_to_parquet: True
  query_papers: True
  query_authors: True
  generate_samples: True
train: True
eval: True
```

1. Dataset preparation is separated into four phases:
    1. `kaggle_json_to_parquet` Converts the json downloaded from kaggle to parquet and filters not-relevant papers. Currently only CS papers are used.
    2. `query_papers` Query Semantic Scholar to get additional info about all papers
    3. `query_authors` Query Semantic Scholar for all authores who cited a paper from the dataset
    4. `generate_samples` Generate a list of sample triplets: `<paper_id, author_id, label>`. The label can be true (the author cited the paper) or false.
2. `train` Train a model and store the trained model to disk
3. `evaluate` Calculate binary classification metrics on the trained model and all data folds

### Evaluation results
The output of the evaluation phase is a dictionary specifying the scores for the train, validation and test data folds. For example:
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
