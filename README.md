This is a training and inference pipeline for a recommender system serving pre-prints to authors.

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
  process_references: True  # Prerequisite: data/unify_papers
  generate_samples: True  # Prerequisite: data/process_references
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
For example (`configs/data_1k.yml`):
```YAML
arxiv_json_path: "/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json"  # Path to the file downloaded from kaggle
semantic_scholar_path: "/share/garg/semantic_scholar_Nov2024/2024-11-05"  # Path to the data downloaded from Semantic Scholar

base_path: "/share/garg/arxiv_runs/tom_1k/"  # Path to store data and models
test_is_2020: True  # When this flag is set to true the test set is the year 2020, validation is 2019 and the rest is training. Otherwise the splits are random
start_year: 2011  # papers included in datasets are all papers where year >= `start_year`
end_year: 2021  # papers included in datasets are all papers where year < `end_year`
citation_years: 3  # In order to be considered a positive sample, a citation needs to be in the `citation_years` years after the paper publication
prepare_authors_data_batch_sizes: [100, 25, 5, 1]  # See src/data_queries.py:prepare_authors_data
num_papers: 1000  # Number of papers to include in the dataset. Use 0 to include all papers
num_negative: 50  # Number of negative samples for each paper
max_author_papers: 500  # Drop authors with more publications than `max_author_papers`. We believe that these result from errors in Semantic Scholar's name disambiguation logic
n_jobs: 8  # How many cores to use when querying data

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
The system is built on the base of kaggle and Semantic Scholar (SemS) data files that were downloaded to directories specified in data config. The data already exists in the shared folder, so there's no need to download it again. In any case, full download instructions can be found in the "Starting from scratch" section below.

The Kaggle dataset is a single table of arxiv papers and some info on each paper in json format. SemS data is much larger and includes several tables:
- `papers` includes info on all papers including corpus id, arxiv id, author id list, year, title and categories
- `citations` a list of pairs `<citing corpus id, cited corpus is>`
- `abstracts` abstracts for all papers indexed by corpus id
- `authors` (not used) little info for each author indexed by author id. Importantly, the papers written by each author are NOT included in this table

Because the data is split between several tables, we need to do some merge operations. Specifically, we need to query the `papers` table multiple times. High level description of the data pipeline:
1. `data/kaggle_json_to_parquet` - Filter out all non-CS papers from kaggle dataset and randomly select a subset of `num_papers` papers as specified in the data config. Note that the paper ids used in this stage are Arxiv ids
1. `data/process_papers` - Query the `papers` table for the info of all papers that were selected from Arxiv Kaggle dataset. Apart from getting their general info this function provide us with the corpus ids for these papers
1. `data/process_citations` - Using the list of corpus ids of the Arxiv papers we query the `citations` table to get corpus ids for all citing papers. This includes all citing papers, disregarding publication year
1. `data/process_citing_papers` - We are interested only in papers which cited Arxiv within `citation_years` of the Arxiv paper publication date. This filtering is done here by quering the `papers` table along with loading the info for all valid citing papers. If a paper cites several papers from Arxiv than it is included if it's valid for any of them
1. `data/process_authors` - The step above provided as with a list of all valid citing papers and for each such paper the list of it's authors. Now, we get the list of publications of each author. This is done by querying the `papers` table again. Additionally, we filter out all authors with more than `max_author_papers` publications and the related papers. We believe that these result from errors in Semantic Scholar's name disambiguation logic
1. `data/get_abstracts` - We query the `abstracts` table to get the abstracts of all papers: Arxiv papers, citing papers and papers written by a citing author
1. `data/unify_papers` - This stage unifies all previous paper queries into a single table including all papers, citing paper ids and abstracts
1. `data/process_references` - After unifying the papers, we again querry the `citations` table to get the corpus ids of the papers referenced by each paper in the single unified papers table. We add this reference data into an updated unified papers table.
1. `data/generate_samples` - Split the papers into training, validation and test folds. All valid citing authors are used as positive samples. Negative samples are generated by sampling

### Starting from scratch
As mentioned above, the data already exists in the shared folder, so there's no need to download it again. If you need to do it, for whatever reason:
1. Download the public ArXiv Dataset available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)
1. Set the Semantic Scholar API Key as an environemnt variable: 
`export API_KEY=<your key>`
1. Run `python src/download_semantic_scholar_data.py`

## Main classes

### The Data class

The `Data` class internally holds three data structures:

1. For each data fold (train/validation/test) a lists of triplets `<paper_id, author_id, label>`. A paper can be in at most one fold. An author can be in more than one fold. The test set could either be the year 2020 or from all years, see data config.
1. A dictionary of authors info. The key is `author_id`
1. A dictionary of papers info. The key is `paper_id`

In order to guarantee consistency, the author info is "shifted" in time to the year the paper was published. In practice, that implies removing all publications by the author that proceed (are after) the paper.

For utility function purposes, the data should be accessed using the `Data:get_fold()` function which returns a list of samples. Each sample contains the unstrucuted paper and author info. For ranking method investigations, when the full matrix of papers and authors is needed, the proper manner for data access is by using the `Data:get_ranking_papers()` and `Data:get_ranking_authors()` methods.

### Model

All models must inherit the base class `src/models/base_model.py` and override the 5 following functions:

1. `fit(self, data: Data):` Fit the model on the data. In our case the model includes the preprocessing pipeline thus the fitting process entails (i) fitting the preprocessing pipeline; (ii) using the fitted preprocessing pipeline to transform the raw data into features and (iii) training the model on the processed data

1. `predict_proba(self, samples: list):` Run inference on a list of samples

1. `_save(self, path: str):` Save the model as files in the folder `path`

1. `_load(self, path: str):` Load the model from the folder `path`

1. (Ranking only) `predict_proba_ranking(self, papers: list, authors: list):` Run inference on the cartesian product between all papers and all authors. This functionality exists seperately from `predict_proba` because in many cases inference can be done more efficiently when there's a need to inference on such a cartesian product

### Embedder

Similarly to model, must inherit the base class `src/paper_embedders/base_embedder.py` and override the 4 functions: `fit`, `embed`, `save_` and `load_`

### Ranker

There is no fit operation for the rankers, must inherit the base class `src/rankers/base_ranker.py` and override `rank(self, proba: pd.DataFrame, paper_embeddings: dict)`
