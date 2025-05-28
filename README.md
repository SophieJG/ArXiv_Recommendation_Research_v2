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

*Note: For those having trouble installing torch version 2.6.0 (or finding a version compatible with adapters version 1.1.0), you may use a different torch version (e.g. the latest) and choose not to install adapters. In this case, you will not be able to run code/models that use the Specter2 model from Huggingface (at the time of writing, this just means you will be unable to use/import ```models/specter2_basic.py```)*

## How to use
The package has a single executable which is `src/runner.py`. Each call must include paths to two YAML configuration files:
1. runner config - specifies which parts of the pipeline to execute
2. data config - specifies dataset attributes such as the number of papers to include in the dataset and number of negative samples to consider for each paper

Additional configuration files may be required:
1. embedder config **(WARNING: DEPRECATED)** - (Previously) required when training/testing/inferencing a paper embedder model - specifies which model to use and the model parameters. *Any embedder models should use the model config now*
2. model config - required when training/testing/inferencing a paper-author utility model - specifies which model to use and the model parameters
3. ranker config - required when evaluating a ranker or evaluating a model using a ranker for ranking evaluations

For example, a run including all configuration files:
```bash
python src/runner.py configs/runner.yml configs/data_1k.yml --model-config configs/catboost.yml --embedder-config configs/category_embedder.yml --ranker-config configs/utility_ranker.yml
```

### Runner config
The complete pipeline from dataset generation to ranking evaluation comprises many steps:
- Dataset preparation
  - See the Data pipeline section below
- Paper embedding **(WARNING: DEPRECATED)**: *Previously used for models using the embedder-config*
  - `train` Fit the paper embedder on the training data. 
- Paper-author utility model training and evaluation:
  - `train` Train a model and store the trained model to disk
  - `evaluate` Calculate binary classification metrics on the trained model and all data folds
- Ranking evaluation
  - `generate_samples` Prepare the data used by the rankers, this is the complete matrix of all papers in the test set and all authors that interacted with a paper in the test set
  - `generate_predictions` Run inference of the paper-author utility model between all authors and all papers
  - `generate_paper_embeddings` **(WARNING: DEPRECATED)** Generate paper embedding for all papers. *Previously used for models using the embedder-config*
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
  process_paper_embedding: False  # Prerequisite: data/unify_papers. Only run this if you want to (re-)store paper embeddings in a Chromadb collection, or haven't done so yet for a specific embedding type. See the Data config for more details.
  generate_samples: True  # Prerequisite: data/process_references
paper_embedding: # **(WARNING: DEPRECATED)**
  fit: False  # Prerequisite: data_queries
model:
  train: True  # Prerequisite: data_queries
  eval: True  # Prerequisite: model/train
ranking:
  generate_samples: False  # Prerequisite: data
  generate_predictions: False  # Prerequisite: ranking/generate_samples & model/train
  generate_paper_embeddings: False  #  **(WARNING: DEPRECATED)** Prerequisite: ranking/generate_samples & paper_embedding/fit
  evaluate: False  # Prerequisite: ranking/generate_predictions
```

### Data config
For example (`configs/data_1k.yml`):
```YAML
arxiv_json_path: "/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json"  # Path to the file downloaded from kaggle
semantic_scholar_path: "/share/garg/semantic_scholar_Nov2024/2024-11-05"  # Path to the data downloaded from Semantic Scholar

base_path: "/share/garg/arxiv_runs/tom_1k/"  # Path to store data and models
vector_db_dir: "/share/garg/arxiv_vector_databases/spector_store"  # Path to the vector database. This is used to efficiently store the embeddings of the papers. Set up using data/process_paper_embedding
vector_collection_name: "Semantic_Scholar_embeddings" # Name of the collection in the vector database where the embeddings are stored. Set up using data/process_paper_embedding
embedding_type: "basic"  # Type of embedding to use. Current options are "basic" (for semantic scholar specter2 embeddings), "queue", or "gte". If nothing is provided, the basic implementation is used.
test_is_2020: True  # When this flag is set to true the test set is the year 2020, validation is 2019 and the rest is training. Otherwise the splits are random
start_year: 2011  # papers included in datasets are all papers where year >= `start_year`
end_year: 2021  # papers included in datasets are all papers where year < `end_year`
citation_years: 3  # In order to be considered a positive sample, a citation needs to be in the `citation_years` years after the paper publication
prepare_authors_data_batch_sizes: [100, 25, 5, 1]  # See src/data_queries.py:prepare_authors_data
num_papers: 1000  # Number of papers to include in the dataset. Use 0 to include all papers
num_negative: 50  # Number of negative samples for each paper in model training
num_negative_ranking: 100  # Number of negative samples for each author in the ranking task involving negative sampling
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
// Found 82549 unique authors with positive papers from 346361 pairs.
// Ranking results using all samples (Total number of papers: 7758):
{
    "MRR @ 101 (clipped to 0)": 0.11803601810032881,
    "Precision @ 1": 0.03099078706898294,
    "Precision @ 5": 0.07998591065391311,
    "Precision @ 10": 0.11414968775352884,
    "Precision @ 101": 0.3274935688486868,
    "Hit rate @ 1": 0.06759621558104884,
    "Hit rate @ 5": 0.1622793734630341,
    "Hit rate @ 10": 0.21900931567917237,
    "Hit rate @ 101": 0.48594168312153996
}

// Ranking results using 100 random negative samples (evaluated on 82549 authors):
{
    "MRR @ 101 (clipped to 0) (100 neg)": 0.37452082319524854,
    "Precision @ 1 (100 neg)": 0.2536554046687422,
    "Precision @ 5 (100 neg)": 0.4982737525590861,
    "Precision @ 10 (100 neg)": 0.6242474166858472,
    "Precision @ 101 (100 neg)": 1.0,
    "Hit rate @ 1 (100 neg)": 0.2536554046687422,
    "Hit rate @ 5 (100 neg)": 0.4982737525590861,
    "Hit rate @ 10 (100 neg)": 0.6242474166858472,
    "Hit rate @ 101 (100 neg)": 1.0
}
```
There are 2 ranking evaluation portions as seen above:
1. The first section ranks and computes the top_k metrics over ALL (positive and negative) ranking papers in the dataset. This means that there can be multiple positive papers corresponding to each author when ranking for an author, and scores are calculated using the highest ranking paper. Note that the scores in the first section are correlated with the size of the dataset.
1. The second section samples 1 positive paper and `num_negative_ranking` (from data config) negative papers for each author when calculating metrics. It then ranks and computes the top_k metrics over these samples for each author.

All metrics are averaged over the authors


Diversity (not yet updated since the deprecation of the old embedder config) is defined as 1 minus the mean cosine distance between the top-k recommended papers - higher diversity implies a more diverse set.

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
1. `data/process_paper_embedding` - Extract and paper embeddings for all the papers in the unified table, and store them in a vector database. Currently supports three embedding types, but more can be added: "basic" (Semantic Scholar's Specter2 embeddings), "queue" (parallel processing of Specter2), and "gte" (General Text Embeddings generated from paper titles and abstracts). This is written in a way such that different datasets (e.g. data_50k, data_all) can share a vector collection for each embedding type; therefore, once the embeddings have been added to the database once, this section of the pipeline doesn't need to be called again. The vector database can later be accessed my models that wish to use embeddings. 
1. `data/generate_samples` - Split the papers into training, validation and test folds. All valid citing authors are used as positive samples. Negative samples are generated by sampling. Note that currently we do not ensure that negative authors sampled have papers published before the paper they're paired with, which could be a future step to take to improve training/testing distribution

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

For the papers info, embeddings data is not included in the dictionary to save memory. Embeddings should be loaded from within the model that is using them. # TODO update

In order to guarantee consistency, the author info is "shifted" in time to the year the paper was published. In practice, that implies removing all publications by the author that proceed (are after) the paper.

For utility function purposes, the data should be accessed using the `Data:get_fold()` function which returns a list of samples. Each sample contains the unstrucuted paper and author info. For ranking method investigations, when the full matrix of papers and authors is needed, the proper manner for data access is by using the `Data:get_ranking_papers()` and `Data:get_ranking_authors()` methods.

### EmbeddingDatabase Class (ChromaDB-based)

The `EmbeddingDatabase` class provides persistent storage for paper embeddings used throughout the pipeline, without having to takeup runtime memory (which is why we separate it from the Data class). Embedding databases are setup in the `process_paper_embedding` functions and accessed by models that require embeddings.

**Key features for this codebase**:
- **Shared collections**: Multiple datasets (e.g., `data_1k`, `data_50k`) can share the same embedding collection, avoiding recomputation when papers overlap between datasets
- **Pipeline integration**: Used in `data/process_paper_embedding` to store embeddings from Semantic Scholar's Specter2 files or generated GTE embeddings, for example. Use the `embedding_type` field in the data config to select which embeddings you want to store.
- **Model access**: Models retrieve embeddings on-demand using paper corpus IDs without loading all embeddings into memory
- **Batch processing**: Handles the large-scale embedding storage with automatic batching (5000 embeddings per batch)

**Configuration**: Set via `vector_db_dir` and `vector_collection_name` in the data config. The database directory is shared across experiments, while collection names typically correspond to embedding types (e.g., "Semantic_Scholar_embeddings").

**REMEMBER**: When you create a new Chromadb Directory and/or collection on G2, make sure to remember to provide others with reading (and writing if you want as well) permissions for the directory/collection. By default Chromadb will only give the person who created the collection CRUD permissions.

**Concurrency**: Multiple processes can read from the same collection simultaneously, but only one process should write to a collection at a time to avoid conflicts, as Chromadb cannot always handle simultaneous writes to a collection.

**Methods used in pipeline**:
- `store_embeddings()`: Called during `process_paper_embedding` to set up the embeddings collection
- `get_embeddings()`: Called by models during training/inference to retrieve paper embeddings
- `has_embedding()`: Used to check if embeddings already exist before processing

*WARNING: Be careful when setting `data/process_paper_embedding` to True when you want to store embeddings in a collection: Make sure that you set the `vector_db_dir`, `vector_collection_name`, and `embedding_type` correctly, so that you don't accidentally overwrite a different chromadb collection. Additionally, if you choose to use the same model (e.g., CosineSimilarity) with different embedding types, make sure to set a different model version in the model config for each embedding type used*

### Model

All models must inherit the base class `src/models/base_model.py` and override the 5 following functions:

1. `fit(self, data: Data):` Fit the model on the data. In the catboost case, for example, the model includes the preprocessing pipeline thus the fitting process entails (i) fitting the preprocessing pipeline; (ii) using the fitted preprocessing pipeline to transform the raw data into features and (iii) training the model on the processed data

1. `predict_proba(self, samples: list):` Run inference on a list of samples

1. `_save(self, path: str):` Save the model as files in the folder `path`

1. `_load(self, path: str):` Load the model from the folder `path`

1. (Ranking only) `predict_proba_ranking(self, papers: list, authors: list):` Run inference on the cartesian product between all papers and all authors. This functionality exists seperately from `predict_proba` because in many cases inference can be done more efficiently when there's a need to inference on such a cartesian product

### Embedder **(WARNING: DEPRECATED)**

Similarly to model, must inherit the base class `src/paper_embedders/base_embedder.py` and override the 4 functions: `fit`, `embed`, `save_` and `load_`

### Ranker

There is no fit operation for the rankers, must inherit the base class `src/rankers/base_ranker.py` and override `rank(self, proba: pd.DataFrame, paper_embeddings: dict)`
