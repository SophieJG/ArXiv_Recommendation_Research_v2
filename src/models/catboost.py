import os
import pandas as pd
import joblib
import re
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from catboost import CatBoostClassifier

from data import Data
from models.base_model import BaseModel
from rank_bm25 import BM25Okapi
from util import passthrough_func


def simple_tokenizer(text):
    """
    A simple tokenizer that:
    - Converts text to lowercase
    - Removes punctuation
    - Splits text into tokens based on whitespace
    """
    if not text:
        return []
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r"[^\w\s]", "", text)
    # Split into tokens based on whitespace
    tokens = text.split()
    return tokens


class CatboostModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = None
        self.feature_processing_pipeline = None
        self.params = params  # Store params to access 'use_bm25_features' and 'use_abstract_word_count'

    def compute_bm25_score(self, query_tokens, doc_tokens):
        """
        Compute the BM25 score between a query and a document.

        Parameters:
        - query_tokens: List of tokens from the target abstract.
        - doc_tokens: List of tokens from one of the author's documents.

        Returns:
        - score: BM25 similarity score between the query and the document.
        """
        score = 0.0
        for term in query_tokens:
            if term in self.bm25_model.idf:
                idf = self.bm25_model.idf[term]
                tf = doc_tokens.count(term)
                dl = len(doc_tokens)
                avgdl = self.bm25_model.avgdl
                k1 = self.bm25_model.k1
                b = self.bm25_model.b
                numerator = idf * tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                score += numerator / denominator
        return score

    def load_fold(
        self,
        data: Data,
        fold: str
    ):
        """
Loads a fold and converts it to pandas dataframe. Some non-trivial data processing is required to convert the paper and author
dictionaries to rows in a dataframe
"""
        samples = data.get_fold(fold)
        new_samples = []
        for sample in tqdm(samples, "Converting samples to dataframe"):
            # Copy fields from the data dictionaries 
            new_sample = {key: sample[key] for key in ["title", "referenceCount", "categories", "label"]}
            # Add abstract if use_abstract_vectorizer is enabled
            if self.params.get("use_abstract_vectorizer", False):
                new_sample["abstract"] = sample.get("abstract") or ""
            new_sample["author_num_papers"] = len(sample["author"]["papers"])
            # Go over the author papers and collect the fiefieldsOfStudy and s2FieldsOfStudy into two lists
            # Additionally, the papers' titles are concatenated into a long string
            new_sample["author_fieldsOfStudy"] = []
            new_sample["author_s2FieldsOfStudy"] = []
            new_sample["author_title"] = ""
            for p in sample["author"]["papers"]:
                for key in ["fieldsOfStudy", "s2FieldsOfStudy"]:
                    if p[key] is not None:
                        new_sample[f"author_{key}"] += p[key]
                if p["title"] is not None:
                    new_sample["author_title"] += " " + p["title"]
            new_sample["is_cited"] = int(sample["author"]["id"]) in sample["cited_authors"]  # Does the paper cites the author
            new_samples.append(new_sample)
        df = pd.DataFrame.from_records(new_samples)
        X = df[[col for col in df.columns if col != "label"]]
        y = df["label"]
        return X, y

    def process_data(self, X, samples):
        """
        Processes the data by adding BM25 features and transforming features using the pipeline.
        """
        if self.params.get("use_bm25_features", False):
            X = self.add_bm25_features(X, samples)
        X_transformed = self.feature_processing_pipeline.transform(X)
        return X_transformed

    def add_bm25_features(self, X, samples):
        """
        Adds BM25 features to the dataset X using the provided samples.
        """
        bm25_max_scores = []
        bm25_avg_scores = []

        for idx, sample in enumerate(samples):
            target_abstract = sample.get("abstract", "")
            target_tokens = simple_tokenizer(target_abstract)
            author_papers = sample["author"]["papers"]
            author_scores = []
            
            for paper in author_papers:
                doc_abstract = paper.get("abstract", "")
                doc_tokens = simple_tokenizer(doc_abstract)
                if doc_tokens:
                    # Compute BM25 score between target and author paper using compute_bm25_score
                    score = self.compute_bm25_score(target_tokens, doc_tokens)
                    author_scores.append(score)

            if author_scores:
                bm25_max_scores.append(max(author_scores))
                bm25_avg_scores.append(sum(author_scores) / len(author_scores))
            else:
                bm25_max_scores.append(0.0)
                bm25_avg_scores.append(0.0)

        X = X.copy()  # Make a copy to avoid modifying the original DataFrame
        # Add the BM25 scores as new columns
        X["bm25_max_score"] = bm25_max_scores
        X["bm25_avg_score"] = bm25_avg_scores
        return X

    def fit(
        self,
        data: Data
    ):
        """
1. Fit the preprocessing pipeline on the training data
2. Convert the training and validation data using the preprocessing pipeline
3. Train the model on the processed training and validation data
"""
        # Build the BM25 model using the training data
        corpus_documents = []
        for sample in tqdm(data.get_fold("train"), desc="Building BM25 corpus"):
            target_abstract = sample.get("abstract", "")
            target_tokens = simple_tokenizer(target_abstract)
            if target_tokens:
                corpus_documents.append(target_tokens)
            for paper in sample["author"]["papers"]:
                paper_abstract = paper.get("abstract", "") or ""
                paper_tokens = simple_tokenizer(paper_abstract)
                if paper_tokens:
                    corpus_documents.append(paper_tokens)

        # Initialize BM25 model with the training corpus
        self.bm25_model = BM25Okapi(corpus_documents)
        print("BM25 model initialized on training corpus.")

        # Load the training data
        X_train, y_train = self.load_fold(data, "train")
        samples_train = data.get_fold("train")

        # Add BM25 features to X_train before fitting the pipeline
        if self.params.get("use_bm25_features", False):
            X_train = self.add_bm25_features(X_train, samples_train)

        # Adjust the pipeline to include BM25 features
        passthrough_features = ["referenceCount", "author_num_papers", "is_cited"]
        if self.params.get("use_bm25_features", False):
            passthrough_features.extend(["bm25_max_score", "bm25_avg_score"])

        transformers = [
            ("passthrough", "passthrough", passthrough_features),
            ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
            ("title", CountVectorizer(), "title"),
            ("author_fieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_fieldsOfStudy"),
            ("author_s2FieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_s2FieldsOfStudy"),
        ]

        # Conditionally add abstract vectorizer if enabled
        if self.params.get("use_abstract_vectorizer", False):
            transformers.append(("abstract", CountVectorizer(), "abstract"))

        self.feature_processing_pipeline = ColumnTransformer(transformers)
        self.feature_processing_pipeline.fit(X_train)
        X_train = self.feature_processing_pipeline.transform(X_train)
        print("Training data size:", X_train.shape, " type:", type(X_train))
        X_val, y_val = self.load_fold(data, "validation")
        samples_val = data.get_fold("validation")
        X_val = self.process_data(X_val, samples_val)
        self.model = CatBoostClassifier().fit(
            X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, use_best_model=True
        )

        # Retrieve feature names from the pipeline and print
        feature_names = self.feature_processing_pipeline.get_feature_names_out()
        importances = self.model.get_feature_importance()
        print("\nFeature Importances:")
        for name, importance in zip(feature_names, importances):
            print(f"Feature: {name}, Importance: {importance}")

    def predict_proba(self, data: Data, fold: str):
        """
Run inference on a fold
"""
        assert self.model is not None
        X, _ = self.load_fold(data, fold)
        samples = data.get_fold(fold)
        X_processed = self.process_data(X, samples)
        return self.model.predict_proba(X_processed)[:, 1]

    def save_(
        self,
        path: str
    ):
        assert self.model is not None
        self.model.save_model(os.path.join(path, "model.cbm"), format="cbm")
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.feature_processing_pipeline, f, protocol=5)
        # Save BM25 model and author indices
        with open(os.path.join(path, "bm25_model.pkl"), "wb") as f:
            joblib.dump(self.bm25_model, f)

    def load_(
        self,
        path: str
    ):
        assert self.model is None
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(path, "model.cbm"), format="cbm")
        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            self.feature_processing_pipeline = joblib.load(f)
        # Load BM25 model, author indices, and corpus documents
        with open(os.path.join(path, "bm25_model.pkl"), "rb") as f:
            self.bm25_model = joblib.load(f)
