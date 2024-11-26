import os
import numpy as np
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
        self.use_bm25_features = params["use_bm25_features"]

    def compute_bm25_score_(self, query_tokens, doc_tokens):
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
    
    def add_bm25_features_(self, X: pd.DataFrame):
        """
Adds BM25 features to the dataset X using the provided samples.
"""
        bm25_max_scores = []
        bm25_avg_scores = []

        for sample in X.itertuples():
            target_tokens = simple_tokenizer(sample.abstract)
            author_abstract = sample.author_abstract
            author_scores = []

            for abstract in author_abstract:
                doc_tokens = simple_tokenizer(abstract)
                if doc_tokens:
                    # Compute BM25 score between target and author paper using compute_bm25_score
                    score = self.compute_bm25_score_(target_tokens, doc_tokens)
                    author_scores.append(score)

            if author_scores:
                bm25_max_scores.append(max(author_scores))
                bm25_avg_scores.append(np.mean(author_scores))
            else:
                bm25_max_scores.append(0.0)
                bm25_avg_scores.append(0.0)

        # Add the BM25 scores as new columns
        X["bm25_max_score"] = bm25_max_scores
        X["bm25_avg_score"] = bm25_avg_scores
        
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
            new_sample = {key: sample[key] for key in ["title", "abstract", "referenceCount", "categories", "label"]}
            new_sample["author_num_papers"] = len(sample["author"]["papers"])
            # Go over the author papers and collect the fiefieldsOfStudy and s2FieldsOfStudy into two lists
            # Additionally, the papers' titles are concatenated into a long string
            new_sample["author_fieldsOfStudy"] = []
            new_sample["author_s2FieldsOfStudy"] = []
            new_sample["author_title"] = ""
            new_sample["author_abstract"] = []
            for p in sample["author"]["papers"]:
                for key in ["fieldsOfStudy", "s2FieldsOfStudy"]:
                    if p[key] is not None:
                        new_sample[f"author_{key}"] += p[key]
                if p["title"] is not None:
                    new_sample["author_title"] += " " + p["title"]
                if p["abstract"] is not None:
                    new_sample["author_abstract"].append(p["abstract"])
            new_sample["is_cited"] = int(sample["author"]["id"]) in sample["cited_authors"]  # Does the paper cites the author
            new_samples.append(new_sample)
        df = pd.DataFrame.from_records(new_samples)
        X = df[[col for col in df.columns if col != "label"]]
        y = df["label"]
        return X, y

    def preprocess_data_(self, X):
        """
Preprocesses the data by adding BM25 features and transforming features using the pipeline.
"""
        if self.use_bm25_features:
            self.add_bm25_features_(X)
        return self.feature_processing_pipeline.transform(X)

    def fit(
        self,
        data: Data
    ):
        """
1. Fit the preprocessing pipeline on the training data
2. Convert the training and validation data using the preprocessing pipeline
3. Train the model on the processed training and validation data
"""
        X_train, y_train = self.load_fold(data, "train")

        if self.use_bm25_features:
            # Build the BM25 model using the training data
            corpus_documents = []
            for paper_abstract, author_abstracts in tqdm(zip(X_train["abstract"], X_train["author_abstract"]), desc="Building BM25 corpus"):
                if paper_abstract:  
                    corpus_documents.append(simple_tokenizer(paper_abstract))
                for author_abstract in author_abstracts:
                    if author_abstract: 
                        corpus_documents.append(simple_tokenizer(author_abstract))
            print("Initializing BM25 model")
            self.bm25_model = BM25Okapi(corpus_documents)
            self.add_bm25_features_(X_train)


        passthrough = ["referenceCount", "author_num_papers", "is_cited"]
        if self.use_bm25_features:
            passthrough += ["bm25_max_score", "bm25_avg_score"]

        self.feature_processing_pipeline = ColumnTransformer([
            ('passthrough', 'passthrough', passthrough),
            ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
            ("title", CountVectorizer(), "title"),
            ("author_fieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_fieldsOfStudy"),
            ("author_s2FieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_s2FieldsOfStudy"),
        ])
        X_train = self.feature_processing_pipeline.fit_transform(X_train)
        print("Training data size:", X_train.shape, " type:", type(X_train))

        X_val, y_val = self.load_fold(data, "validation")
        X_val = self.preprocess_data_(X_val)

        self.model = CatBoostClassifier().fit(
            X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, use_best_model=True
            )

    def predict_proba(self, data: Data, fold: str):
        """
Run inference on a fold
"""
        assert self.model is not None
        X, _ = self.load_fold(data, fold)
        X = self.preprocess_data_(X)
        return self.model.predict_proba(X)[:, 1]

    def save_(
        self,
        path: str
    ):
        assert self.model is not None
        self.model.save_model(os.path.join(path, "model.cbm"), format="cbm")
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.feature_processing_pipeline, f, protocol=5)        
        if self.use_bm25_features:
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
        if self.use_bm25_features:
            # Load BM25 model, author indices, and corpus documents
            with open(os.path.join(path, "bm25_model.pkl"), "rb") as f:
                self.bm25_model = joblib.load(f)

