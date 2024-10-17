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
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Split into tokens based on whitespace
    tokens = text.split()
    return tokens

class CatboostModel(BaseModel):
    def __init__(self, params: dict) -> None:
        self.model = None
        self.feature_processing_pipeline = None
        self.params = params  # Store params to access 'use_bm25_features' and 'use_abstract_word_count'

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
            if self.params.get('use_abstract_vectorizer', False):
                new_sample['abstract'] = sample.get('abstract') or '' 
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
            
            # Compute BM25 features if enabled
            if self.params.get('use_bm25_features', False):
                # Get the target abstract
                target_abstract = sample.get('abstract') or ''
                target_tokens = simple_tokenizer(target_abstract) 
                # Get author's previous abstracts
                previous_abstracts = [p.get('abstract', '') for p in sample['author']['papers'] if p.get('abstract')]
                if previous_abstracts:
                    corpus = [simple_tokenizer(abstract) for abstract in previous_abstracts] 
                    bm25 = BM25Okapi(corpus)
                    scores = bm25.get_scores(target_tokens)
                    new_sample['bm25_max_score'] = max(scores)
                    new_sample['bm25_avg_score'] = sum(scores) / len(scores)
                else:
                    new_sample['bm25_max_score'] = 0.0
                    new_sample['bm25_avg_score'] = 0.0
            else:
                new_sample['bm25_max_score'] = 0.0
                new_sample['bm25_avg_score'] = 0.0

            new_samples.append(new_sample)
        df = pd.DataFrame.from_records(new_samples)
        X = df[[col for col in df.columns if col != "label"]]
        y = df["label"]
        return X, y

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

        # Determine features to pass through
        passthrough_features = ["referenceCount", "author_num_papers", "is_cited"]
        if self.params.get('use_bm25_features', False):
            passthrough_features.extend(['bm25_max_score', 'bm25_avg_score'])

        transformers = [
            ('passthrough', 'passthrough', passthrough_features),
            ("paper_categories", CountVectorizer(analyzer=passthrough_func), "categories"),
            ("title", CountVectorizer(), "title"),
            ("author_fieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_fieldsOfStudy"),
            ("author_s2FieldsOfStudy", CountVectorizer(analyzer=passthrough_func), "author_s2FieldsOfStudy"),
        ]
        
        # Conditionally add abstract vectorizer if enabled
        if self.params.get('use_abstract_vectorizer', False):
            transformers.append(("abstract", CountVectorizer(), "abstract"))  

        self.feature_processing_pipeline = ColumnTransformer(transformers)
        X_train = self.feature_processing_pipeline.fit_transform(X_train)
        print("Training data size:", X_train.shape, " type:", type(X_train))
        X_val, y_val = self.load_fold(data, "validation")
        X_val = self.feature_processing_pipeline.transform(X_val)
        self.model = CatBoostClassifier().fit(
            X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, use_best_model=True
            )

        # Retrieve feature names from the pipeline
        feature_names = self.feature_processing_pipeline.get_feature_names_out()
        
        # Get feature importances from the trained model
        importances = self.model.get_feature_importance()
        
        # Print feature importances
        print("\nFeature Importances:")
        for name, importance in zip(feature_names, importances):
            print(f"Feature: {name}, Importance: {importance}")

    def predict_proba(self, data: Data, fold: str):
        """
Run inference on a fold
"""
        assert self.model is not None
        X, _ = self.load_fold(data, fold)
        X = self.feature_processing_pipeline.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def save_(
        self,
        path: str
    ):
        assert self.model is not None
        self.model.save_model(os.path.join(path, "model.cbm"), format="cbm")
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            joblib.dump(self.feature_processing_pipeline, f, protocol=5)        

    def load_(
        self,
        path: str
    ):
        assert self.model is None
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(path, "model.cbm"), format="cbm")
        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            self.feature_processing_pipeline = joblib.load(f)


