import json
import numpy as np
from data import Data
from models.catboost import CatboostModel
from models.mlp import MLPClassifier
from util import models_dir

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def get_model(config):
    return {
        "catboost": CatboostModel(config["model"]["params"]),
        "mlp": MLPClassifier(config["model"]["params"])
    }[config["model"]["model"]]


def train(
    config: dict
):
    """
Train a model and store the trained model to disk
"""
    print("\n*****\nTraining")
    data = Data(config)
    model = get_model(config)
    model.fit(data)
    metrics = {}
    for fold in ["train", "validation", "test"]:
        proba = model.predict_proba(data, fold)
        labels = data.parse_fold(fold)["label"]
        metrics[fold] = calc_metrics(labels, proba)
    print(json.dumps(metrics, indent=4))
    model.save(models_dir(config), config["model"]["model"], config["model"]["version"])


def calc_metrics(labels, proba):
    return {
        "average_precision_score": average_precision_score(labels, proba),
        "roc_auc_score": roc_auc_score(labels, proba),
        "accuracy_score": max([accuracy_score(labels, proba > th) for th in np.arange(0.01, 0.99, 0.01)]),
    }


def eval(
    config: dict
):
    """
Calculate binary classification metrics on the trained model and all data folds
"""
    print("\n*****\nEvaluation")
    data = Data(config)
    model = get_model(config)
    model.load(models_dir(config), config["model"]["model"], config["model"]["version"])
    metrics = {}
    for fold in ["train", "validation", "test"]:
        proba = model.predict_proba(data, fold)
        labels = data.parse_fold(fold)["label"]
        metrics[fold] = calc_metrics(labels, proba)
    print(json.dumps(metrics, indent=4))
