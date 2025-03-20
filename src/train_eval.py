import json
import numpy as np
from data import Data
from models.catboost import CatboostModel
from models.cocitation_sigmoid import CocitationSigmoidModel
from models.cocitation_logistic import CocitationLogistic
from models.dual_model import DualModel
from matrices.SVD_basic import SVDBasicModel
from util import models_dir, data_dir
import os

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def get_model(config):
    assert config["model"] is not None, "Model config is required"
    model_type = config["model"]["model"]
    match model_type:
        case "catboost": return CatboostModel(config["model"]["params"])
        case "cocitation_sigmoid": return CocitationSigmoidModel(config["model"]["params"])
        case "cocitation_logistic": return CocitationLogistic(config["model"]["params"])
        case "dual_model": return DualModel(config["model"]["params"])
        case "svd_basic": return SVDBasicModel(config["model"]["params"])
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

def is_matrix_model(config):
    """
    Check if the model is a matrix model. This is used to determine how the model needs to be fit.
    """
    model_type = config["model"]["model"]
    matrix_models = ["svd_basic"]
    return model_type in matrix_models

def train(
    config: dict
):
    """
Train a model and store the trained model to disk
"""
    print("\n*****\nTraining")
    data = Data(config)
    model = get_model(config)
    if is_matrix_model(config):
        # Matrix models do not need data folds
        matrix_path = os.path.join(data_dir(config), "citation_matrix.npz") # TODO: update save path for csr matrix
        model.load_matrix(matrix_path)
        model.fit()
        model.save(models_dir(config), config["model"]["model"], config["model"]["version"])
        return
    train_samples = data.get_fold("train")
    validation_samples = data.get_fold("validation")
    model.fit(train_samples, validation_samples)
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
        # if is_matrix_model(config) and fold != "test":
        #     print(f"Skipping {fold} fold for matrix model")
        #     continue
        samples = data.get_fold(fold)
        proba = model.predict_proba(samples)
        labels = [s["label"] for s in samples]
        metrics[fold] = calc_metrics(labels, proba)
    print(json.dumps(metrics, indent=4))
