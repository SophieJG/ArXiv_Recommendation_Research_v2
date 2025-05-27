import json
import numpy as np
from data import Data
from models.catboost import CatboostModel
from models.cocitation_sigmoid import CocitationSigmoidModel
from models.cocitation_logistic import CocitationLogistic
from models.specter2_basic import Specter2Basic
from models.dual_model import DualModel
from util import models_dir, model_version_path
from models.cosine_sim import CosineSimilarityModel

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def get_model(config):
    assert config["model"] is not None, "Model config is required"
    model_type = config["model"]["model"]
    model_params = config["model"]["params"]

    match model_type:
        case "catboost":
            return CatboostModel(model_params)
        case "cocitation_sigmoid":
            return CocitationSigmoidModel(model_params)
        case "cocitation_logistic":
            return CocitationLogistic(model_params)
        case "dual_model":
            params = model_params.copy()
            params["vector_db_dir"] = config["data"]["vector_db_dir"]
            params["vector_collection_name"] = config["data"]["vector_collection_name"]
            return DualModel(params)
        case "specter2_basic":
            params = model_params.copy()
            params["load_path"] = model_version_path((
                models_dir(config),
                config["model"]["model"],
                config["model"]["version"]
            ))
            return Specter2Basic(params)
        case "cosine_sim":
            params = model_params.copy()
            params["vector_db_dir"] = config["data"]["vector_db_dir"]
            params["vector_collection_name"] = config["data"]["vector_collection_name"]
            return CosineSimilarityModel(params)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def train(
    config: dict
):
    """
Train a model and store the trained model to disk
"""
    print("\n*****\nTraining")
    data = Data(config)
    model = get_model(config)
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
        samples = data.get_fold(fold)
        proba = model.predict_proba(samples)
        labels = [s["label"] for s in samples]
        metrics[fold] = calc_metrics(labels, proba)
    print(json.dumps(metrics, indent=4))
