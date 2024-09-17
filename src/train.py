import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

from data import Data
from models.catboost import CatboostModel


def calc_metrics(labels, proba):
    return {
        "average_precision_score": average_precision_score(labels, proba),
        "roc_auc_score": roc_auc_score(labels, proba),
        "accuracy_score": max([accuracy_score(labels, proba > th) for th in np.arange(0.01, 0.99, 0.01)]),
    }


def train(
    data: Data
):
    model = CatboostModel()
    model.fit(data)
    model.save("/tmp/")

    model = CatboostModel()
    model.load("/tmp/")
    metrics = {}
    for fold in ["train", "validation", "test"]:
        proba = model.predict_proba(data, fold)
        labels = data.parse_fold(fold)["label"]
        metrics[fold] = calc_metrics(labels, proba)
    print(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    data = Data()
    train(data)
