import json

from data import Data
from models.catboost import CatboostModel


def train(
    data: Data
):
    model = CatboostModel()
    model.train(data)


if __name__ == '__main__':
    data = Data()
    train(data)
