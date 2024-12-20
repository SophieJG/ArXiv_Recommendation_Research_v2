import numpy as np
from models.base_model import BaseModel
from models.catboost import CatboostModel
from models.cocitation_logit import CocitationLogitModel


class DualModel(BaseModel):
    def __init__(self, params: dict) -> None:
        """
        Precondtions: 
        - models have been trained
        - samples are not mutated when calling inner models' predict_proba or predict_proba_ranking
        """
        self.model_info = params
        self.model1 : BaseModel = self._get_model(params['model1'])
        self.model2 : BaseModel = self._get_model(params['model2'])
        self.agg_method = params.get('agg_method', 'max')
        if self.agg_method not in ['mean', 'max']:
            raise ValueError("agg_method must be either 'mean' or 'max'")
        self.loaded = False # flag to indicate that models have been loaded, must be true before running inference

    @staticmethod
    def _get_model(model_info: dict) -> BaseModel:
        def get_model_class(model_name: str) -> type:
            return {
                "catboost": CatboostModel,
                "cocitation_logit": CocitationLogitModel,
                "dual_model": DualModel,
            }[model_name]

        model_class = get_model_class(model_info['name'])
        return model_class(model_info["params"])

    def _aggregate_predictions(self, pred1, pred2):
        if self.agg_method == 'mean':
            return np.mean([pred1, pred2], axis=0)
        else:  # max
            return np.maximum(pred1, pred2)

    def fit(self, train_samples: list, validation_samples: list):
        # No training needed for cocitation model
        print("Models in Dual model have already been trained - skipping")
        pass

    def predict_proba(self, samples: list):
        """
        Precondtions: 
        - models have been trained
        - self.load has been called (self.loaded == True)
        - samples are not mutated when calling inner models' predict_proba
        """
        assert self.model1 is not None and self.model2 is not None
        assert self.loaded == True
        pred1 = self.model1.predict_proba(samples)
        pred2 = self.model2.predict_proba(samples)
        return self._aggregate_predictions(pred1, pred2)

    def predict_proba_ranking(self, papers: list, authors: list):
        """
        Precondtions: 
        - models have been trained
        - self.load has been called (self.loaded == True)
        - samples are not mutated when calling inner models' predict_proba_ranking
        """
        assert self.model1 is not None and self.model2 is not None
        assert self.loaded == True
        pred1 = self.model1.predict_proba_ranking(papers, authors)
        pred2 = self.model2.predict_proba_ranking(papers, authors)
        return self._aggregate_predictions(pred1, pred2)

    def _save(self, path: str):
        # No model parameters to save
        print("Dual model does not require saving - skipping")
        pass

    def _load(self, path: str):
        # No model parameters to load
        raise NotImplementedError("To load Dual model, do not call _load directly, call load instead")

    def load(
        self,
        path: str,
        model: str,
        version: str
    ):  
        self.model1.load(path, self.model_info['model1']['name'], self.model_info['model1']['version'])
        self.model2.load(path, self.model_info['model2']['name'], self.model_info['model2']['version'])
        self.loaded = True