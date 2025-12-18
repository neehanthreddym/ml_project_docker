import os
import sys

import numpy as np
import pandas as pd
# import pickle
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, X_test, y_train, y_test, models, param_grid):
    try:
        report = {}
        trained_models = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name  = list(models.keys())[i]
            params=param_grid[list(models.keys())[i]]

            # Special handling for CatBoostRegressor
            if "CatBoost Regressor" in model_name:
                # no GridSearchCV: CatBoost is not sklearn-compatible for tags
                model.fit(X_train, y_train)
                fitted_model = model
            else:
                if params:
                    # tune with GridSearchCV
                    gs = GridSearchCV(model, param_grid=params, cv=3)
                    gs.fit(X_train,y_train)
                    fitted_model = gs.best_estimator_
                else:
                    # no hyperparams to search, just fit directly
                    model.fit(X_train,y_train)
                    fitted_model = model
            
            y_train_pred = fitted_model.predict(X_train)
            y_test_pred = fitted_model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            gap = abs(train_model_score - test_model_score)

            report[model_name] = {
                'train_model_score': train_model_score,
                'test_model_score': test_model_score,
                'gap': gap
            }
            
            # store the trained instance
            trained_models[model_name] = fitted_model
        return report, trained_models
    except Exception as e:
        raise CustomException(e, sys)

def print_models_report(model_report):
    print("\nModel performance summary:")
    try:
        for name, scores in model_report.items():
            train_r2 = scores["train_model_score"]
            test_r2 = scores["test_model_score"]
            gap = scores["gap"]

            print(
                f"- {name:30s} | "
                f"train R^2 = {train_r2:6.4f} | "
                f"test R^2 = {test_r2:6.4f} | "
                f"gap = {gap:6.4f}"
            )
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)