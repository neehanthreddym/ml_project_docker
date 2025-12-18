import os
import sys
from dataclasses import dataclass

# Modeling
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# Evaluation
from sklearn.metrics import r2_score

# Custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, print_models_report

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the data into training and testing sets")
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )
            
            # Initialize models
            models = {
                "Linear Regression": LinearRegression(),
                # "Ridge Regression": Ridge(),
                # "Lasso Regression": Lasso(),
                # "Support Vector Regressor": SVR(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                # "K-Nearest Neighbors Regressor": KNeighborsRegressor()
            }

            params={
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['sqrt','log2', None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }            

            logging.info("Evaluating models (with hyperparameter tuning)")
            model_report, trained_models = evaluate_models(X_train=X_train, y_train=y_train, 
                                                X_test=X_test, y_test=y_test, 
                                                models=models, param_grid=params)
            print_models_report(model_report=model_report)
            
            # Best model based on R^2 score
            # Selection logic using train-test gap
            MIN_TEST_R2 = 0.6
            MAX_GAP = 0.05
            GAP_EPS = 1e-6
            
            # keep only models with good test R^2 AND small but non-zero gap
            candidate_models = {
                name: scores
                for name, scores in model_report.items()
                if scores["test_model_score"] >= MIN_TEST_R2
                and GAP_EPS < scores["gap"] <= MAX_GAP
            }

            if not candidate_models:
                raise CustomException(
                    f"No suitable model found. Either all models underperform "
                    f"(test R^2 < {MIN_TEST_R2}) or have gap <= {GAP_EPS} or > {MAX_GAP}."
                )
            
            best_model_name = max(
                candidate_models.keys(),
                key=lambda name: candidate_models[name]["test_model_score"],
            )
            best_model_score = candidate_models[best_model_name]["test_model_score"]
            best_model = trained_models[best_model_name]

            logging.info(
                f"Best model (after gap filter): {best_model_name} | "
                f"Train R^2: {candidate_models[best_model_name]['train_model_score']:.4f}, "
                f"Test R^2: {best_model_score:.4f}, "
                f"Gap: {candidate_models[best_model_name]['gap']:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return f"R^2 score of the best model ({best_model_name}): {r2:.4f}"
        except Exception as e:
            raise CustomException(e, sys)