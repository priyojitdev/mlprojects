import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "ElasticNet Regression": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "Support Vector Regressor": SVR(),
                "Multi-layer Perceptron": MLPRegressor(),
                "XGBoost Regressor": XGBRegressor()
            }
            params = {
                "Linear Regression": {},
                "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso Regression": {"alpha": [0.1, 1.0, 10.0]},
                "ElasticNet Regression": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
                "Decision Tree": {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
                "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
                "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
                "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1.0]},
                "Extra Trees": {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
                "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                "Support Vector Regressor": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
                "Multi-layer Perceptron": {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"], "learning_rate_init": [0.001, 0.01]},
                "XGBoost Regressor": {"n_estimators": [100, 200 ], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}   
            }
            

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score above threshold.", sys)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
        
            