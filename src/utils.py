import os
import pickle
import sys
from xml.parsers.expat import model
import dill
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException   

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    try:
        model_report = {}
        for i in range(len(list(models))):
           
           
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params[list(models.keys())[i]]
            grid = GridSearchCV(model, param, cv=3)
            grid.fit(X_train, y_train)
            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)
                        
           
            y_pred = model.predict(X_test)
            train_model_score = model.score(X_train, y_train)
            test_model_score = model.score(X_test, y_test)
            model_report[model_name] = test_model_score
        return model_report
    except Exception as e:
        raise CustomException(e, sys)  

      
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)    