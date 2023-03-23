import os
import sys

import numpy as np 
import pandas as pd
# import dill
import pickle
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
scoring = {'r2': make_scorer(r2_score), 
           'root_mean_squared_error': make_scorer(mean_squared_error, squared=False)}
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for model_name in models:
            logger.info(f'training model {model_name}')
            model = models[model_name]
            para=param[model_name]

            gs = GridSearchCV(model,para,cv=3, scoring=scoring, refit='r2', n_jobs=2)
            gs.fit(X_train,y_train)
            
            logger.info(f'found the best param for the model {model_name}')
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            print(best_params)
            best_score = gs.best_score_
            print(best_score)

            # model.set_params(**gs.best_params_)
            # model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_test_pred = best_model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)