import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logger

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logger.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                # "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                # "KNeighborsRegressor": KNeighborsRegressor()
                # "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                # "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    'max_depth': [32, 64, None],
                    # 'max_features':['sqrt','log2'],
                    'min_samples_leaf': [1, 2]
                },
                # "Random Forest":{
                #     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                #     'bootstrap': [True],
                #     'random_state': [42],
                #     'ccp_alpha': [0, 0.1, 0.5, 1, 5, 10],
                #     'max_features':['sqrt','log2',None],
                #     'n_estimators': [64,128,256]
                # },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':[None,'sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Linear Regression":{},
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                # "KNeighborsRegressor":{
                #     'n_neighbors': [5,10,15, 20, 25, 30],
                #     'weights': ['uniform', 'distance'],
                #     'p': [1, 2]
                # },
                # "AdaBoost Regressor":{
                #     'learning_rate':[.1,.01,0.5,.001],
                #     # 'loss':['linear','square','exponential'],
                #     'n_estimators': [8,16,32,64,128,256]
                # }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                              X_test=X_test,y_test=y_test,
                                              models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            #  = {model: score for model, score in sorted(model_report.items(), key=lambda item: item[1], reversed=True)}

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logger.info(f"Best found model on both training and testing dataset")

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=best_model
            # )

            # predicted=best_model.predict(X_test)

            # r2_square = r2_score(y_test, predicted)
            return best_model_score
            
        except Exception as e:
            raise CustomException(e,sys)