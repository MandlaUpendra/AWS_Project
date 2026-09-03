import os
import sys
from pathlib import Path
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object,evaluate_models,save_object

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    model_trainer_filepath: str = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting train and test into X and y variables")
            X_train,y_train,X_test,y_test = (train_array[:,:-1],
                                             train_array[:,-1],
                                             test_array[:,:-1],
                                             test_array[:-1])

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'XGBoosting Regression': XGBRegressor(),
                'CatBoosting Regression': CatBoostRegressor(),
                'AdaBoost Regression': AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion': ['squared_error','friedman_mse','absolute_error','poisson']
                },
                'Gradient Boosting': {
                    'learing_rate': [.1,.01,.05,.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear_regression": {},
                "XGBoosting Regression": {
                    'learning_rate': [.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regression": {
                    'depth': [6,8,10],
                    'learning_rate': [ 0.01,0.05,.001],
                    'iterations': [30,50,100]
                },
                "AdaBoost Regression": {
                    'learning_rate': [.1,.01,0.5,0.1],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_scores = dict(sorted(model_report.items(),key=lambda x: x[1],reverse=True))

            best_model_name = list(best_model_scores.keys())[0]
            best_model_score = list(best_model_scores.values())[0]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("NO best model found!")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.model_trainer_filepath,
                obj= best_model
            )

            predicted = best_model.predict(X_test)

            r2_squared = r2_score(y_test,predicted)
            return r2_squared

        except Exception as e:
            raise CustomException(e,sys)