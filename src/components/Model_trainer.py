# Basic Import
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import catboost as cb
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestClassifier
from src.utils import save_object
from src.utils import evaluate_model,metrics
from sklearn.utils import column_or_1d


from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def model_params(self,models,best_model_name,best_model_score):
        self.models=models
        self.best_model_name=best_model_name
        self.best_model_score=best_model_score
        return self.models,self.best_model_name,self.best_model_score
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'LogisticRegression':LogisticRegression(),
            'CatBoostClassifier' : cb.CatBoostClassifier(),
            'XGBClassifier' : xgb.XGBClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            
        }
            y_train=pd.DataFrame(y_train,columns=['Income'])
            y_train=y_train.replace({' <=50K': 0, ' >50K': 1})

            y_train = column_or_1d(y_train, warn=True)
           
            
            y_test=pd.DataFrame(y_test,columns=['Income'])
            y_test=y_test.replace({' <=50K': 0, ' >50K': 1})

            y_test = column_or_1d(y_test, warn=True)
          
            print(type(y_train),type(y_test))
            #accuracy,precision,recall,f1_score=metrics(X_train,y_train,X_test,y_test,models)
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            """print("Accuracy",accuracy)
            print("Precision",precision)
            print("Recall",recall)
            print("F1_score",f1_score)"""
            
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            print(f"Model report :{model_report}")

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
            return models,best_model_name,best_model_score
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
        
