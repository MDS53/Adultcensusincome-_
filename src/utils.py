import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def metrics(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        accuracy={}
        precision={}
        recall={}
        f1_score={}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)
            accuracy[list(models.keys())[i]]=accuracy_score(y_test,y_test_pred)
            prf_models = precision_recall_fscore_support(y_test,y_test_pred)
            precision[list(models.keys())[i]]=prf_models[0]
            recall[list(models.keys())[i]]=prf_models[1]
            f1_score[list(models.keys())[i]]=prf_models[2]
            report[list(models.keys())[i]] =  test_model_score

        return accuracy,precision,recall,f1_score

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
    
        
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        accuracy={}
        precision={}
        recall={}
        f1_score={}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)
            accuracy[list(models.keys())[i]]=accuracy_score(y_test,y_test_pred)
            prf_models = precision_recall_fscore_support(y_test,y_test_pred)
            precision[list(models.keys())[i]]=prf_models[0]
            recall[list(models.keys())[i]]=prf_models[1]
            f1_score[list(models.keys())[i]]=prf_models[2]
            report[list(models.keys())[i]] =  test_model_score

        return report,accuracy,precision,recall,f1_score

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

    