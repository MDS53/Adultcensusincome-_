
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.Data_ingestion import DataIngestion
from src.components.Data_transformation import DataTransformation
from src.components.Model_trainer import ModelTrainer

import json





if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    models,best_model_name,best_model_score=model_trainer.initate_model_training(train_arr,test_arr)
    print("Training values")
    #print(model_report)
    print(models.keys())
    print('*'*100)
    print(best_model_name,best_model_score)
    # Create a dictionary to hold all three variables

    # Create a dictionary to hold all three variables
    variables_to_save = {
        'models': list(models.keys()),
        'best_model_str': str(best_model_name),
        'best_model_score_str': best_model_score
    }

    # Specify the full path where you want to save the JSON file

    # Save variables to the specified location
    with open("variables/Best_model_info.json", 'w') as f:
        json.dump(variables_to_save, f)


