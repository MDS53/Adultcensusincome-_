import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            tnf1=ColumnTransformer(transformers=[
                ("Handling Int-Missing values",SimpleImputer(),[0,2,3,8,9,10]),
                ("Handling Cat-Missing values",SimpleImputer(strategy='most_frequent'),[1,4,5,6,7,11])
                 ],remainder='passthrough')
            
            
            tnf2=ColumnTransformer(transformers=[
                ('OneHotEncoding',OneHotEncoder(sparse=False,drop='first',dtype='int',handle_unknown='ignore'),slice(6,13))
                ],remainder='passthrough')
            
            
            tnf3=ColumnTransformer(transformers=[
                ('Standardization',StandardScaler(),slice(0,76))
                ],remainder='passthrough')
            
            
            preprocessor=Pipeline([
            ('Imputation',tnf1),
            ('Encoding',tnf2),
            ('Scaling',tnf3),
                ]
    
            )
            logging.info('Pipeline Completed')
            
            return preprocessor

            

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head  : \n{test_df.head().to_string()}")

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Income'
            drop_columns = [target_column_name,'Unnamed: 0']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
