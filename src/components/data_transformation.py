import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    def get_data_preprocessor(self):

        try:
            numerical_cols = ["writing_score", "reading_score"]
            categorical_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy='median')),
                                                 ("scaler",StandardScaler())])

            categorical_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy='mode')),
                                                   ("Encoding",OneHotEncoder())])

            preprocessor_obj = ColumnTransformer([('num_pipeline',numerical_pipeline,numerical_cols),
                                                  ('obj_pipeline',categorical_pipeline,categorical_cols)])

            logging.info("Preprocessor object is created")

            return preprocessor_obj
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info("Train and Test datasets are loaded")
            logging.info("Calling the data_preprocessor obj")

            preprocessor_obj = self.get_data_preprocessor()

            target_col = "math_score"

            input_train_features_df = train_df.drop(columns=[target_col])
            target_train_df = train_df[target_col]

            input_test_features_df = test_df.drop(columns=[target_col])
            target_test_df = test_df[target_col]

            input_train_features_arr = preprocessor_obj.fit_transform(input_train_features_df)
            input_test_features_arr  = preprocessor_obj.transform(input_test_features_df)

            train_arr = np.c_[input_train_features_arr, np.array(target_train_df)]
            test_arr  = np.c_[input_test_features_arr, np.array(target_test_df)]

            logging.info(f"Saved preprocessed object.")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)
        