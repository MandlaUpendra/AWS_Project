import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts",'train.csv')
    test_data_path:  str = os.path.join("artifacts",'test.csv')
    raw_data_path:   str = os.path.join("artifacts",'data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Extracting the data from the source")

        try:
            data = pd.read_csv("D:\projects\AWS_DS_END_TO_END\Research\DATA\stud.csv")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

            logging.info("Splitting the data into train and test sets")
            train_data,test_data = train_test_split(data,test_size=0.3,random_state=333)

            data.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed and stored in artifacts")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_df,test_df,preprocessor_path = data_transformation.initiate_data_transformation(train_path,test_path)

    model_trainer = ModelTrainer()
    r2_squared = model_trainer.initiate_model_trainer(train_df,test_df)