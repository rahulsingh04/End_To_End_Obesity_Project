import sys
import os 
### Giving Project Root Location To The Python
sys.path.append(os.getcwd())
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("Artifact","train.csv")
    test_data_path  = os.path.join("Artifact", "test.csv")
    raw_data_path   = os.path.join("Artifact", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered The Data Ingestion Method Or Components:->")
        try:
            df = pd.read_csv(r"notebooks\data\train.csv").drop(columns='id', axis =1)
            logging.info("Label Encoding Preprocess start :-->")
            # Provided categorical mapping
            cat_mapping = {'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Obesity_Type_I': 2, 
               'Obesity_Type_II': 3, 'Obesity_Type_III': 4, 'Overweight_Level_I': 5, 
               'Overweight_Level_II': 6}

            # Use replace to map categorical values to numerical values
            df['NObeyesdad'] = df['NObeyesdad'].replace(cat_mapping)
            logging.info(cat_mapping)
            logging.info("Logging The Dataset And The DataFrame:->")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header = True)
            
            logging.info("Train Test Split Initiated:->")

            train_set, test_set = train_test_split(df, random_state=42, test_size=0.2,shuffle=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            
            logging.info("Ingestion Of The Data Is Completed !!!")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        except Exception as e:
            CustomException(e,sys)



if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print("Final Accuracy Of The Model :->>",model_trainer.initiate_model_trainer(train_arr,test_arr))
