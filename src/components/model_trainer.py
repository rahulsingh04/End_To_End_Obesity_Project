import sys
import os
from dataclasses import dataclass
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score, 
                             roc_auc_score, roc_curve, classification_report, precision_score, f1_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("Artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Training And Test Input Data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1], train_array[:,-1],
                test_array[:,:-1], test_array[:,-1]
            )
            models = {
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier()
            }            
            model_report = evaluate_models(x_train, y_train, x_test, y_test, models)
            logging.info(f"Model Report Are : {model_report}")
            
            # Get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            # Get best model name from dictionary

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best found model is {0} on both training and testing dataset And Its Accuracy Is :--> {1}".format(best_model, best_model_score))
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy


        except Exception as e:
            raise CustomException(e, sys)