import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        runs: int,
        wickets: int,
        overs: int,
        runs_last_5: int,
        wickets_last_5: int,
        bat_team: str,
        bowl_team: str):

        self.runs = runs

        self.wickets = wickets

        self.overs = overs

        self.runs_last_5 = runs_last_5

        self.wickets_last_5 = wickets_last_5

        self.bat_team = bat_team

        self.bowl_team = bowl_team

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "runs": [self.runs],
                "wickets": [self.wickets],
                "overs": [self.overs],
                "runs_last_5": [self.runs_last_5],
                "wickets_last_5": [self.wickets_last_5],
                "bat_team": [self.bat_team],
                "bowl_team": [self.bowl_team],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
