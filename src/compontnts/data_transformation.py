import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
import os
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class dataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', "preprocessor.pkl") 


class Datatranformation:
    def __init__(self):
        self.data_transformation_config = dataTransformationConfig()
        logging.info(f"Preprocessor pickle file saved at: {self.data_transformation_config.preprocessor_obj_file}")


    def get_data_transformer_object(self):
        try:
            numerical_column = ["reading_score", "writing_score"]
            categorical_column = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # ✅ sparse matrix fix
                ]
            )

            logging.info(f"categorical columns: {categorical_column}")
            logging.info(f"numerical columns: {numerical_column}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_column),
                    ("cat_pipeline", cat_pipeline, categorical_column)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_column = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying the preprocessor object on training and testing dataframes.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Save the preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object at: {self.data_transformation_config.preprocessor_obj_file}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file,
            )

        except Exception as e:
            raise CustomException(e, sys)
