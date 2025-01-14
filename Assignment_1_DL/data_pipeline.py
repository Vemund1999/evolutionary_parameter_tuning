
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import time


from FeatureSpecificFunctions.feature_dropper import FeatureDropper
from FeatureSpecificFunctions.remove_months_part import RemoveMonthsPart
from FeatureSpecificFunctions.convert_grade_to_numeric import ConvertGradeToNumeric
from FeatureSpecificFunctions.convert_to_numeric import ConvertToNumeric
from FeatureSpecificFunctions.convert_to_emp_length import ConvertEmpLengthToNumeric







from dataset import Dataset


class DataPipeline:
    def __init__(self):
        pass




    def get_columns_with_missing_values(self, df, threshold):
        # Calculate the percentage of missing values for each column
        missing_percentage = df.isnull().mean() * 100
        columns_above_threshold = missing_percentage[missing_percentage > threshold].index.tolist()
        return columns_above_threshold



    def define_features(self, df):

        # removing various features
        too_various_features = ["member_id", "zip_code", "emp_title"]
        trouble = ["emp_title", "last_week_pay"]
        text_features = ["desc", "title"]

        one_corr = ["funded_amnt", "funded_amnt_inv"] # også loan_amnt, men holder på den

        self.features_with_high_missing = self.get_columns_with_missing_values(df, 50)

        self.features_to_remove = too_various_features + text_features + trouble + self.features_with_high_missing + one_corr

        # numerical values: filling null values, and normelizing
        self.numerical_features = [
            "loan_amnt",
            "funded_amnt",
            "funded_amnt_inv",
            "int_rate",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "inq_last_6mths",
            "mths_since_last_delinq",
            "mths_since_last_record",
            "mths_since_last_major_derog",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "total_acc",
            "total_rec_int",
            "total_rec_late_fee",
            "recoveries",
            "collection_recovery_fee",
            "collections_12_mths_ex_med",
            "acc_now_delinq",
            "tot_coll_amt",
            "tot_cur_bal",
            "total_rev_hi_lim",
        ]
        self.numerical_features = [i for i in self.numerical_features if i not in self.features_to_remove]

        # categorical nominal features
        self.categorical_nominal_features = [
            "batch_enrolled",
            "emp_title",
            "home_ownership",
            "verification_status",
            "purpose",
            "title",
            "addr_state",
            "initial_list_status",
            "application_type",
            "verification_status",
            "last_week_pay"
        ]
        self.categorical_nominal_features = [i for i in self.categorical_nominal_features if i not in self.features_to_remove]



        self.ordinal_categorical_features = [
            "term",
            "grade",
            "sub_grade",
            "emp_length"
        ]
        self.ordinal_categorical_features = [i for i in self.ordinal_categorical_features if i not in self.features_to_remove]







    def create_ordinal_encoder(self, categories):
        return OrdinalEncoder(categories=[categories])






    def get_transformer(self, df):
        self.define_features(df)


        # Reversed grade_order
        grade_order = ['G1', 'G2', 'G3', 'G4', 'G5',  # G grades
                       'F1', 'F2', 'F3', 'F4', 'F5',  # F grades
                       'E1', 'E2', 'E3', 'E4', 'E5',  # E grades
                       'D1', 'D2', 'D3', 'D4', 'D5',  # D grades
                       'C1', 'C2', 'C3', 'C4', 'C5',  # C grades
                       'B1', 'B2', 'B3', 'B4', 'B5',  # B grades
                       'A1', 'A2', 'A3', 'A4', 'A5']  # A grades

        ordered_years = [
            '0',
            '< 1 year',  # Less than 1 year
            '1 year',  # 1 year
            '2 years',  # 2 years
            '3 years',  # 3 years
            '4 years',  # 4 years
            '5 years',  # 5 years
            '6 years',  # 6 years
            '7 years',  # 7 years
            '8 years',  # 8 years
            '9 years',  # 9 years
            '10+ years'  # More than 10 years
        ]

        ordinal_categories = {
            'term': ['36 months', '60 months'],
            'grade': ['G', 'F', 'E', 'D', 'C', 'B', 'A'],
            'sub_grade': grade_order,
            'emp_length': ordered_years
        }


        pca_treshold = 0.90


        numeric_pipeline = Pipeline(steps=[
            ('convert_to_numeric', ConvertToNumeric()),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', MinMaxScaler())
            # ('pca', PCA(n_components=pca_treshold))
        ])

        # pipeline for categorical nominal features
        category_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='?')),  # Fill missing with 'missing'
            ('ordinal_encoder', OrdinalEncoder()) # TODO: var one-hot encoder...
        ])


        # Ordinal categorical pipeline with specified orders
        ordinal_categorical_pipelines = []
        for feature, categories in ordinal_categories.items():
            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='0')),  # Fill missing with '0'
                ('ordinal_encoder', self.create_ordinal_encoder(categories))  # Create encoder with specific categories
                # ('pca', PCA(n_components=pca_treshold))
            ])
            ordinal_categorical_pipelines.append((feature, ordinal_pipeline))


        transformers = [
            ('remove_features', 'drop', self.features_to_remove),
            ('numeric_pipeline', numeric_pipeline, self.numerical_features),
            ('categorical_pipeline', category_pipeline, self.categorical_nominal_features)
        ]

        # Add the ordinal categorical pipelines
        for feature, pipeline in ordinal_categorical_pipelines:
            transformers.append((f'ordinal_categorical_pipeline_{feature}', pipeline, [feature]))

        transformer = ColumnTransformer(transformers=transformers)

        return transformer

















