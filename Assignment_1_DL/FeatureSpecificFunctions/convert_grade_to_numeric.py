
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd



class ConvertGradeToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        grade_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1, '0': 0}
        return X.map(self.grade_mapping)








