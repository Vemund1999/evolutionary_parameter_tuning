

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ConvertEmpLengthToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pd.to_numeric(X.str.replace(' years', '', regex=False), errors='coerce')
        pd.to_numeric(X.str.replace(' year', '', regex=False), errors='coerce')
        pd.to_numeric(X.str.replace('+', '', regex=False), errors='coerce')
        pd.to_numeric(X.str.replace('< ', '', regex=False), errors='coerce')
        return X







