from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class ConvertToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.apply(pd.to_numeric, errors='coerce')
        return X