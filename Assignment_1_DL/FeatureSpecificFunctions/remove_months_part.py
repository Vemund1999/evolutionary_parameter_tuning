import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveMonthsPart(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure input is a pandas Series


        try:
            # Remove 'months' and convert to numeric (years)
            X = X['term'].str.replace(' months', '', regex=False)
            X = X['term'].str.strip().astype(float)
        except:
            print(type(X))
        return X
