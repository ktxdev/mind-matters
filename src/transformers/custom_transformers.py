import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.transformers.feature_engineering import InteractionFeatureEngineer
from src.transformers.scalers import ConditionalScaler


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    A wrapper to convert NumPy output from ColumnTransformer back to a DataFrame.
    """

    def __init__(self, transformer, feature_names):
        self.transformer = transformer
        self.feature_names = feature_names

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        transformed = self.transformer.transform(X)
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)


class ScaleNumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = ConditionalScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


class GenerateInteractionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_engineer = InteractionFeatureEngineer()

    def fit(self, X, y=None):
        self.feature_engineer.fit(X)
        return self

    def transform(self, X):
        X = self.feature_engineer.transform(X)
        return X
