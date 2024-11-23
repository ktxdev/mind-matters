from sklearn.base import BaseEstimator, TransformerMixin

from src.data.data_preprocessing import preprocess_data


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Add parameters here if you want to make the preprocessing customizable

    def fit(self, X, y=None):
        # No fitting needed for preprocessing, but this method must exist
        return self

    def transform(self, X):
        # Apply your preprocessing function
        return preprocess_data(X)



