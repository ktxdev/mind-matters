import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from src.data_preprocessing import preprocess_data


class ConditionalScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cgpa_scaler = MinMaxScaler()
        self.age_scaler = MinMaxScaler()

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ConditionalScaler requires a DataFrame during fit.")

        # Fit scalers
        student_mask = X['CGPA'] > 0
        self.cgpa_scaler.fit(X.loc[student_mask, ['CGPA']])
        self.age_scaler.fit(X[['Age']])
        return self

    def transform(self, X):
        # Scale CGPA
        student_mask = X['CGPA'] > 0
        X.loc[student_mask, 'CGPA'] = self.cgpa_scaler.transform(X.loc[student_mask, ['CGPA']]).flatten()

        # Scale Age
        X['Age'] = self.age_scaler.transform(X[['Age']]).flatten()
        return X

    def get_feature_names_out(self, input_features=None):
        return ['CGPA_Scaled', 'Age_Scaled']


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoding_maps = {}

    def fit(self, X, y=None):
        # Calculate target means for each column using X and y
        for col in self.columns:
            temp_df = pd.DataFrame({col: X[col], 'Depression': y})
            self.encoding_maps[col] = temp_df.groupby(col)['Depression'].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)

        for col in self.columns:
            X[col] = X[col].map(self.encoding_maps[col])
        return X

    def get_feature_names_out(self, input_features=None):
        # Return the column names for the target-encoded features
        return self.columns


# Custom Label Encoding Transformer
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.label_encoders = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)

        for col in self.columns:
            X[col] = X[col].astype(str).replace('Not Applicable', '0')
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))
        return X

    def get_feature_names_out(self, input_features=None):
        # Return the column names for the target-encoded features
        return self.columns


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Add parameters here if you want to make the preprocessing customizable

    def fit(self, X, y=None):
        # No fitting needed for preprocessing, but this method must exist
        return self

    def transform(self, X):
        # Apply your preprocessing function
        return preprocess_data(X)
