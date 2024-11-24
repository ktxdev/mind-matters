import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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


# Custom Transformer for encoding categorical features
class EncodeCategoricalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_features, target_encoded_features, label_encoded_features):
        self.one_hot_features = one_hot_features
        self.target_encoded_features = target_encoded_features
        self.label_encoded_features = label_encoded_features
        self.transformer = None

    def fit(self, X, y):
        self.transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(sparse_output=False), self.one_hot_features),
                ('target_encoding', TargetEncoder(self.target_encoded_features), self.target_encoded_features),
                ('label_encoding', LabelEncoderTransformer(self.label_encoded_features), self.label_encoded_features)
            ],
            remainder='passthrough'
        )
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        transformed = self.transformer.transform(X)
        return pd.DataFrame(transformed, columns=self.transformer.get_feature_names_out())
