from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class ConditionalScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cgpa_scaler = MinMaxScaler()
        self.age_scaler = MinMaxScaler()

    def fit(self, X, y=None):
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
