from itertools import compress

import pandas as pd
from sklearn.metrics import classification_report


def train_model(pipeline, X_train, y_train):
    return pipeline.fit(X_train, y_train)


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return classification_report(y_test, y_pred)


if __name__ == '__main__':
    import os
    import sys
    from sklearn.model_selection import train_test_split

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utils.data import load_data
    from utils.utils import save_model
    from src.logistic_regression_pipeline import grid_search_logreg

    # load data
    data_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    file_path = os.path.join(data_dir_path, 'raw/train.csv')
    data = load_data(file_path)
    X = data.drop(['Depression'], axis=1)
    y = data['Depression']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train = pd.DataFrame(X_train, columns=X.columns)
    # Train model
    pipeline = train_model(grid_search_logreg, X_train, y_train)
    best_model = pipeline.best_estimator_

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models/logistic_regression_v1.joblib'))
    save_model(best_model, model_path)
