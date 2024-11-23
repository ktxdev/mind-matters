import os

import joblib
import pandas as pd


def check_and_print_missing_value_counts(data: pd.DataFrame, column_name: str) -> None:
    """Counts and displays missing value counts for given column_name"""
    missing_values_rows = data[column_name].isnull()
    # Count missing values
    missing_values_count = missing_values_rows.sum()
    # Calculate missing values percentage
    missing_values_percentage = round(missing_values_rows.mean() * 100, 2)
    print(f"Missing values count: {missing_values_count}\nMissing values percentage: {missing_values_percentage}%")


def validate_dataframe(X, expected_columns):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input must be a DataFrame.")
    for col in expected_columns:
        if col not in X.columns:
            raise ValueError(f"Missing expected column: {col}")


def load_model(model_path):
    return joblib.load(model_path)


def save_model(pipeline, save_path, compress=1):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path, compress=compress)
