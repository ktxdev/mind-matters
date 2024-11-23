import os
import re
import joblib
import pandas as pd

from utils.logger import get_logger

logger = get_logger("Utilities")

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


def load_model(model_name):
    logger.info(f"Loading model {model_name}")
    models_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model_path = os.path.join(models_dir_path, model_name)
    return joblib.load(model_path)


def save_model(pipeline, model_name, compress=1):
    logger.info(f"Saving model {model_name}")
    model_path = os.path.join(os.path.abspath('models'), model_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path, compress=compress)


def load_latest_model(model_name):
    """
    Loads the latest version of a model based on its naming convention.

    Args:
        directory (str): The directory where the model files are stored.
        model_name (str): The base name of the model (e.g., 'xgb', 'logistic_regression').

    Returns:
        The loaded model or None if no matching model is found.
    """
    models_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    # Compile a regex pattern to match model files with versioning
    pattern = re.compile(rf"{model_name}_v(\d+)\.joblib")

    # List all files in the directory
    files = os.listdir(models_dir_path)

    # Find all matching files with their version numbers
    model_files = [(file, int(pattern.search(file).group(1)))
                   for file in files if pattern.search(file)]

    if not model_files:
        logger.error(f"No model files found for {model_name} in {models_dir_path}.")
        return None

    # Sort by version number in descending order
    latest_model_file = sorted(model_files, key=lambda x: x[1], reverse=True)[0][0]

    # Load the latest model
    logger.info(f"Loading latest model: {latest_model_file}")
    return load_model(latest_model_file)
