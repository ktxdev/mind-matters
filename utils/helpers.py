import pandas as pd

def check_and_print_missing_value_counts(data: pd.DataFrame, column_name: str) -> None:
    """Counts and displays missing value counts for given column_name"""
    missing_values_rows = data[column_name].isnull()
    # Count missing values
    missing_values_count = missing_values_rows.sum()
    # Calculate missing values percentage
    missing_values_percentage = round(missing_values_rows.mean() * 100, 2)
    print(f"Missing values count: {missing_values_count}\nMissing values percentage: {missing_values_percentage}%")