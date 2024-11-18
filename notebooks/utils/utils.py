from pandas import DataFrame
from typing import Dict


def check_missing_values(data: DataFrame, column: str):
    missing_values_rows = data[column].isnull()
    missing_values_count = missing_values_rows.sum()
    missing_values_percentage = round(missing_values_rows.mean() * 100, 2)
    print(f"Missing values count: {missing_values_count}\nMissing values percentage: {missing_values_percentage}%")


def handle_group_specific_missing_values(data: DataFrame, column: str, group_column: str, group_value: str):
    data = data.copy()
    # Calculate median for the specific group
    group_median = data.loc[data[group_column] == group_value, column].median()

    # Fill missing values for the specific group
    data.loc[data[group_column] == group_value, column] = data.loc[data[group_column] == group_value, column].fillna(
        group_median)

    # Fill remaining missing values with -1 for "Not Applicable"
    data[column] = data[column].fillna(-1)

    # Add an applicability column
    data[f'{column} Applicable'] = data[group_column].apply(lambda x: 1 if x == group_value else 0)

    return data


def handle_profession_missing_values(data: DataFrame, values: Dict[str, str]):
    """Fills missing values in the 'Profession' column based on the value in the
    'Working Professional or Student' column.

    Parameters:
        data (pd.DataFrame): The input DataFrame
        values (Dict[str, str]): Dictionary of values to fill in the missing values

    Returns:
        pd.DataFrame: The DataFrame with missing values in 'Profession' filled appropriately.
    """
    data = data.copy()

    for key, value in values.items():
        data.loc[
            data['Working Professional or Student'] == key, 'Profession'
        ] = data.loc[
            data['Working Professional or Student'] == key, 'Profession'
        ].fillna(value)

    return data


def handle_financial_stress_missing_values(data: DataFrame):
    data = data.copy()
    data['Financial Stress'] = data['Financial Stress'].fillna(data['Financial Stress'].median())
    return data
