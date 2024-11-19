import os

import pandas as pd
from pandas import DataFrame
from typing import Dict, List

from sklearn.preprocessing import StandardScaler


def check_missing_values(data: DataFrame, column: str):
    missing_values_rows = data[column].isnull()
    missing_values_count = missing_values_rows.sum()
    missing_values_percentage = round(missing_values_rows.mean() * 100, 2)
    print(f"Missing values count: {missing_values_count}\nMissing values percentage: {missing_values_percentage}%")


def handle_group_specific_missing_values(data: DataFrame, columns: List[str], group_column: str,
                                         group_value: str) -> DataFrame:
    data = data.copy()

    for column in columns:
        # Calculate median for the specific group
        group_median = data.loc[data[group_column] == group_value, column].median()

        # Fill missing values for the specific group
        data.loc[data[group_column] == group_value, column] = data.loc[
            data[group_column] == group_value, column].fillna(
            group_median)

        # Fill remaining missing values with -1 for "Not Applicable"
        data[column] = data[column].fillna(-1)

        # Add an applicability column
        data[f'{column} Applicable'] = data[group_column].apply(lambda x: 1 if x == group_value else 0)

    return data


def handle_profession_missing_values(data: DataFrame, values: Dict[str, str]) -> DataFrame:
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


def handle_financial_stress_missing_values(data: DataFrame) -> DataFrame:
    data = data.copy()
    data['Financial Stress'] = data['Financial Stress'].fillna(data['Financial Stress'].median())
    return data


def handle_categorical_missing_values(data: DataFrame, columns: List[str]) -> DataFrame:
    data = data.copy()
    for column in columns:
        data[column] = data[column].fillna(data[column].mode()[0])
    return data


def handle_categorical_outliers(data: DataFrame,
                                columns: List[str],
                                threshold: int = 10) -> DataFrame:
    data = data.copy()
    for column in columns:
        # Handle outliers in the "City" column
        value_counts = data[column].value_counts()
        # Identify categories below the threshold
        categories_to_group = value_counts[value_counts < threshold].index
        # Replace categories below the threshold with 'Other'
        data[column] = data[column].replace(categories_to_group, 'Other')

    return data


def encode_categorical_features(data: DataFrame) -> DataFrame:
    categorical_features = ['City', 'Working Professional or Student', 'Profession', 'Gender',
                            'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                            'Family History of Mental Illness']
    return pd.get_dummies(data, columns=categorical_features, drop_first=True)


def standardize_numerical_features(data: DataFrame) -> DataFrame:
    data = data.copy()
    scaler = StandardScaler()
    # Standardize numerical features
    numerical_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
                          'Work/Study Hours', 'Financial Stress']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data


def save_data(data: DataFrame, dir_path: str, file_name: str):
    dir_path = os.path.abspath(os.path.join(os.getcwd(), f"../data/{dir_path}"))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, f"{file_name}.csv")

    # Assuming `df` is your dataframe
    data.to_csv(file_path, index=False)

    print(f"Dataframe saved!")


if __name__ == '__main__':
    df = pd.read_csv("https://raw.githubusercontent.com/ktxdev/mind-matters/refs/heads/master/data/raw/test.csv")

    df = handle_group_specific_missing_values(df,
                                              columns=["Academic Pressure", "CGPA", "Study Satisfaction"],
                                              group_column="Working Professional or Student",
                                              group_value="Student")

    df = handle_group_specific_missing_values(df,
                                              columns=["Work Pressure", "Job Satisfaction"],
                                              group_column="Working Professional or Student",
                                              group_value="Working Professional")

    fill_values = {
        'Working Professionals': 'Unknown',
        'Student': 'Student'
    }
    df = handle_profession_missing_values(df, fill_values)

    columns_to_impute = ['Degree', 'Dietary Habits']
    df = handle_categorical_missing_values(df, columns_to_impute)

    columns_to_handle_outliers = ['Profession', 'City', 'Sleep Duration', 'Dietary Habits', 'Degree']
    df = handle_categorical_outliers(df, columns=columns_to_handle_outliers, threshold=20)

    save_data(df, 'cleaned', 'test')
