from enum import Enum
from typing import List

import pandas as pd
from sklearn.impute import SimpleImputer

from utils.logger import get_logger

logger = get_logger('Data Preprocessing')


class NumericImputationStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    NOT_APPLICABLE = 'NA'


class CategoricalImputationStrategy(Enum):
    MODE = 'mode'
    NEW_CATEGORY = 'new_category'


def _handle_numeric_missing_values(data: pd.DataFrame,
                                   columns: List[str],
                                   strategy: NumericImputationStrategy = NumericImputationStrategy.MEAN,
                                   fill_value: float = -1) -> pd.DataFrame:
    """
    Handles missing numerical values in the dataset.

    Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (List[str]): List of column to impute
        strategy (NumericImputationStrategy): Strategy for imputation ('mean', 'median')

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # Create a copy
    data = data.copy()
    # Instantiate an imputer with provided strategy
    if strategy == NumericImputationStrategy.NOT_APPLICABLE:
        for column in columns:
            data[column] = data[column].fillna(fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy.value)
        for column in columns:
            data[[column]] = imputer.fit_transform(data[[column]])

    return data


def _handle_categorical_missing_values(data: pd.DataFrame,
                                       columns: List[str],
                                       strategy: CategoricalImputationStrategy = CategoricalImputationStrategy.NEW_CATEGORY,
                                       category_name: str = None) -> pd.DataFrame:
    """
    Handles missing values in categorical variables.

    Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (List[str]): List of columns to impute
        strategy (CategoricalImputationStrategy): Strategy for imputation ('mode' or 'new_category')

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    pd.set_option('future.no_silent_downcasting', True)
    # Create a copy
    data = data.copy()
    if strategy.value == "mode":
        for column in columns:
            mode_value = data[column].mode()[0]
            data[column] = data[column].astype(str).replace('nan', mode_value).astype('object')
    elif strategy.value == "new_category" and category_name is not None:
        data[columns] = data[columns].astype(str).replace('nan', category_name).astype('object')
    return data


def convert_data_types(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Converting data types")
    # Copy dataframe
    data = data.copy()
    columns_to_convert = ['Academic Pressure', 'Work Pressure', 'Study Satisfaction',
                          'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
    data[columns_to_convert] = data[columns_to_convert].astype(str)
    logger.info("Converting data types successful")
    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling missing values")
    # Copy dataframe
    data = data.copy()
    # Group data into students and working professional
    students_data = data[data['Working Professional or Student'] == 'Student']
    working_professionals_data = data[data['Working Professional or Student'] == 'Working Professional']
    # handle missing values for CGPA
    students_data = _handle_numeric_missing_values(students_data,
                                                   ['CGPA'],
                                                   strategy=NumericImputationStrategy.MEDIAN)
    working_professionals_data = _handle_numeric_missing_values(working_professionals_data,
                                                                ['CGPA'],
                                                                strategy=NumericImputationStrategy.NOT_APPLICABLE,
                                                                fill_value=-1)
    # handle missing values for Academic Pressure and Study Satisfaction
    working_professionals_data = _handle_categorical_missing_values(working_professionals_data,
                                                                    ['Academic Pressure', 'Study Satisfaction'],
                                                                    CategoricalImputationStrategy.NEW_CATEGORY,
                                                                    'Not Applicable')
    students_data = _handle_categorical_missing_values(students_data,
                                                       ['Academic Pressure', 'Study Satisfaction'],
                                                       CategoricalImputationStrategy.MODE)
    # handle missing values for Work Pressure and Job Satisfaction
    students_data = _handle_categorical_missing_values(students_data,
                                                       ['Work Pressure', 'Job Satisfaction'],
                                                       CategoricalImputationStrategy.NEW_CATEGORY,
                                                       'Not Applicable')

    working_professionals_data = _handle_categorical_missing_values(working_professionals_data,
                                                                    ['Work Pressure', 'Job Satisfaction'],
                                                                    CategoricalImputationStrategy.MODE)
    # handle missing values for Profession and Degree
    working_professionals_data = _handle_categorical_missing_values(working_professionals_data,
                                                                    ['Profession'],
                                                                    CategoricalImputationStrategy.NEW_CATEGORY,
                                                                    'Unknown')
    students_data = _handle_categorical_missing_values(students_data,
                                                       ['Profession'],
                                                       CategoricalImputationStrategy.NEW_CATEGORY,
                                                       'Student')
    # Update students and working professionals data
    data.update(students_data)
    data.update(working_professionals_data)
    # handle missing values for Financial Stress, Dietary Habits and Degree
    data = _handle_categorical_missing_values(data,
                                              ['Financial Stress', 'Dietary Habits', 'Degree'],
                                              CategoricalImputationStrategy.MODE)
    logger.info("Handling missing values successful!")
    return data


def handle_outliers(data: pd.DataFrame, threshold: int = 20) -> pd.DataFrame:
    logger.info("Handling outliers...")
    # Copy the dataframe
    data = data.copy()
    columns_to_handle_outliers = ['Profession', 'City', 'Sleep Duration', 'Dietary Habits', 'Degree']
    # Categories with fewer than this count will be replaced
    for col in columns_to_handle_outliers:
        # Get the frequency of each category
        freq = data[col].value_counts()
        # Identify categories to replace
        infrequent_categories = freq[freq < threshold].index
        # Replace infrequent categories with 'Other'
        data[col] = data[col].apply(lambda category: 'Other' if category in infrequent_categories else category)
    logger.info("Handling outliers successful!")
    return data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data...")
    # Create copy of data
    df = df.copy()
    # Drop unwanted features
    df = df.drop(columns=['id', 'Name'], errors='ignore')
    # Convert data types
    df = convert_data_types(df)
    # handle missing values
    df = handle_missing_values(df)
    # handle outliers
    df = handle_outliers(df)
    logger.info("Data preprocessing successful!")
    return df
