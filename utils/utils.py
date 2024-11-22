import os

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


def encode_categorical_features(data: DataFrame) -> DataFrame:
    categorical_features = ['City', 'Working Professional or Student', 'Profession', 'Gender',
                            'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                            'Family History of Mental Illness']
    # categorical_features = ['Working Professional or Student', 'Sleep Duration', 'Dietary Habits',
    #                         'Have you ever had suicidal thoughts ?']
    return pd.get_dummies(data, columns=categorical_features, drop_first=True)


def standardize_numerical_features(data: DataFrame) -> DataFrame:
    data = data.copy()
    scaler = StandardScaler()
    # Standardize numerical features
    numerical_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
                          'Work/Study Hours', 'Financial Stress']
    # numerical_features = ['Age', 'Academic Pressure', 'Work Pressure', 'Job Satisfaction',
    #                       'Work/Study Hours', 'Financial Stress']
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
