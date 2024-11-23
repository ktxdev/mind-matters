import os
from typing import Tuple

import pandas as pd


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data into a pandas dataframe"""
    data_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data'))
    file_path = os.path.join(data_dir_path, 'raw/train.csv')
    data = pd.read_csv(file_path)
    target = 'Depression'
    return data.drop(columns=[target]), data[target]


def save_data(df: pd.DataFrame, save_path: str) -> None:
    """Saves pandas dataframe to disk as csv"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
