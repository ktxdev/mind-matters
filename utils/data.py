import os

import pandas as pd

data_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def load_data(file_name: str = 'train.csv') -> pd.DataFrame:
    """Loads data into a pandas dataframe"""
    file_path = os.path.join(data_dir_path, f'raw/{file_name}')
    return pd.read_csv(file_path)


def save_data(df: pd.DataFrame, save_path: str) -> None:
    """Saves pandas dataframe to disk as csv"""
    save_path = os.path.join(data_dir_path, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
