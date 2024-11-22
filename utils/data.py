import pandas as pd


def load_data(data_file_path: str) -> pd.DataFrame:
    """Loads data into a pandas dataframe"""
    return pd.read_csv(data_file_path)


def save_data(df: pd.DataFrame, save_path: str) -> None:
    """Saves pandas dataframe to disk as csv"""
    df.to_csv(save_path, index=False)
