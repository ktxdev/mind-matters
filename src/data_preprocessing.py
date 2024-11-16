import pandas as pd

from enum import Enum

DATA_URL = "https://raw.githubusercontent.com/ktxdev/mind-matters/refs/heads/master/data/{}.csv"


class DataType(Enum):
    TRAIN_RAW = 'raw/train'
    TEST_RAW = 'raw/test'


def load_data(data_type: DataType = DataType.TRAIN_RAW) -> pd.DataFrame:
    """Loads data from csv files."""
    file_url = DATA_URL.format(data_type.value)
    return pd.read_csv(file_url)
