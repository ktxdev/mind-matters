from pandas import DataFrame


def check_missing_values(data: DataFrame, column: str):
    missing_values_rows = data[column].isnull()
    missing_values_count = missing_values_rows.sum()
    missing_values_percentage = round(missing_values_rows.mean() * 100, 2)
    print(f"Missing values count: {missing_values_count}\nMissing values percentage: {missing_values_percentage}%")
