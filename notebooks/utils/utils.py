from pandas import DataFrame


def check_missing_values(data: DataFrame, column: str):
    missing_values_rows = data[column].isnull()
    missing_values_count = missing_values_rows.sum()
    missing_values_percentage = round(missing_values_rows.mean() * 100, 2)
    print(f"Missing values count: {missing_values_count}\nMissing values percentage: {missing_values_percentage}%")


def handle_group_specific_missing_values(data: DataFrame, column: str, group_column: str, group_value: str):
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