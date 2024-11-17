import pandas as pd


def handle_group_specific_missing_values(data: pd.DataFrame, column: str, group_column: str, group_value: str):
    """
    Handles missing values for a specific group in a dataset.

    Args:
        data (pd.DataFrame): The dataset containing the data.
        column (str): The column with missing values to handle.
        group_column (str): The column indicating the group type.
        group_value (str): The specific group for which to handle missing values (e.g., 'Student').

    Returns:
        pd.DataFrame: Updated dataset with missing values handled and applicability column added.
    """
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


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values in data and saves the cleaned data to disk."""
    # Create a copy of the data
    data = data.copy()

    # Handle missing values for students in the "Study Satisfaction", "Academic Pressure" and "CGPA" column
    data = handle_group_specific_missing_values(data,
                                                column="Study Satisfaction",
                                                group_column="Working Professional or Student",
                                                group_value="Student")
    data = handle_group_specific_missing_values(data,
                                                column="Academic Pressure",
                                                group_column="Working Professional or Student",
                                                group_value="Student")
    data = handle_group_specific_missing_values(data,
                                                column="CGPA",
                                                group_column="Working Professional or Student",
                                                group_value="Student")

    # Handle missing values for working professionals in the "Job Satisfaction" and "Work Pressure" column
    data = handle_group_specific_missing_values(data,
                                                column="Job Satisfaction",
                                                group_column="Working Professional or Student",
                                                group_value="Working Professional")
    data = handle_group_specific_missing_values(data,
                                                column="Work Pressure",
                                                group_column="Working Professional or Student",
                                                group_value="Working Professional")

    # Handle missing values in the "Dietary Habits", "Degree" and "Financial Stress" column
    data['Degree'] = data['Degree'].fillna(data['Degree'].mode().iloc[0])
    data['Dietary Habits'] = data['Dietary Habits'].fillna(data['Dietary Habits'].mode().iloc[0])
    data['Financial Stress'] = data['Financial Stress'].fillna(data['Financial Stress'].median())

    # Handle missing values in the "Profession" column
    data.loc[data['Working Professional or Student'] == 'Student', 'Profession'] = data.loc[
        data['Working Professional or Student'] == 'Student', 'Profession'].fillna('Student')
    data.loc[data['Working Professional or Student'] == 'Working Professional', 'Profession'] = data.loc[
        data['Working Professional or Student'] == 'Working Professional', 'Profession'].fillna('Unknown')

    return data


def handle_categorical_outliers(data: pd.DataFrame,
                                column: str,
                                threshold: int) -> pd.DataFrame:
    data = data.copy()
    # Handle outliers in the "City" column
    value_counts = data[column].value_counts()
    # Identify categories below the threshold
    categories_to_group = value_counts[value_counts < threshold].index
    # Replace categories below the threshold with 'Other'
    data[column] = data[column].replace(categories_to_group, 'Unknown')
    return data


def handling_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Handles outliers for a specific columns in the dataset."""
    # Make a copy of the dataframe
    data = data.copy()
    # Handle outliers in the "City" column
    data = handle_categorical_outliers(data, 'City', 10)
    # Handle outliers in the "Profession" column
    data = handle_categorical_outliers(data, 'Profession', 10)
    # Handle outliers in the "Sleep Duration" column
    data = handle_categorical_outliers(data, 'Sleep Duration', 20)
    # Handle outliers in the "Dietary Habits" column
    data = handle_categorical_outliers(data, 'Dietary Habits', 10)
    # Handle outliers in the "Degree" column
    data = handle_categorical_outliers(data, 'Degree', 10)

    return data


if __name__ == '__main__':
    from data_preprocessing import load_data

    data = load_data()
    data = handle_missing_values(data)
    data = handling_outliers(data)
