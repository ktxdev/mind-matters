from sklearn.base import BaseEstimator, TransformerMixin


class InteractionFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting needed; the logic is independent of data distribution
        return self

    def transform(self, X):
        # Copy input data to avoid modifying the original DataFrame
        X = X.copy()

        # Create a mask for students (CGPA greater than 0)
        student_mask = X['remainder__CGPA'] > 0

        new_col_name = "CGPA x Study Satisfaction"

        # Compute the interaction for students
        X[new_col_name] = -1.0  # Initialize with default value
        X.loc[student_mask, new_col_name] = (
                X.loc[student_mask, 'label_encoding__Study Satisfaction'].values *
                X.loc[student_mask, 'remainder__CGPA'].values
        )

        X['Work Pressure x Financial Stress'] = X['label_encoding__Work Pressure'] * X[
            'label_encoding__Financial Stress']

        sleep_duration_categories = ['onehot__Sleep Duration_5-6 hours', 'onehot__Sleep Duration_7-8 hours',
                                     'onehot__Sleep Duration_Less than 5 hours',
                                     'onehot__Sleep Duration_More than 8 hours',
                                     'onehot__Sleep Duration_Other']
        for cat in sleep_duration_categories:
            X[f"Job Satisfaction x {cat.split('__')[-1]}"] = X[cat] * X['label_encoding__Job Satisfaction']

        suicidal_thoughts_categories = [
            'onehot__Have you ever had suicidal thoughts ?_No',
            'onehot__Have you ever had suicidal thoughts ?_Yes'
        ]

        # Create a mask for students (Academic Pressure not equal to 0)
        student_mask = X['label_encoding__Academic Pressure'] != 0

        # Define the default value for non-students
        default_value = -1.0

        # Iterate over the suicidal thoughts categories to compute interaction terms
        for cat in suicidal_thoughts_categories:
            # Create a new column for each interaction
            new_col_name = f"Academic Pressure x {cat.split('__')[-1]}"

            # Compute the interaction for students
            X[new_col_name] = default_value  # Initialize with default value
            X.loc[student_mask, new_col_name] = (
                    X.loc[student_mask, 'label_encoding__Academic Pressure'].values *
                    X.loc[student_mask, cat].values
            )

        dietary_categories = ['onehot__Dietary Habits_Healthy', 'onehot__Dietary Habits_Moderate',
                              'onehot__Dietary Habits_Unhealthy', 'onehot__Dietary Habits_Other']

        # Iterate over the dietary categories to compute interaction terms
        for cat in dietary_categories:
            # Create a new column for each interaction
            new_col_name = f"Financial Stress x {cat.split('__')[-1]}"

            # Compute the interaction for students
            X[new_col_name] = default_value  # Initialize with default value
            X.loc[student_mask, new_col_name] = (
                    X.loc[student_mask, 'label_encoding__Financial Stress'].values *
                    X.loc[student_mask, cat].values
            )

        return X
