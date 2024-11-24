from sklearn.base import BaseEstimator, TransformerMixin


class InteractionFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, from_pipeline=True):
        super().__init__()
        self.from_pipeline = from_pipeline

    def _handle_remainder_column_name(self, column_name):
        """Adjust column names based on whether data is from a pipeline."""
        return f'remainder__{column_name}' if self.from_pipeline else column_name

    def _handle_label_encoded_column_name(self, column_name):
        """Adjust label encoded column names based on whether data is from a pipeline."""
        return f'label_encoding__{column_name}' if self.from_pipeline else column_name

    def _handle_target_encoded_column_name(self, column_name):
        """Adjust label encoded column names based on whether data is from a pipeline."""
        return f'target_encoding__{column_name}' if self.from_pipeline else column_name

    def _create_interaction(self, X, col1, col2, new_col_name, mask=None, default_value=-1.0):
        """
        Create a new column in X representing the interaction between col1 and col2.
        Apply a mask if provided and initialize with a default value.
        """
        X[new_col_name] = default_value
        if mask is not None:
            X.loc[mask, new_col_name] = (
                    X.loc[mask, col1].values * X.loc[mask, col2].values
            )
        else:
            X[new_col_name] = X[col1] * X[col2]
        return X

    def fit(self, X, y=None):
        # No fitting needed; the logic is independent of data distribution
        return self

    def transform(self, X, y=None):
        # Copy input data to avoid modifying the original DataFrame
        X = X.copy()

        # Mask for students
        student_mask = X[self._handle_remainder_column_name('CGPA')] > 0
        working_professionals_mask = X[self._handle_remainder_column_name('CGPA')] == -1

        # Interaction configurations
        interactions = [
            # Single-column interaction
            {
                "col1": self._handle_remainder_column_name('CGPA'),
                "col2": self._handle_label_encoded_column_name('Study Satisfaction'),
                "new_col": "CGPA x Study Satisfaction",
                "mask": student_mask,
            },
            {
                "col1": self._handle_remainder_column_name('Work Pressure'),
                "col2": self._handle_remainder_column_name('Financial Stress'),
                "new_col": "Work Pressure x Financial Stress",
            },
            # Profession × Job Satisfaction
            {
                "col1": self._handle_label_encoded_column_name('Job Satisfaction'),
                "col2": self._handle_target_encoded_column_name('Profession'),
                "new_col": "Job Satisfaction x Profession",
                "mask": working_professionals_mask,
            },
            # Study Satisfaction × Financial Stress
            {
                "col1": self._handle_label_encoded_column_name('Study Satisfaction'),
                "col2": self._handle_label_encoded_column_name('Financial Stress'),
                "new_col": "Study Satisfaction x Financial Stress",
                "mask": student_mask,
            }
        ]

        # Sleep duration interaction categories
        sleep_duration_categories = [
            'Sleep Duration_5-6 hours', 'Sleep Duration_7-8 hours',
            'Sleep Duration_Less than 5 hours', 'Sleep Duration_More than 8 hours',
            'Sleep Duration_Other',
        ]
        if self.from_pipeline:
            sleep_duration_categories = [f'onehot__{cat}' for cat in sleep_duration_categories]

        for cat in sleep_duration_categories:
            interactions.append({
                "col1": self._handle_label_encoded_column_name('Job Satisfaction'),
                "col2": cat,
                "new_col": f"Job Satisfaction x {cat.split('__')[-1]}",
            })

        # Suicidal thoughts interaction categories
        suicidal_thoughts_categories = [
            'Have you ever had suicidal thoughts ?_No',
            'Have you ever had suicidal thoughts ?_Yes',
        ]
        if self.from_pipeline:
            suicidal_thoughts_categories = [f'onehot__{cat}' for cat in suicidal_thoughts_categories]

        for cat in suicidal_thoughts_categories:
            interactions.append({
                "col1": self._handle_label_encoded_column_name('Academic Pressure'),
                "col2": cat,
                "new_col": f"Academic Pressure x {cat.split('__')[-1]}",
                "mask": student_mask,
            })

        # Dietary habits interaction categories
        dietary_categories = [
            'Dietary Habits_Healthy', 'Dietary Habits_Moderate',
            'Dietary Habits_Unhealthy', 'Dietary Habits_Other',
        ]
        if self.from_pipeline:
            dietary_categories = [f'onehot__{cat}' for cat in dietary_categories]

        for cat in dietary_categories:
            interactions.append({
                "col1": self._handle_label_encoded_column_name('Financial Stress'),
                "col2": cat,
                "new_col": f"Financial Stress x {cat.split('__')[-1]}",
                "mask": student_mask,
            })

        # Age × Sleep Duration
        for cat in sleep_duration_categories:
            interactions.append({
                "col1": self._handle_remainder_column_name('Age'),
                "col2": cat,
                "new_col": f"Job Satisfaction x {cat.split('__')[-1]}",
            })

        role_categories = [
            'Working Professional or Student_Student',
            'Working Professional or Student_Working Professional'
        ]

        if self.from_pipeline:
            role_categories = [f'onehot__{cat}' for cat in role_categories]

        # Working Professional or Student × Work/Study Hours
        for cat in role_categories:
            interactions.append({
                "col1": self._handle_label_encoded_column_name('Work/Study Hours'),
                "col2": cat,
                "new_col": f"Work/Study Hour x {cat.split('__')[-1]}",
            })

        # Work Pressure × Suicidal Thoughts
        for cat in suicidal_thoughts_categories:
            interactions.append({
                "col1": self._handle_label_encoded_column_name('Work Pressure'),
                "col2": cat,
                "new_col": f"Work Pressure x {cat.split('__')[-1]}",
                "mask": working_professionals_mask
            })

        # Sleep Duration × Academic/Work Pressure
        for cat in sleep_duration_categories:
            interactions.append({
                "col1": self._handle_label_encoded_column_name('Work Pressure'),
                "col2": cat,
                "new_col": f"Work Pressure x {cat.split('__')[-1]}",
            })

            interactions.append({
                "col1": self._handle_label_encoded_column_name('Academic Pressure'),
                "col2": cat,
                "new_col": f"Academic Pressure x {cat.split('__')[-1]}",
            })

        # Dietary Habits × Sleep Duration
        interactions.extend([
            {
                "col1": cat1,
                "col2": cat2,
                "new_col": f"{cat1.split('__')[-1]} x {cat2.split('__')[-1]}"
            }
            for cat1 in sleep_duration_categories
            for cat2 in dietary_categories
        ])
        # Suicidal Thoughts × Financial Stress
        interactions.extend([
            {
                "col1": self._handle_label_encoded_column_name('Financial Stress'),
                "col2": cat,
                "new_col": f"Financial Stress x {cat.split('__')[-1]}",
            }
            for cat in suicidal_thoughts_categories
        ])

        # Work/Study Hours × Sleep Duration
        interactions.extend([
            {
                "col1": self._handle_label_encoded_column_name('Work/Study Hours'),
                "col2": cat,
                "new_col": f"Work/Study Hour x {cat.split('__')[-1]}",
            }
            for cat in sleep_duration_categories
        ])

        # Apply all interactions
        for interaction in interactions:
            X = self._create_interaction(
                X,
                interaction["col1"],
                interaction["col2"],
                interaction["new_col"],
                mask=interaction.get("mask"),
            )

        return X
