import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.custom_transformers import LabelEncoderTransformer, TargetEncoder, ConditionalScaler

one_hot_encoded_features = ['Gender', 'Working Professional or Student', 'Have you ever had suicidal thoughts ?',
                            'Sleep Duration', 'Dietary Habits', 'Family History of Mental Illness']

target_encoded_features = ['City', 'Profession', 'Degree']

label_encoded_features = ['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction',
                          'Work/Study Hours', 'Financial Stress']
numeric_features = ['CGPA', 'Age']

feature_engineering_pipeline = ColumnTransformer(
    transformers=[
        ('scaler', ConditionalScaler(), ['CGPA', 'Age']),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), one_hot_encoded_features),
        ('target_encoding', TargetEncoder(target_encoded_features), target_encoded_features),
        ('label_encoding', LabelEncoderTransformer(label_encoded_features), label_encoded_features)
    ],
    remainder='passthrough'  # Keeps other columns as they are
)


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = ConditionalScaler()
    df = scaler.fit_transform(df)
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    encode_categorical_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse_output=False), one_hot_encoded_features),
            ('target_encoding', TargetEncoder(target_encoded_features), target_encoded_features),
            ('label_encoding', LabelEncoderTransformer(label_encoded_features), label_encoded_features)
        ],
        remainder='passthrough'  # Keeps other columns
    )

    transformed = encode_categorical_transformer.fit_transform(df, df['Depression'])
    # Convert to DataFrame with appropriate column names
    return pd.DataFrame(transformed, columns=encode_categorical_transformer.get_feature_names_out())
