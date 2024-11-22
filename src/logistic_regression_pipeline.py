from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from src.custom_transformers import DataPreprocessor, TargetEncoder, LabelEncoderTransformer, ConditionalScaler
from src.feature_engineering import one_hot_encoded_features, label_encoded_features, target_encoded_features

categorical_encoder = ColumnTransformer(
    transformers=[
        ('scaler', ConditionalScaler(), ['CGPA', 'Age']),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), one_hot_encoded_features),
        ('target_encoding', TargetEncoder(target_encoded_features), target_encoded_features),
        ('label_encoding', LabelEncoderTransformer(label_encoded_features), label_encoded_features)
    ],
    remainder='passthrough'  # Keeps other columns as they are
)

# conditional_scaler = ColumnTransformer(
#     transformers=[
#         ('conditional_scaler', MinMaxScaler(), ['Age'])
#     ],
#     remainder='passthrough'
# )

logreg_pipeline = Pipeline(steps=[
    ('preprocessor', DataPreprocessor()),  # Applies initial preprocessing
    # ('conditional_scaler', conditional_scaler),  # Scales Age and CGPA conditionally
    ('categorical_encoder', categorical_encoder),  # Encodes categorical features
    ('scaler', MinMaxScaler()),
    ('model', LogisticRegression(random_state=42))  # Logistic Regression Model
])
