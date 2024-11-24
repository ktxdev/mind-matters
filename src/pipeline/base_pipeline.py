from typing import List

from src.transformers.custom_transformers import ScaleNumericFeatures, GenerateInteractionFeatures
from src.transformers.encoders import EncodeCategoricalFeatures
from src.transformers.preprocessors import DataPreprocessor


def get_base_pipeline_steps() -> List:
    one_hot_encoded_features = ['Working Professional or Student', 'Have you ever had suicidal thoughts ?',
                                'Sleep Duration', 'Dietary Habits', 'Family History of Mental Illness']

    target_encoded_features = ['City', 'Profession']

    label_encoded_features = ['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction',
                              'Work/Study Hours', 'Financial Stress']

    return [
        ('preprocessor', DataPreprocessor()),
        ('scale_numeric', ScaleNumericFeatures()),
        ('encode_categorical', EncodeCategoricalFeatures(
            one_hot_encoded_features, target_encoded_features, label_encoded_features
        )),
        ('generate_interactions', GenerateInteractionFeatures())
    ]
