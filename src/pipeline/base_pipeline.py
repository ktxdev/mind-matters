from typing import List
from imblearn.over_sampling import SMOTE

from src.transformers.custom_transformers import ScaleNumericFeatures, GenerateInteractionFeatures
from src.transformers.encoders import EncodeCategoricalFeatures
from src.transformers.preprocessors import DataPreprocessor


def get_base_pipeline_steps() -> List:
    one_hot_encoded_features = ['Working Professional or Student', 'Have you ever had suicidal thoughts ?',
                                'Sleep Duration', 'Dietary Habits', 'Family History of Mental Illness', 'Gender']

    target_encoded_features = ['City', 'Profession', 'Degree']

    label_encoded_features = ['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction',
                              'Work/Study Hours', 'Financial Stress']

    smote = SMOTE()

    return [
        ('preprocessor', DataPreprocessor()),
        ('scale_numeric', ScaleNumericFeatures()),
        ('encode_categorical', EncodeCategoricalFeatures(
            one_hot_encoded_features, target_encoded_features, label_encoded_features
        )),
        ('generate_interactions', GenerateInteractionFeatures()),
        ('oversampling', smote)
    ]
