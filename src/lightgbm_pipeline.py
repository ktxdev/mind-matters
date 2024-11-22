from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.custom_transformers import DataPreprocessor
from src.feature_engineering import feature_engineering_pipeline

lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', DataPreprocessor()),  # Applies initial preprocessing
    ('feature_engineering', feature_engineering_pipeline),  # Encodes categorical features and scale numerical values
    ('lgbm', LGBMClassifier())
])

param_grid_lgbm = {
    'lgbm__n_estimators': [100, 200, 300],
    'lgbm__num_leaves': [15, 31, 63],
    'lgbm__learning_rate': [0.01, 0.1, 0.2],
    'lgbm__feature_fraction': [0.7, 0.8, 1.0],
    'lgbm__bagging_fraction': [0.7, 0.8, 1.0]
}

grid_search_lgbm = GridSearchCV(estimator=lgbm_pipeline, param_grid=param_grid_lgbm, cv=5, scoring='accuracy',
                                verbose=3, n_jobs=-1)
