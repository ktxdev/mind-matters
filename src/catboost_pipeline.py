from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.custom_transformers import DataPreprocessor
from src.feature_engineering import feature_engineering_pipeline

catboost_pipeline = Pipeline(steps=[
    ('preprocessor', DataPreprocessor()),  # Applies initial preprocessing
    ('feature_engineering', feature_engineering_pipeline),  # Encodes categorical features and scale numerical values
    ('catboost', CatBoostClassifier(verbose=False))
])

param_grid_catboost = {
    'catboost__iterations': [100, 200, 300],
    'catboost__depth': [4, 6, 8],
    'catboost__learning_rate': [0.01, 0.1, 0.2],
    'catboost__l2_leaf_reg': [1, 3, 5]
}

grid_search_catboost = GridSearchCV(estimator=catboost_pipeline,
                                    param_grid=param_grid_catboost,
                                    cv=5,
                                    scoring='accuracy',
                                    verbose=3,
                                    n_jobs=-1)
