from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.custom_transformers import DataPreprocessor
from src.feature_engineering import feature_engineering_pipeline

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', DataPreprocessor()),  # Applies initial preprocessing
    ('feature_engineering', feature_engineering_pipeline),  # Encodes categorical features and scale numerical values
    ('xgb', XGBClassifier(eval_metric="logloss"))
])

param_grid_xgb = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.7, 0.8, 0.9],
    'xgb__colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search_xgb = GridSearchCV(estimator=xgb_pipeline, param_grid=param_grid_xgb, cv=5, scoring='accuracy', verbose=3,
                               n_jobs=-1)
