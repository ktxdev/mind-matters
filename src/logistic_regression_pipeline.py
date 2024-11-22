from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.custom_transformers import DataPreprocessor
from src.feature_engineering import feature_engineering_pipeline

logreg_pipeline = Pipeline(steps=[
    ('preprocessor', DataPreprocessor()),  # Applies initial preprocessing
    ('feature_engineering', feature_engineering_pipeline),  # Encodes categorical features
    ('logreg', LogisticRegression(random_state=42))  # Logistic Regression Model
])

param_grid = {
    'logreg__penalty': ['l1', 'l2', 'elasticnet', None],  # Regularization types
    'logreg__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength (inverse of lambda)
    'logreg__solver': ['liblinear', 'saga', 'lbfgs'],  # Optimization algorithms
    'logreg__max_iter': [100, 500, 1000],  # Maximum number of iterations
    'logreg__l1_ratio': [0.1, 0.5, 0.9],  # ElasticNet mixing ratio (only if penalty='elasticnet')
}

grid_search_logreg = GridSearchCV(logreg_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
