from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.custom_transformers import DataPreprocessor
from src.feature_engineering import feature_engineering_pipeline

rf_pipeline = Pipeline(steps=[
    ('preprocessor', DataPreprocessor()),  # Applies initial preprocessing
    ('feature_engineering', feature_engineering_pipeline),  # Encodes categorical features and scale numerical values
    ('rf', RandomForestClassifier(random_state=42))  # Logistic Regression Model
])

param_grid = {
    'rf__n_estimators': [50, 100, 200],  # Number of trees
    'rf__max_depth': [None, 10, 20],  # Maximum depth of trees
    'rf__min_samples_split': [2, 5, 10],  # Minimum samples required to split
    'rf__min_samples_leaf': [1, 2, 4],  # Minimum samples in a leaf
    'rf__bootstrap': [True, False]  # Bootstrap samples
}

# Perform Grid Search with Pipeline
rf_grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=3,  # Display progress
    n_jobs=-1  # Use all processors
)
