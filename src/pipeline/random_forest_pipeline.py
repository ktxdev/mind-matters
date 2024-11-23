from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.pipeline.base_pipeline import get_base_pipeline_steps
from src.validation.grid_search import build_grid_search_cv


def build_random_forest_pipeline():
    rf = RandomForestClassifier(random_state=42)

    base_pipeline_steps = get_base_pipeline_steps()
    # Add model to pipeline
    base_pipeline_steps.append(('rf', rf))

    pipeline = Pipeline(base_pipeline_steps)

    rf_param_grid = {
        'rf__n_estimators': [50, 100, 200],  # Number of trees
        'rf__max_depth': [None, 10, 20],  # Maximum depth of trees
        'rf__min_samples_split': [2, 5, 10],  # Minimum samples required to split
        'rf__min_samples_leaf': [1, 2, 4],  # Minimum samples in a leaf
        'rf__bootstrap': [True, False]  # Bootstrap samples
    }

    return build_grid_search_cv(pipeline, rf_param_grid)
