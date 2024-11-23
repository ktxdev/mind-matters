from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.pipeline.base_pipeline import get_base_pipeline_steps
from src.validation.grid_search import build_grid_search_cv


def build_logistic_regression_pipeline():
    logreg = LogisticRegression(random_state=42)

    base_pipeline_steps = get_base_pipeline_steps()
    # Add model to pipeline
    base_pipeline_steps.append(('logreg', logreg))

    pipeline = Pipeline(base_pipeline_steps)

    logreg_param_grid = {
        'logreg__penalty': ['l1', 'l2', 'elasticnet', None],  # Regularization types
        'logreg__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength (inverse of lambda)
        'logreg__solver': ['liblinear', 'saga', 'lbfgs'],  # Optimization algorithms
        'logreg__max_iter': [100, 500, 1000],  # Maximum number of iterations
        'logreg__l1_ratio': [0.1, 0.5, 0.9],  # ElasticNet mixing ratio (only if penalty='elasticnet')
    }

    return build_grid_search_cv(pipeline, logreg_param_grid)
