from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.pipeline.base_pipeline import get_base_pipeline_steps
from src.validation.grid_search import build_grid_search_cv


def build_lightgbm_pipeline():
    lgbm = LGBMClassifier()

    base_pipeline_steps = get_base_pipeline_steps()
    # Add model to pipeline
    base_pipeline_steps.append(('lgbm', lgbm))

    pipeline = Pipeline(base_pipeline_steps)

    lgbm_param_grid = {
        'lgbm__n_estimators': [100, 200, 300],
        'lgbm__num_leaves': [15, 31, 63],
        'lgbm__learning_rate': [0.01, 0.1, 0.2],
        'lgbm__feature_fraction': [0.7, 0.8, 1.0],
        'lgbm__bagging_fraction': [0.7, 0.8, 1.0]
    }

    return build_grid_search_cv(pipeline, lgbm_param_grid)
