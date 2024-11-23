from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from src.pipeline.base_pipeline import get_base_pipeline_steps
from src.validation.grid_search import build_grid_search_cv


def build_catboost_pipeline():
    catboost = CatBoostClassifier(verbose=False)

    base_pipeline_steps = get_base_pipeline_steps()
    # Add model to pipeline
    base_pipeline_steps.append(('catboost', catboost))

    pipeline = Pipeline(base_pipeline_steps)

    catboost_param_grid = {
        'catboost__iterations': [100, 200, 300],
        'catboost__depth': [4, 6, 8],
        'catboost__learning_rate': [0.01, 0.1, 0.2],
        'catboost__l2_leaf_reg': [1, 3, 5]
    }

    return build_grid_search_cv(pipeline, catboost_param_grid)
