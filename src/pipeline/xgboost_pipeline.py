from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.pipeline.base_pipeline import get_base_pipeline_steps
from src.validation.grid_search import build_grid_search_cv


def build_xgb_pipeline():
    xgb = XGBClassifier(eval_metric="logloss")

    base_pipeline_steps = get_base_pipeline_steps()
    # Add model to pipeline
    base_pipeline_steps.append(('xgb', xgb))

    pipeline = Pipeline(base_pipeline_steps)

    xgb_param_grid = {
        # 'oversampling__sampling_strategy': ['auto', 0.5, 0.8],
        # 'oversampling__k_neighbors': [3, 5, 7],
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'xgb__subsample': [0.7, 0.8, 0.9],
        'xgb__colsample_bytree': [0.7, 0.8, 1.0],
        'xgb__scale_pos_weight': [1, 10, 50]
    }

    return build_grid_search_cv(pipeline, xgb_param_grid)
