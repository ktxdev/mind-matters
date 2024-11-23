import pandas as pd

from sklearn.model_selection import train_test_split

from src.pipeline.catboost_pipeline import build_catboost_pipeline
from src.pipeline.lightgbm_pipeline import build_lightgbm_pipeline
from src.pipeline.logistic_regression_pipeline import build_logistic_regression_pipeline
from src.pipeline.random_forest_pipeline import build_random_forest_pipeline
from src.pipeline.xgboost_pipeline import build_xgb_pipeline
from src.validation.evaluation import evaluate_classification_model, log_metrics
from utils.data import load_data, save_data
from utils.helpers import save_model, load_latest_model
from utils.logger import get_logger

logger = get_logger("Train Model")


def train_and_evaluate(model: str):
    # Load data
    data = load_data()
    target = 'Depression'
    X, y = data.drop(columns=[target]), data[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline
    if model.lower() == 'xgb':
        pipeline = build_xgb_pipeline()
    elif model.lower() == 'catboost':
        pipeline = build_catboost_pipeline()
    elif model.lower() == 'rf':
        pipeline = build_random_forest_pipeline()
    elif model.lower() == 'lgbm':
        pipeline = build_lightgbm_pipeline()
    elif model.lower() == 'logreg':
        pipeline = build_logistic_regression_pipeline()
    else:
        logger.error(f"Model {model} not supported")
        raise ValueError(f"Model {model} not supported")

    # Train model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate
    metrics = evaluate_classification_model(y_test, y_pred)

    # Log evaluation metrics
    log_metrics(metrics, logger)

    save_model(pipeline, f'{model}_v2.joblib')


def evaluate_model(model_name: str):
    X = load_data('test.csv')

    model = load_latest_model(model_name)
    y_pred = model.predict(X)

    submission_df = pd.DataFrame({'id': X['id'], 'Depression': y_pred})

    save_data(submission_df, 'submission/submission.csv')
