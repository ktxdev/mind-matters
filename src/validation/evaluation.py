from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_classification_model(y_true, y_pred):
    """Evaluates classification model performance."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics


def log_metrics(metrics, logger):
    """Logs the calculated metrics."""
    logger.info("Model Evaluation:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
