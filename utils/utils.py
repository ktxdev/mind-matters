import os
import joblib


def save_model(pipeline, save_path, compress=1):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path, compress=compress)
