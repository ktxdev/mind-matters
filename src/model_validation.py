import pandas as pd



def validate_and_save_predictions(model, X_test):
    y_pred = model.predict(X_test)
    submission_df = pd.concat([pd.DataFrame(X_test['id']), pd.DataFrame(y_pred)], axis=1, names=['id', 'Depression'])
    save_data(submission_df, 'data/submission.csv')


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utils.data import load_data, save_data
    from utils.utils import load_model

    data_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    file_path = os.path.join(data_dir_path, 'raw/test.csv')

    test_data = load_data(file_path)

    model_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model_path = os.path.join(model_dir_path, 'catboost_v1.joblib')
    model = load_model(model_path)

    validate_and_save_predictions(model, test_data)