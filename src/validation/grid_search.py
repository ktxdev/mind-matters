from sklearn.model_selection import GridSearchCV, StratifiedKFold


def build_grid_search_cv(pipeline, param_grid, cv: int = 5, scoring: str = 'roc_auc', verbose: int = 3):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    return GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=skf, scoring=scoring, verbose=verbose,
                        n_jobs=-1)
