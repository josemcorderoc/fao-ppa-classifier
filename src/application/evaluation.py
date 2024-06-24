
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from interfaces.classifier import Classifier


def ppa_experiment(clf: Classifier, X_train: np.ndarray, y_train: np.ndarray, 
               X_test: np.ndarray, y_test_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)
    y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=classes)

    experiment = []
    for ppa in classes:
        y_test_ppa = y_test_df[ppa]
        experiment.append({
            "ppa": ppa,
            "roc_auc": roc_auc_score(y_test_ppa, y_pred_proba_df[ppa]),
            "average_precision": average_precision_score(y_test_ppa, y_pred_proba_df[ppa]),
        })
    return pd.DataFrame(experiment)


def get_top_predictions(y_pred: np.ndarray, classes: list[str], n: int = 3):
    return pd.DataFrame(y_pred, columns=classes).apply(lambda x: x.sort_values(ascending=False).head(n).index.to_list(), axis=1)