import numpy as np
from sklearn.linear_model import RidgeClassifier

class CustomRidge(RidgeClassifier):
    def predict_proba(self, X):
        scores = self.decision_function(X)
        y_pred_proba = np.exp(scores) / np.sum(np.exp(scores))
        return y_pred_proba
