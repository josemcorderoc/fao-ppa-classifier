import numpy as np
from sklearn.ensemble import RandomForestClassifier


class CustomRandomForest(RandomForestClassifier):
    def predict_proba(self, X):
        probs = super().predict_proba(X)
        return np.array(probs)[:, :, 0].T