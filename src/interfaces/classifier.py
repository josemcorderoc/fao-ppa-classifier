from typing import Protocol

import numpy as np


class Classifier(Protocol):
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the given data."""
        return np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates for the given data."""
        return np.array([])
