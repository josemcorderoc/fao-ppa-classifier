import os
from pathlib import Path
import h2o
import numpy as np
from h2o import automl
import pandas as pd
from pyparsing import col
from sklearn.base import BaseEstimator
import tqdm


class H2O(BaseEstimator):
    def __init__(self, **kwargs):
        h2o.connect(verbose=False)
        self.clf = automl.H2OAutoML(**kwargs)
        # self._CustomH2O__input = self._H2OAutoML__input
        # super().__init__(**kwargs)
        
    # def __init__(self):
    #     # Initialize your H2O model here
    #     pass
    def fit(self, X: np.ndarray, y: np.ndarray):
        h2o.connect(verbose=False)
        
        X_df = pd.DataFrame(X).add_prefix("X_")
        self.X_columns = X_df.columns.to_list()
        y_df = pd.DataFrame(y).add_prefix("y_").astype("int")
                                        
        y_col = y_df.columns[0]
        train_h2o_df = h2o.H2OFrame(X_df.join(y_df))
        train_h2o_df[y_col] = train_h2o_df[y_col].asfactor()
        self.clf.train(x=self.X_columns, y=y_df.columns[0], training_frame=train_h2o_df)

    def predict(self, X: np.ndarray) -> np.ndarray:
        h2o.connect(verbose=False)
        # Use your trained H2O model to make predictions on the given features (X)
        prediction =  self.clf.predict(h2o.H2OFrame(X, column_names=self.X_columns))
        if prediction is None:
            raise ValueError("Prediction is None")
        return prediction.as_data_frame()["predict"].to_numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h2o.connect(verbose=False)
        # Use your trained H2O model to make predictions on the given features (X)
        prediction =  self.clf.predict(h2o.H2OFrame(X, column_names=self.X_columns))
        if prediction is None:
            raise ValueError("Prediction is None")
        return prediction.as_data_frame()["p1"].to_numpy()

    def save(self, path: str):
        leader_clf = self.clf.leader
        if leader_clf:
            leader_clf.save_mojo(path)
    
    def load(self, path: str):
        self.clf = h2o.import_mojo(path)
    
    
class H2OMultiLabel:
    def __init__(self, **kwargs):
        h2o.connect(verbose=False)
        self.clfs = [H2O(**kwargs) for _ in range(20)]
        # self._CustomH2O__input = self._H2OAutoML__input
        # super().__init__(**kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        h2o.connect(verbose=False)
        
        X_df = pd.DataFrame(X).add_prefix("X_")
        self.X_columns = X_df.columns.to_list()
        y_df = pd.DataFrame(y).add_prefix("y_").astype("int")
        # train individual models
        for i, clf in tqdm.tqdm(enumerate(self.clfs), total=len(self.clfs)):           
            y_col = y_df.columns[i]
            train_h2o_df = h2o.H2OFrame(X_df.join(y_df))
            train_h2o_df[y_col] = train_h2o_df[y_col].asfactor()
            clf.fit(X, y[:, i])

    def predict(self, X: np.ndarray) -> np.ndarray:
        h2o.connect(verbose=False)

        predictions = [clf.predict(X) for clf in self.clfs]
        # return np.concatenate(predictions, axis=1)
        return np.stack(predictions).T
        # # Use your trained H2O model to make predictions on the given features (X)
        # # prediction =  self.clf.predict(h2o.H2OFrame(X, column_names=self.X_columns))
        # if prediction is None:
        #     raise ValueError("Prediction is None")
        # return prediction.as_data_frame()["predict"].to_numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h2o.connect(verbose=False)
        predictions_proba = [clf.predict_proba(X) for clf in self.clfs]
        # return np.concatenate(predictions_proba, axis=1)
        return np.stack(predictions_proba).T
        # Use your trained H2O model to make predictions on the given features (X)
        # prediction =  self.clf.predict(h2o.H2OFrame(X, column_names=self.X_columns))
        # if prediction is None:
        #     raise ValueError("Prediction is None")
        # return prediction.as_data_frame()["p1"].to_numpy()    
        
    def save(self, path):
        for i, clf in enumerate(self.clfs):
            clf.save(os.path.join(path, f"model_{i}.zip"))
            
    def load(self, path):
        self.clfs = [h2o.import_mojo(os.path.join(path, f"model_{i}.zip")) for i, _ in enumerate(self.clfs)]
            