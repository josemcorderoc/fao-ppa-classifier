
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, check_array, check_is_fitted
from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode
import networkx as nx


class Hierarchical:
    def __init__(self, clf: BaseEstimator, classes: list[str], **kwargs) -> None:
        self.clf = MultiLabelLocalClassifierPerNode(local_classifier=clf)
        self.classes = classes

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # adapted from MultiLabelLocalClassifierPerNode predict implementation
        
        # Check if fit has been called
        check_is_fitted(self.clf)

        # Input validation
        if not self.clf.bert:
            X = check_array(
                X, accept_sparse="csr"
            )  # TODO: Decide allow_nd True or False
        else:
            X = np.array(X)

        # Initialize array that holds predictions
        y = [[[]] for _ in range(X.shape[0])]

        bfs = nx.bfs_successors(self.clf.hierarchy_, source=self.clf.root_)

        self.clf.logger_.info("Predicting")

        probas = []
        for predecessor, successors in bfs:
            if predecessor == self.clf.root_:
                mask = [True] * X.shape[0]
                subset_x = X[mask]
                y_row_indices = [[i, [0]] for i in range(X.shape[0])]
            else:
                # get indices of rows that have the predecessor
                y_row_indices = []
                for row in range(X.shape[0]):
                    # create list of indices
                    _t = [z for z, ls in enumerate(y[row]) if ls[-1] == predecessor]

                    # y_row_indices is a list of lists, each list contains the index of the row and a list of column indices
                    y_row_indices.append([row, _t])

                # Filter
                mask = [True if ld[1] else False for ld in y_row_indices]
                y_row_indices = [k for k in y_row_indices if k[1]]
                subset_x = X[mask]

            if subset_x.shape[0] > 0:
                probabilities = np.zeros((subset_x.shape[0], len(successors)))
                for row, successor in enumerate(successors):
                    successor_name = str(successor).split(self.clf.separator_)[-1]
                    self.clf.logger_.info(f"Predicting for node '{successor_name}'")
                    classifier = self.clf.hierarchy_.nodes[successor]["classifier"]
                    positive_index = np.where(classifier.classes_ == 1)[0]

                    probabilities[:, row] = classifier.predict_proba(subset_x)[
                        :, positive_index
                    ][:, 0]

                probas.append(pd.DataFrame(probabilities, columns=successors, index=[x0 for x0, x1 in y_row_indices]))

                # get indices of probabilities that are within tolerance of max

                # highest_probabilities = np.max(probabilities, axis=1).reshape(-1, 1)

                # indices_probabilities_within_tolerance = np.argwhere(
                #     np.greater_equal(probabilities, highest_probabilities - _tolerance)
                # )
                
                indices_probabilities_within_tolerance = np.argwhere(
                    np.greater_equal(probabilities, 0)
                )

                prediction = [[] for _ in range(subset_x.shape[0])]
                for row, column in indices_probabilities_within_tolerance:
                    prediction[row].append(successors[column])

                k = 0  # index of prediction
                for row, col_list in y_row_indices:
                    for j in col_list:
                        if not prediction[k]:
                            y[row][j].append("")
                        else:
                            for pi, _suc in enumerate(prediction[k]):
                                if pi == 0:
                                    y[row][j].append(_suc)
                                else:
                                    # in case of mulitple predictions, copy the previous prediction up to (but not including) the last prediction and add the new one
                                    _old_y = y[row][j][:-1].copy()
                                    y[row].insert(j + 1, _old_y + [_suc])
                    k += 1

        # y = make_leveled(y)
        # self.clf._remove_separator(y)
        # y = np.array(y, dtype=self.clf.dtype_)
        pred_df = pd.concat(probas, axis=1).fillna(0)
        pred_df.columns = pred_df.columns.str[-4:]
        for label in self.classes:
            pred_df[label] = pred_df.apply(lambda _row: _row[label]*_row[label[:2]], axis=1)
        return pred_df[self.classes].values