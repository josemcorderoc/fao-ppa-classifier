

import numpy as np
import pandas as pd
from interfaces.classifier import Classifier
from interfaces.embedding import Embedding


def predict_top_ppas(text_embedding: np.ndarray, clf: Classifier, classes: list[str], n: int) -> list[str]:
    return (pd.DataFrame(clf.predict_proba(text_embedding.reshape(1, -1)), columns=classes)
            .iloc[0]
            .sort_values(ascending=False)
            .head(n)
            .index
            .to_list())


def predict_ppas(text: str, principal_clf: Classifier, multi_clf: Classifier, embedding: Embedding, classes: list[str]) -> tuple[str, list[str]]:
    text_embedding = embedding.generate(text)
    
    
    principal_ppa = predict_top_ppas(text_embedding, principal_clf, classes, 1)[0]
    multi_ppas = predict_top_ppas(text_embedding, multi_clf, classes, 3)
    
    other_ppas = [ppa for ppa in multi_ppas if ppa != principal_ppa][:2]
    
    return principal_ppa, other_ppas

