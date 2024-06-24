import pandas as pd

from interfaces.embedding import Embedding


def convert_to_input(df: pd.DataFrame, embedding: Embedding, classes: list[str], text_col="text", ppas_col="ppas"):
    X_df = df[text_col].apply(lambda t: pd.Series(embedding.generate(t)))
    X = X_df.to_numpy()
    for ppa in classes:
        df[ppa] = df[ppas_col].apply(lambda x: ppa in x)
    y = df[classes].to_numpy()
    return X, y
