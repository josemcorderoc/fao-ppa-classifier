import pandas as pd

def dummy_classify(text: str) -> pd.Series:
    return pd.Series({
        "Positive": 0.2,
        "Negative": 0.1,
        "Neutral": 0.7
    })