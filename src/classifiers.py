from functools import partial
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def dummy_classify(embedding: np.ndarray) -> pd.Series:
    return pd.Series({
        "Positive": 0.2,
        "Negative": 0.1,
        "Neutral": 0.7
    })
    
def embedding_classifier(text_embedding: np.ndarray, compare_embeddings: pd.Series) -> pd.Series:
    
    similarity = cosine_similarity(text_embedding.reshape(1, -1), np.vstack(compare_embeddings.to_numpy()))
    recommendations_df_sim = pd.DataFrame(similarity, columns=compare_embeddings.index)
    return recommendations_df_sim.iloc[0]
    # recommendations_df_sim["prediction"] = recommendations_df_sim[embeddings.index].idxmax(axis=1).iloc[0]
    
ppa_embeddings_small_df = pd.read_parquet("data/embeddings/ppa_embeddings_small.parquet").set_index("initials")
ppa_embeddings_small_outcome = ppa_embeddings_small_df["ada_embedding_outcome"]

outcome_embedding_classifier = partial(embedding_classifier, compare_embeddings=ppa_embeddings_small_outcome)