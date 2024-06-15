from datetime import datetime
from operator import call
import os
from time import sleep
from turtle import up

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import streamlit as st
import yaml
from scipy.special import softmax
from streamlit import session_state as ss
from infrastructure.s3_repository import S3Repository
from interfaces.embedding import Embedding
from interfaces.repository import Repository
from models.feedback import Feedback
from models.embeddings.openai_embedding import OpenAIEmbedding
from src import classifiers
# from streamlit_extras.stateful_button import button 

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
    for key in config:
        env_var = os.getenv(key)
        if env_var is not None:
            config[key] = env_var

classifier = getattr(classifiers, config['classifier_function_name'])
# embeddings_model = getattr(embeddings, config['embeddings_model_class'])()
ppas = ['BE1', 'BE2', 'BE3', 'BE4', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'BL6', 'BN1', 'BN2', 'BN3', 'BN4', 'BN5', 'BP1', 'BP2', 'BP3', 'BP4', 'BP5']


for attribute in ["predicted_classification", "selected_classification", "liked", "disliked"]:
    if attribute not in ss:
        ss[attribute] = None



def main(repo: Repository, embedding: Embedding): 
    
    def run_classification(prompt_input: str):
        text_embedding = embedding.generate(prompt_input)
        similarities = classifier(text_embedding)
        probabilities = pd.Series(softmax(similarities + 1), index=similarities.index)
        classification = probabilities.idxmax()
        ss.predicted_classification = classification
        ss.selected_classification = None
        
    
    st.title(config['title'])
    st.write(config['description'])
    
    user_input = st.text_area(config['text_area_label'])
    
    classify_button = st.button(config['button_label'], on_click=run_classification, key='classification', args=(user_input,))

    if ss.predicted_classification:
        col1, col2, col3, _ = st.columns([.2, .08, .08, .64  ])
        with col1:
            st.write(f"Classification: {ss.predicted_classification}")
        
        with col2:
            st.button("👍", key="liked")
            
        with col3:
            st.button("👎", key="disliked")
            
    
    if ss.disliked or ss.selected_classification:
        st.selectbox(
            "Select the correct classification:",
            ppas,
            index=None, key='selected_classification')
        
    if ss.selected_classification:
        if st.button("Save"):
            feedback = Feedback(datetime.now(pytz.utc), user_input, ss.predicted_classification, ss.selected_classification, False, True)
            repo.AddFeedback(feedback)
            st.info("Feedback saved successfully.")
            st.button("Clear All", key="reset", on_click=ss.clear)
            
    if ss.liked:
        feedback = Feedback(datetime.now(pytz.utc), user_input, ss.predicted_classification, ss.predicted_classification, True, False)
        repo.AddFeedback(feedback)
        st.info("Feedback saved successfully.")
        st.button("Clear All", key="reset", on_click=ss.clear)

if __name__ == "__main__":
    s3repo = S3Repository()
    embedding = OpenAIEmbedding()
    main(s3repo, embedding)