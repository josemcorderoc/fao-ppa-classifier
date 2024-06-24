import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import streamlit as st
import yaml
from streamlit import session_state as ss
from streamlit_extras.stateful_button import button

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from application.ppas import predict_ppas
from infrastructure.s3_repository import S3Repository
from interfaces.classifier import Classifier
from interfaces.embedding import Embedding
from interfaces.repository import Repository
from models.embeddings.openai_embedding import OpenAIEmbedding
from models.feedback import Feedback



with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

    for key in config:
        env_var = os.getenv(key)
        if env_var is not None:
            config[key] = env_var


ppas = ['BE.1',
        'BE.2',
        'BE.3',
        'BE.4',
        'BL.1',
        'BL.2',
        'BL.3',
        'BL.4',
        'BL.5',
        'BL.6',
        'BN.1',
        'BN.2',
        'BN.3',
        'BN.4',
        'BN.5',
        'BP.1',
        'BP.2',
        'BP.3',
        'BP.4',
        'BP.5']


for attribute in ["predicted_classification", "main_ppa_liked", "main_ppa_disliked", "user_input", "classified_text", "main_ppa", "other_ppas", "selected_main_ppa", "other_ppas_liked", "other_ppas_disliked"]:
    if attribute not in ss:
        ss[attribute] = None

for attribute in ["selected_incorrect_other_ppas", "selected_missing_other_ppas"]:
    if attribute not in ss:
        ss[attribute] = []


def main(repo: Repository, embedding: Embedding, principal_clf: Classifier, multi_clf: Classifier):

    def run_classification(prompt_input: str):
        ss.main_ppa, ss.other_ppas = predict_ppas(prompt_input, principal_clf, multi_clf, embedding, ppas)
        ss.classified_text = prompt_input

    st.set_page_config(
        page_title=config['title'],
        page_icon="./data/media/img/favicon.ico",
    )
    st.title(config['title'])
    st.write(config['description'])

    user_input = st.text_area(config['text_area_label'], key="user_input")

    if ss.user_input:
        st.button(config['button_label'], on_click=run_classification, key='classification', args=(user_input,))

    if ss.main_ppa and ss.user_input == ss.classified_text:
        col1, col2, col3, _ = st.columns([.2, .08, .08, .64])
        with col1:
            st.write(f"Main PPA: {ss.main_ppa}")

        with col2:
            def on_main_ppa_liked():
                ss.main_ppa_disliked = False
            button("👍", key="main_ppa_liked", on_click=on_main_ppa_liked)

        with col3:
            def on_main_ppa_disliked():
                ss.main_ppa_liked = False
            button("👎", key="main_ppa_disliked", on_click=on_main_ppa_disliked)

    if ss.main_ppa and (ss.main_ppa_disliked or ss.selected_main_ppa):
        st.selectbox(
            "What is the main PPA? 🎯",
            [ppa for ppa in ppas if ppa != ss.main_ppa],
            key="selected_main_ppa"
        )

    if ss.main_ppa_liked or (ss.main_ppa_disliked and ss.selected_main_ppa):
        col1, col2, col3, _ = st.columns([.2, .08, .08, .64])
        with col1:
            st.write(f"Secondary PPAs: {', '.join(ss.other_ppas)}")

        with col2:
            def on_other_ppas_liked():
                ss.other_ppas_disliked = False
            button("👍 ", key="other_ppas_liked", on_click=on_other_ppas_liked)

        with col3:
            def on_other_ppas_disliked():
                ss.other_ppas_liked = False
            button("👎 ", key="other_ppas_disliked", on_click=on_other_ppas_disliked)
    
        if ss.other_ppas and (ss.other_ppas_disliked or ss.selected_incorrect_other_ppas or ss.selected_missing_other_ppas):
            st.multiselect(
                "What secondary PPAs are incorrect? ❌",
                ss.other_ppas,
                key="selected_incorrect_other_ppas"
            )

            st.multiselect(
                "What secondary PPAs are missing? 🔎",
                [ppa for ppa in ppas if ppa not in ss.other_ppas],
                key="selected_missing_other_ppas"
            )
    
    if (ss.main_ppa_liked or (ss.main_ppa_disliked and ss.selected_main_ppa)) and (ss.other_ppas_liked or (ss.other_ppas_disliked and (ss.selected_incorrect_other_ppas or ss.selected_missing_other_ppas))):
        if st.button("Save"):
            feedback = Feedback(
                datetime.now(pytz.utc),
                ss.user_input,
                ss.main_ppa,
                ss.main_ppa_liked,
                ss.main_ppa_disliked,
                ss.selected_main_ppa,
                ss.other_ppas,
                ss.other_ppas_liked,
                ss.other_ppas_disliked,
                ss.selected_incorrect_other_ppas,
                ss.selected_missing_other_ppas
            )
            repo.AddFeedback(feedback)
            st.info("Feedback saved successfully.")
            st.button("Clear All", key="reset", on_click=ss.clear)


if __name__ == "__main__":
    s3repo = S3Repository()
    embedding = OpenAIEmbedding()
    with open('./models/principal_ppa_clf/ridge_v1.pkl', 'rb') as f:
        principal_clf = pickle.load(f)
        
    with open('./models/multi_ppa_clf/ridge_v1.pkl', 'rb') as f:
        multi_clf = pickle.load(f)
    main(s3repo, embedding, principal_clf, multi_clf)
