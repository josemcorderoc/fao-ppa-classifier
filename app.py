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


for attribute in ["predicted_classification", "liked", "disliked", "user_input", "classified_text"]:
    if attribute not in ss:
        ss[attribute] = None

for attribute in ["selected_missing_ppas", "selected_incorrect_ppas"]:
    if attribute not in ss:
        ss[attribute] = []


def main(repo: Repository, embedding: Embedding, clf: Classifier):

    def run_classification(prompt_input: str):
        text_embedding = embedding.generate(prompt_input)
        classification_result = pd.DataFrame(clf.predict_proba(
            text_embedding.reshape(1, -1)), columns=ppas).iloc[0]
        classification_result_top3 = classification_result.sort_values(
            ascending=False).head(3).index.to_list()
        ss.predicted_classification = classification_result_top3
        ss.classified_text = prompt_input

    st.set_page_config(
        page_title=config['title'],
        page_icon="./data/media/img/favicon.ico",
    )
    st.title(config['title'])
    st.write(config['description'])

    user_input = st.text_area(config['text_area_label'], key="user_input")

    if ss.user_input:
        classify_button = st.button(
            config['button_label'], on_click=run_classification, key='classification', args=(user_input,))

    if ss.predicted_classification and ss.user_input == ss.classified_text:
        col1, col2, col3, _ = st.columns([.2, .08, .08, .64])
        with col1:
            st.write(
                f"Classification: {', '.join(ss.predicted_classification)}")

        with col2:
            def on_liked():
                ss.disliked = False
            button("👍", key="liked", on_click=on_liked)

        with col3:
            def on_disliked():
                ss.liked = False
            button("👎", key="disliked", on_click=on_disliked)

    if ss.predicted_classification and (ss.liked or ss.disliked or ss.selected_incorrect_ppas or ss.selected_missing_ppas):
        selected_incorrect_ppas = st.multiselect(
            "What PPAs are incorrect? ✘",
            ss.predicted_classification,
            key="selected_incorrect_ppas"
        )

        selected_missing_ppas = st.multiselect(
            "What PPAs are missing? 🔎",
            [ppa for ppa in ppas if ppa not in ss.predicted_classification],
            key="selected_missing_ppas"
        )

    if ss.liked or ss.disliked or ss.selected_incorrect_ppas or ss.selected_missing_ppas:
        if st.button("Save"):
            feedback = Feedback(datetime.now(pytz.utc), user_input, ss.predicted_classification,
                                ss.selected_incorrect_ppas, ss.selected_missing_ppas, ss.liked, ss.disliked)
            repo.AddFeedback(feedback)
            st.info("Feedback saved successfully.")
            st.button("Clear All", key="reset", on_click=ss.clear)


if __name__ == "__main__":
    s3repo = S3Repository()
    embedding = OpenAIEmbedding()
    with open('./data/classifiers/ridge_clf.pkl', 'rb') as f:
        clf = pickle.load(f)
    main(s3repo, embedding, clf)
