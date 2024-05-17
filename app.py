import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from scipy.special import softmax

from src import classifiers, embeddings

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
    for key in config:
        env_var = os.getenv(key)
        if env_var is not None:
            config[key] = env_var

classifier = getattr(classifiers, config['classifier_function_name'])
embeddings_model = getattr(embeddings, config['embeddings_model_class'])()

def main():   
    st.title(config['title'])
    st.write(config['description'])
    
    user_input = st.text_area(config['text_area_label'])

    if st.button(config['button_label']):
        text_embedding = embeddings_model.generate(user_input)
        similarities = classifier(text_embedding)
        probabilities = pd.Series(softmax(similarities + 1), index=similarities.index)
        classification = probabilities.idxmax()
        
        fig = go.Figure(
            data=[
                go.Bar(
                    y=probabilities.to_list(),
                    x=probabilities.index.to_list(),
                    marker_color=["orange" if ppa == classification else "blue" for ppa in probabilities.index ]
                )
            ]
        )
        
        fig.update_layout(
            title=f"Classification: {classification}",
            xaxis_title='Class',
            yaxis_title='Probability'
        )

        st.plotly_chart(fig)

if __name__ == "__main__":
    main()