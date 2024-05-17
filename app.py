import os

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src import classifiers

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
    for key in config:
        env_var = os.getenv(key)
        if env_var is not None:
            config[key] = env_var

classifier = getattr(classifiers, config['classifier_function_name'])

def main():   
    st.title(config['title'])
    st.write(config['description'])
    
    user_input = st.text_area(config['text_area_label'])

    if st.button(config['button_label']):
        
        probabilities = classifier(user_input)

        classification = probabilities.idxmax()
        st.write(f"Classification:\n**{classification}**")
        
        fig = go.Figure(data=[go.Bar(y=probabilities.to_list(), x=probabilities.index.to_list(),)])
        fig.update_layout(
            xaxis_title='Class',
            yaxis_title='Probability'
        )

        st.plotly_chart(fig)

if __name__ == "__main__":
    main()