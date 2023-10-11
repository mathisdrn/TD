import streamlit as st
from streamlit_extras.mention import mention
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="📈 TD Macro dynamique",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title('📈 TD Macro dynamique')

mention(
    label="Mathis Derenne",
    icon="github",  # GitHub is also featured!
    url="https://github.com/mathisdrn",
)

st.markdown("""
### Note sur le rendu :
> Je vous propose le rendu de mes TD au format d'une web-app Streamlit. 
>
> L'ensemble du code de cette web-app est disponible sur GitHub ici.
>
> Si vous souhaitez pouvoir exécuter ou modifier mon notebook un fichier requirements.txt est fourni. Il vous suffit d'exécuter la commande suivante :
> ```pip install -r requirements.txt```
""")


hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


