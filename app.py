import streamlit as st
import base58
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

st.write("# :orange[REVIEW GRAPH NEURAL NETWORK AND APPLICATIONS]")
page_bg_img = '''
    <style>
    .stApp {
        background-image: url("./img/backfround.jpg");
        background-size: cover;
    }
    </style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
image = Image.open('./assets/img/gnn.jpg')
st.image(image, caption='Graph Neural Networks - An overview')
