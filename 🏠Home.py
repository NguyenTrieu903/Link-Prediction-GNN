import streamlit as st
from PIL import Image
from assets.theme import set_custom_theme

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

set_custom_theme("REVIEW GRAPH NEURAL NETWORK AND APPLICATIONS")
image = Image.open('./assets/img/review_gnn.png')
st.image(image, caption='Graph Neural Networks - An overview')