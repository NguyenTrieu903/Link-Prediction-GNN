import streamlit as st
from assets.theme import set_custom_theme, display_picture

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

set_custom_theme("REVIEW GRAPH NEURAL NETWORK AND APPLICATIONS")
display_picture('review_gnn.png', 'Graph Neural Networks - An overview')