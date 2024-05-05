import streamlit as st
from LogisticRegression_Linkprediction.data.understanding_data import create_graph, plot_graph, load_data
from assets.theme import set_custom_theme
import warnings
from PIL import Image

warnings.filterwarnings("ignore")
import argparse

st.set_page_config(page_title="Chart", page_icon="📉")


def main():
    fb_df, node_list_1, node_list_2 = load_data()
    G = create_graph(fb_df)

    # Display graph
    plot_graph(G)

if __name__ == "__main__":
    set_custom_theme("GRAPH VISUALIZATION")
    image = Image.open('./assets/img/Facebook_gnn.png')
    st.image(image)
    # main()
