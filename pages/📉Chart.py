import warnings

import streamlit as st

from LogisticRegression_Linkprediction.data.understanding_data import create_graph, load_data
from assets.theme import set_custom_theme, display_picture

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Chart", page_icon="📉")


def main():
    fb_df, node_list_1, node_list_2 = load_data()
    G = create_graph(fb_df)
    # Display graph
    # plot_graph(G)
    col1, col2 = st.columns([4, 3])
    with col1:
        display_picture('Facebook_gnn.png', 'Graph Visualization')
    with col2:
        display_picture('network_data_statistics.png', 'Network Data Statistics')
    col3, col4 = st.columns([1, 1])
    with col3:
        display_picture('metadata.png', 'Metadata')
    with col4:
        display_picture('data_review.png', 'Network Data Preview')


if __name__ == "__main__":
    set_custom_theme("GRAPH VISUALIZATION")
    main()
