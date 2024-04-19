import streamlit as st
from LogisticRegression_Linkprediction.data.understanding_data import create_graph, plot_graph, load_data
import warnings
warnings.filterwarnings("ignore")
import argparse

st.set_page_config(page_title="Plot", page_icon="📉")

#app_mode = st.session_state['app_mode']

def main():
    st.title("Graph Visualization")
    fb_df, node_list_1, node_list_2 = load_data()
    G = create_graph(fb_df)

    # Display graph
    st.subheader("Graph Visualization")
    plot_graph(G)

if __name__ == "__main__":
    main()
