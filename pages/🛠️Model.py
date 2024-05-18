import streamlit as st
from LogisticRegression_Linkprediction.data.understanding_data import create_graph, plot_graph
from LogisticRegression_Linkprediction.model.link_prediction import link_prediction_with_logistic, read_the_results_logistic
from SEAL.operators.seal_link_predict import execute, read_the_results_seal
from TwoWL import TwoWL_work
import warnings
import matplotlib.pyplot as plt
from constant import *
import argparse
from assets.theme import *

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Model", page_icon="🛠️")


def link_prediction_menu(model_option, train):
    if model_option == "Logistic":
        if train:
            link_prediction_with_logistic()
        else:
            display_picture('roc_curve_logistic.png', 'Roc auc score with logistic regression')
            auc_value , time = read_the_results_logistic()
            st.write("### Logistic model runtime: ", time)
    elif model_option == "SEAL":
        if train:
            execute(0, 0.1, 100, "auto", 0.00001)
        else:
            display_picture('roc_curve_seal.png', 'Roc auc score with seal framework')
            auc_value, time_value = read_the_results_seal()
            st.write("#### Time consumption: ", time_value)
    elif model_option == "TwoWL":
        if train:
            # resest file
            with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_twowl.txt', 'w') as f:
                f.write("")
            args = argparse.Namespace(model="TwoWL", dataset="fb-pages-food", pattern="2wl_l", epoch=100, episode=500, seed=0, device="cpu", path="Opt/", test=False, check=False)
            TwoWL_work.work(args, args.device) 
            values, info_values, auc, best_auc_twowl = TwoWL_work.read_results_twowl()
            creat_pylot_twowl(values, info_values, auc, True)
            plot_auc_with_twowl(roc=best_auc_twowl, name = 'roc_curve_twowl')
        else:
            values, info_values, auc, best_auc_twowl = TwoWL_work.read_results_twowl()
            display_picture('roc_curve_twowl.png', 'Roc auc score with twowl')
            creat_pylot_twowl(values, info_values, auc, False)
        
    elif model_option == "Compare":
        # Lấy giá trị AUC của SEAL
        results_seal, time_value= read_the_results_seal()
        # Lấy giá trị AUC lớn nhất của mô hình TwoWL
        values, info_values, auc, best_auc_twowl = TwoWL_work.read_results_twowl()
        
        # Lấy giá trị AUC của Logistic
        results_logistic, time_logistic = read_the_results_logistic()

        # Vẽ biểu đồ so sánh
        Model = ['Logistic', 'SEAL', 'TwoWL']
        Auc = [results_logistic, results_seal, best_auc_twowl]
        fig, ax = plt.subplots()
        ax.bar(Model, Auc)
        ax.set_xlabel('Model', fontweight='bold', fontsize=10, color="green")
        ax.set_ylabel('AUC', fontweight='bold', fontsize=10, color="green")
        ax.set_title('Chart comparing AUC values ​​between models', fontweight='bold', fontsize=14, color="red")
        for i, v in enumerate(Auc):
            ax.text(i, v, str(v), ha='center', va='bottom')
        st.pyplot(fig)



def main():
    selected_tab = st.sidebar.radio("**Option**", ["Logistic", "SEAL", "TwoWL", "Compare"])
    train = st.sidebar.checkbox("**TRAIN**")  #run train ?
    submitted = st.sidebar.button("**RUN**", type="primary")
    if submitted:
        link_prediction_menu(selected_tab, train)
    else:
        display_picture('review_gnn.png', 'Graph Neural Networks - An overview')
        

if __name__ == "__main__":
    set_custom_theme("LINK PREDICTION")
    main()
