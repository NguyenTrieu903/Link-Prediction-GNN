import streamlit as st
from assets.theme import set_custom_theme
from LogisticRegression_Linkprediction.data.understanding_data import create_graph, plot_graph
from LogisticRegression_Linkprediction.model.link_prediction import link_prediction_with_logistic, read_the_results_logistic
from SEAL.operators.seal_link_predict import execute, read_the_results_seal
# from TwoWL import TwoWL_work
import json
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from constant import *
from PIL import Image
import argparse

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Model", page_icon="🛠️")

def creat_pylot_twowl(values, info_values, auc):
    # Tạo danh sách các giá trị trục x
    axis_x = list(range(1, len(values) + 1))

    # Tạo danh sách các chú thích
    annotations_value = []
    for log in info_values:
        annotation = json.dumps(log, indent=4)
        annotations_value.append(annotation) 

    values_auc = []
    annotations_auc = []
    for line in auc:
        line = line.strip()
        if line:
            AUC, time = line.split()
            #x_txt.append(len(x_txt) + 1)
            values_auc.append(float(AUC.split(":")[1]))
            annotations_auc.append(float(time.split(":")[1]))
    
    # Tạo biểu đồ đường kết hợp với điểm và chú thích
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=axis_x, y=values, mode="markers+lines", name="Values", line=dict(color="green"),
    #                 text=annotations_value, hovertemplate="<b>Value:</b> %{y}<br><b>Annotation:</b> %{text}"))
    fig.add_trace(go.Scatter(x=axis_x, y=values_auc, mode="markers+lines", name="AUC", line=dict(color="blue"),
                    text= annotations_auc, hovertemplate="<b>AUC:</b> %{y}<br><b>Time:</b> %{text}"))
    fig.update_layout(hovermode="closest", hoverdistance=10)
    for xi in axis_x:
        fig.add_vline(x=xi, line_dash="dot", line_color="lightgrey", opacity=0.5, name="Vertical Line")
    fig.update_layout(title="<b>THE CHART SHOWS THE CHANGE IN VALUE AND AUC ACCORDING TO EACH TRIAL</b>", title_font=dict(color="red", size=20))
    fig.update_layout(yaxis_title="<b>Values</b>", xaxis_title ="<b>Trials</b>")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black")
    st.plotly_chart(fig)

def read_results_twowl():
    auc_twowl = "fb-pages-food_auc_record_twowl.txt"

    with open("values.json", "r") as f1, open("logs.json", "r") as f2, open(PATH_SAVE_TEST_AUC + auc_twowl, "r") as f3:
        values = json.load(f1)
        info_values = json.load(f2)
        auc = f3.readlines()
    return values, info_values, auc

def link_prediction_menu(model_option, train):

    if model_option == "Logistic":
        # if train:
        #     link_prediction_with_logistic()
        # else:
            auc_value , time = read_the_results_logistic()
            st.write("### Roc auc score with logistic regression: ", auc_value)
            st.write("### Logistic model runtime: ", time)
    elif model_option == "SEAL":
        # if train:
        #     execute(0, 0.1, 100, "auto", 0.00001)
        # else:
            auc_value, time_value, test_acc_value, pos_score_value_one, prediction_one = read_the_results_seal()
            st.write("#### AUC: " ,auc_value)
            st.write("#### Time consumption: ", time_value)
            # st.write("#### Pos_score_value: ",pos_score_value_one)
            # st.write("#### The predicted probability for the first element is:", prediction_one)
    elif model_option == "TwoWL":
        # if train:
        #     # Để hạn ché bị lỗi thì trước khi chạy sẽ resest file txt trước khi chạy để hạn chế bị lỗi
        #     with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_twowl.txt', 'w') as f:
        #         f.write("")
        #     args = argparse.Namespace(model="TwoWL", dataset="fb-pages-food", pattern="2wl_l", epoch=1000, episode=500, seed=0, device="cpu", path="Opt/", test=False, check=False)
        #     TwoWL_work.work(args, args.device)
        #     values, info_values, auc = TwoWL_work.read_results_twowl()
        #     creat_pylot_twowl(values, info_values, auc)
        # else:
            values, info_values, auc =read_results_twowl()
            creat_pylot_twowl(values, info_values, auc)
        
    elif model_option == "Compare":
        # Lấy giá trị AUC của SEAL
        results_seal, time_value, test_acc_value, pos_score_value_one, prediction_one = read_the_results_seal()
        # Lấy giá trị AUC lớn nhất của mô hình TwoWL
        values, info_values, auc = read_results_twowl()
        best_auc_twowl = 0.0
        annotations_auc_twowl = 0.0
        for line in auc:
            line = line.strip()
            if line:
                AUC, time = line.split()
                AUC = float(AUC.split(":")[1])
                if AUC >= best_auc_twowl:
                    best_auc_twowl = AUC
                    #annotations_auc_twowl = float(time.split(":")[1])
        
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
    train = st.sidebar.checkbox("**TRAIN**")  #có muốn thực hiện quá trình train không
    submitted = st.sidebar.button("**RUN**", type="primary")
    # if st.sidebar.button("RUN", type="primary"):
    if submitted:
        link_prediction_menu(selected_tab, train)
    else:
        image = Image.open('./assets/img/review_gnn.png')
        st.image(image, caption='Graph Neural Networks - An overview')

if __name__ == "__main__":
    set_custom_theme("LINK PREDICTION")
    main()
