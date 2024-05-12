import streamlit as st
from PIL import Image
from constant import *
import matplotlib.pyplot as plt
from sklearn import metrics
import os
from sklearn.metrics import  roc_curve
import json
import plotly.graph_objects as go
import numpy as np


def set_custom_theme(title):
    st.write(f"<h2 style='text-align: center; color: red; font-size: 38px; font-weight: bold; font-family: Times New Roman, sans-serif ;'>{title}</h2>", unsafe_allow_html=True)
    page_bg_img = '''
        <style>
        .stApp {
            background-image: url("https://img.freepik.com/free-vector/blue-curve-background_53876-113112.jpg");
            background-size: 100vw 100vh; 
            background-position: center;  
            background-repeat: no-repeat;
        }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

def update_time( time_display, start_time, end_time):
    global seconds_passed
    if 'seconds_passed' not in globals():
        seconds_passed = 0
    seconds_passed = seconds_passed + ( end_time - start_time)
    time_now = f"Time elapsed: {seconds_passed} seconds"
    time_display.empty()
    time_display.text(time_now)

def display_picture(a, b):
    image = Image.open( PATH_PICTURE +  a)
    st.image(image, caption=b)

def plot_auc(ytest, predictions, roc, name):
    # Tính toán FPR và TPR cho mỗi ngưỡng quyết định
    fpr, tpr, thresholds = roc_curve(ytest, predictions)

    # Vẽ biểu đồ ROC
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    # lưu biểu đồ ROC
    os.makedirs(PATH_PICTURE, exist_ok=True)
    output_path = os.path.join(PATH_PICTURE, name)
    plt.savefig(output_path)

    # Hiển thị biểu đồ trên Streamlit
    st.pyplot(fig)
    
def creat_pylot_twowl(values, info_values, auc, istrain):
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
    if istrain == False:
        fig.add_trace(go.Scatter(x=axis_x, y=values, mode="markers+lines", name="Values", line=dict(color="green"),
                        text=annotations_value, hovertemplate="<b>Value:</b> %{y}<br><b>Annotation:</b> %{text}"))
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

def plot_auc_with_twowl(roc, name):

    # Đọc dữ liệu từ tệp JSON
    with open('fpr.json', 'r') as f:
        fpr_list = json.load(f)
    fpr= np.array(fpr_list, dtype=float)

    with open('tpr.json', 'r') as f:
        tpr_list = json.load(f)
    tpr= np.array(tpr_list, dtype=float)

    # Vẽ biểu đồ ROC
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    # lưu biểu đồ ROC
    os.makedirs(PATH_PICTURE, exist_ok=True)
    output_path = os.path.join(PATH_PICTURE, name)
    plt.savefig(output_path)

    # Hiển thị biểu đồ trên Streamlit
    st.pyplot(fig)