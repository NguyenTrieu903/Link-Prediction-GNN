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

