import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import  roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
import streamlit as st
from constant import *
import time

def split_data(data, x):
    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.3, 
                                                random_state = 35)
    return xtrain, xtest, ytrain, ytest
    

def logistic_regression(xtrain, xtest, ytrain, ytest):
    start_t = time.time()
    #st.title("Logistic Regression Model")
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(xtrain, ytrain)
    predictions = lr.predict_proba(xtest)
    end_t = time.time()
    roc = roc_auc_score(ytest, predictions[:,1])
    #print("Roc auc score with logistic regression : " ,roc)
    st.write("### Roc auc score with logistic regression: ", roc)
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_logistic.txt', 'w') as f:
            f.write('AUC:' + str(round(roc, 4)) + '   ' + 'Time:' + 
                    str(round(end_t - start_t, 4)) + '\n')


# def logistic_regression(xtrain, xtest, ytrain, ytest):

#     progress_bar = st.sidebar.progress(0)
#     status_text = st.sidebar.empty()
#     last_rows = np.zeros((1, 1))
#     chart = st.line_chart()

#     # Tạo mô hình logistic regression với class_weight="balanced" và số lần lặp lại (epoch) là 100
#     lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    
#     train_accs = []
#     test_accs = []

#     # Huấn luyện mô hình trên dữ liệu xtrain, ytrain
#     for epoch, i in zip(range(1000), range(1, 1001)):
#         lr.fit(xtrain, ytrain)
        
#         train_acc = lr.score(xtrain, ytrain)
#         test_acc = lr.score(xtest, ytest)
#         print(f"Epoch {epoch+1}, Training Acc: {train_acc}, Test Acc: {test_acc}")
        
#         train_accs.append(train_acc)
#         test_accs.append(test_acc)

#         status_text.text("%i%% Complete" % (i/10))  # Hiển thị tiến độ
#         new_rows = np.full((1, 1), train_acc)
#         chart.add_rows(new_rows)
#         progress_bar.progress(i)  # Hiển thị tiến độ trên thanh tiến trình
#         time.sleep(0.01)  # Đợi 0.01 giây để mô phỏng quá trình huấn luyện
#     progress_bar.empty()
    
#     predictions = lr.predict_proba(xtest)
#     #end_t = time.time()
#     roc = roc_auc_score(ytest, predictions[:,1])
#     st.write("### Roc auc score with logistic regression: ", roc)

