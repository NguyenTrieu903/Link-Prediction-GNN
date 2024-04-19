import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import train_test_split
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

