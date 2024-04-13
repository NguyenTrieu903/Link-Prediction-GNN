import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import train_test_split


def split_data(data, x):
    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.3, 
                                                random_state = 35)
    return xtrain, xtest, ytrain, ytest
    

def logistic_regression(xtrain, xtest, ytrain, ytest):
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(xtrain, ytrain)
    predictions = lr.predict_proba(xtest)
    roc = roc_auc_score(ytest, predictions[:,1])
    print("Roc auc score with logistic regression : " ,roc)

def avcx():
    print("asc")