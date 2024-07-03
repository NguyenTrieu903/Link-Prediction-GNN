import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from assets.theme import *


def split_data(data, x):
    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                    test_size=0.3,
                                                    random_state=35)
    return xtrain, xtest, ytrain, ytest


def logistic_regression(xtrain, xtest, ytrain, ytest):
    start_t = time.time()
    # st.title("Logistic Regression Model")
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(xtrain, ytrain)
    predictions = lr.predict_proba(xtest)
    end_t = time.time()
    roc = roc_auc_score(ytest, predictions[:, 1])
    plot_auc(ytest, predictions[:, 1], roc, name="roc_curve_logistic.png")

    # st.write("### Roc auc score with logistic regression: ", roc)
    st.write("#### Time consumption: ", end_t - start_t)
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_logistic.txt', 'w') as f:
        f.write('AUC:' + str(round(roc, 4)) + '   ' + 'Time:' +
                str(round(end_t - start_t, 4)) + '\n')
