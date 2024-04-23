from LogisticRegression_Linkprediction.model import build_model
from LogisticRegression_Linkprediction.model.build_model import split_data
from LogisticRegression_Linkprediction.data import dataset_preparation, understanding_data
from LogisticRegression_Linkprediction.utils.feature_extraction import feature_extraction
from LogisticRegression_Linkprediction.data.understanding_data import load_data
from constant import *


def link_prediction_with_logistic():
    fb_df, node_list_1, node_list_2 = load_data()
    G = understanding_data.create_graph(fb_df)

    data = dataset_preparation.retrieve_unconnected(node_list_1, node_list_2, G)
    omissible_links_index = dataset_preparation.remove_link_connected(fb_df, G)
    data, fb_df_ghost = dataset_preparation.data_for_model_training(fb_df, omissible_links_index, data)

    x = feature_extraction(fb_df, fb_df_ghost, data)

    xtrain, xtest, ytrain, ytest = split_data(data, x)
    build_model.logistic_regression(xtrain, xtest, ytrain, ytest)

def read_the_results_logistic():
    with open(PATH_SAVE_TEST_AUC + 'fb-pages-food_auc_record_logistic.txt', 'r') as f:
        data = f.readlines()
    for line in data:
        line = line.strip()
        if line:
            auc, time = line.split()
            auc_value = float(auc.split(":")[1])
            annotations_auc_logis = float(time.split(":")[1])
    return auc_value , annotations_auc_logis



    
