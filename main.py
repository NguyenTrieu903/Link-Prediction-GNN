from LogisticRegression_Linkprediction.data.understanding_data import create_graph, plot_graph
from LogisticRegression_Linkprediction.model.link_prediction import link_prediction_with_logistic
from LogisticRegression_Linkprediction.data.understanding_data import *
# from SEAL.config.data import load_data
from SEAL.operators.seal_link_predict import execute
from TwoWL import TwoWL_work
import warnings
warnings.filterwarnings("ignore")
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Link prediction with GNN.")
    parser.add_argument("--model", type=str, help="model name.", default="SEAL")
    parser.add_argument('--dataset', type=str, default="fb-pages-food")
    parser.add_argument('--pattern', type=str, default="2wl_l")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--episode', type=int, default=200)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--path', type=str, default="Opt/")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--check', action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.model == 'Logistic':
        link_prediction_with_logistic()
    if args.model == 'SEAL':
        execute(0, 0.1, 100, "auto",0.00001)
    if args.model == 'TwoWL':
        args.device="cpu"
        TwoWL_work.work(args, args.device)