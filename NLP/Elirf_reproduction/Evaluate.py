import argparse
import numpy as np
from joblib import load
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from NLP.CrisCa_reproduction.Train import preprocess
from NLP.CrisCa_reproduction.Parse_xml import parse_corpus_and_gt


def evaluate_svc(path_to_corpus, path_to_gt, path_to_model):
    print("Loading data")
    X_test = np.loadtxt(path_to_corpus, dtype=float, delimiter=",")
    Y_test = np.loadtxt(path_to_gt, dtype=float, delimiter=",")
    print("Loading model")
    clf = load(path_to_model)
    Y_pred = clf.predict(X_test)
    res = f1_score(Y_test, Y_pred, labels=[0,1,2], average=None)
    for i in range(len(res)):
        print("F1 score for class {}: {}".format(i, res[i]))
    print("Final_score: {}".format((res[0] + res[2]) / 2))

    print("Macro F1: {}".format(f1_score(Y_test, Y_pred, labels=[0,1,2], average="macro")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Path to eval_data.csv.")
    parser.add_argument("--truth", help="Path to ground truth.")
    parser.add_argument("--model", help="Path to model")
    args = parser.parse_args()
    evaluate_svc(args.corpus, args.truth, args.model)
