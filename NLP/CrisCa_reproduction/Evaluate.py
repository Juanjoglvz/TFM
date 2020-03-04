import argparse
import numpy as np
from joblib import load
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def evaluate_svc(path_to_corpus, path_to_gt, path_to_model):
    X_test = np.loadtxt(path_to_corpus, dtype=float, delimiter=",")
    Y_test = np.loadtxt(path_to_gt, dtype=float, delimiter=",")
    clf = load(path_to_model)
    Y_pred = clf.predict(X_test)
    prec = precision_score(Y_test, Y_pred, labels=[0,1,2], average=None)
    for i in range(len(prec)):
        print("Precision for class {}: {}".format(i, prec[i]))
    fscore = f1_score(Y_test, Y_pred, labels=[0,1,2], average="macro")
    print("F_score: {}".format(fscore))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Path to eval_data.csv")
    parser.add_argument("--truth", help="Path to ground truth")
    parser.add_argument("--model", help="Path to model")

    args = parser.parse_args()
    evaluate_svc(args.corpus, args.truth, args.model)
