import argparse
import numpy as np
from joblib import load
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from NLP.system.Neutrals.Train import preprocess
from NLP.system.Neutrals.Parse_xml import parse_corpus_and_gt


def evaluate_svc(path_to_corpus, path_to_gt, path_to_model):
    print("Loading data")
    X_test = np.loadtxt(path_to_corpus, dtype=float, delimiter=",")
    Y_test = np.loadtxt(path_to_gt, dtype=float, delimiter=",")
    print("Loading model")
    clf = load(path_to_model)
    Y_pred = clf.predict(X_test)

    cm = confusion_matrix(Y_test, Y_pred)
    print("\t\t\t\tNot neutral\tNeutral")
    print("Not neutral\t\t\t{}\t{}".format(cm[0,0], cm[0,1]))
    print("Neutral\t\t\t\t{}\t{}".format(cm[1, 0], cm[1, 1]))

    res = precision_score(Y_test, Y_pred, labels=[0, 1], average=None)
    for i in range(len(res)):
        print("Precision score for class {}: {}".format(i, res[i]))

    res = f1_score(Y_test, Y_pred, labels=[0,1], average=None)
    for i in range(len(res)):
        print("F1 score for class {}: {}".format(i, res[i]))

    print("Macro F1: {}".format(f1_score(Y_test, Y_pred, labels=[0,1], average="macro")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Path to eval_data.csv.")
    parser.add_argument("--truth", help="Path to ground truth.")
    parser.add_argument("--model", help="Path to model")
    args = parser.parse_args()
    evaluate_svc(args.corpus, args.truth, args.model)
