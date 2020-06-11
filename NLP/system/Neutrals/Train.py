import argparse
from os.path import join
import numpy as np
from joblib import dump
from sklearn.svm import LinearSVC
from NLP.system.Neutrals.Preprocess import preprocess, load_kaggle
from NLP.system.Neutrals.Parse_xml import parse_corpus_and_gt
from NLP.system.Graphics import barplot


def train_svc(path_to_corpus_es, path_to_gt_es, path_to_save_model, path_to_sentiments):
    if path_to_sentiments:
        load_kaggle(path_to_sentiments)

    corpus_es, ground_truth_es = parse_corpus_and_gt(path_to_corpus_es, path_to_gt_es)
    X_train, X_test, Y_train, Y_test, true_y = preprocess(corpus_es, ground_truth_es)

    barplot(true_y, ["Not Neutral", "Neutral"], join(path_to_save_model, "Class_dist"))

    # Fit classifier
    print("Starting the training")
    clf = LinearSVC(verbose=1, random_state=7, tol=1e-5, max_iter=100000)
    clf.fit(X_train, Y_train)

    if path_to_save_model:
        print("saving model and data")
        # save model with dump. Load it with joblib.load
        dump(clf, join(path_to_save_model, "svc_neutrals.joblib"))
        np.savetxt(join(path_to_save_model, "train_data.csv"), X_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "train_truth.csv"), Y_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "eval_data.csv"), X_test, delimiter=",")
        np.savetxt(join(path_to_save_model, "eval_truth.csv"), Y_test, delimiter=",")
        print("Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_es", help="Path to es.xml")
    parser.add_argument("--truth_es", help="Path to spanish ground truth")
    parser.add_argument("--save", help="Path to save model")
    parser.add_argument("--senti", help="Path to sentiment datasets")

    args = parser.parse_args()
    train_svc(args.path_es, args.truth_es, args.save, args.senti)
