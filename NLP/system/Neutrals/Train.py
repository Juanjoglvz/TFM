import argparse
from os.path import join
import numpy as np
from joblib import dump
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from NLP.system.Neutrals.Preprocess import preprocess, load_kaggle, load_ml_senticon
from NLP.system.Neutrals.Parse_xml import parse_corpus_and_gt, parse_ml_senticon
from NLP.system.Graphics import barplot
from numpy import loadtxt
from sklearn.metrics import f1_score, confusion_matrix



def train_svc(path_to_corpus_es, path_to_gt_es, path_to_save_model, path_to_sentiments, x_train_path, y_train_path, x_test_path, y_test_path):
    if path_to_corpus_es and path_to_gt_es:
        if path_to_sentiments:
            load_kaggle(path_to_sentiments)
            ml_senticon = parse_ml_senticon(path_to_sentiments)
            load_ml_senticon(ml_senticon)

        corpus_es, ground_truth_es = parse_corpus_and_gt(path_to_corpus_es, path_to_gt_es)
        X_train, X_test, Y_train, Y_test, true_y, vocabulary = preprocess(corpus_es, ground_truth_es)
    elif x_train_path and y_train_path:
        print("Reading data from {}".format(x_train_path))
        X_train = loadtxt(x_train_path, dtype=float, delimiter=',')
        Y_train = loadtxt(y_train_path, dtype=float, delimiter=',')
        X_test = loadtxt(x_test_path, dtype=float, delimiter=',')
        Y_test = loadtxt(y_test_path, dtype=float, delimiter=',')

    #barplot(true_y, ["Not Neutral", "Neutral"], join(path_to_save_model, "Class_dist"))

    # Train various classifiers with best params
    best_clf = None
    best_f1 = 0
    print("Starting the search")
    names = ["Logistic Regression",
             "Linear SVM",
             "kNN",
             "Random Forest"]

    classifiers = [LogisticRegression(C=54.5559478116852, max_iter=100000),
                   LinearSVC(C=2.976351441631313, max_iter=100000),
                   KNeighborsClassifier(n_neighbors=3),
                   RandomForestClassifier(max_depth=1, n_estimators=100)]

    for name, clf in zip(names, classifiers):
        print(name)
        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)
        cm = confusion_matrix(Y_test, Y_pred)
        print("\t\t\t\tNot neutral\tNeutral")
        print("Not neutral\t\t\t{}\t{}".format(cm[0, 0], cm[0, 1]))
        print("Neutral\t\t\t\t{}\t{}".format(cm[1, 0], cm[1, 1]))

        score = f1_score(Y_test, Y_pred, labels=[0, 1], average="macro")
        print("Macro F1: {}".format(score))
        if score > best_f1:
            best_clf = clf
            best_f1 = score

    print(best_clf)

    # Fit a classifier
    #clf = LinearSVC(verbose=1, random_state=7, tol=1e-5, max_iter=100000)
    #clf.fit(X_train, Y_train)

    if path_to_save_model:
        print("saving model and data")
        # save model with dump. Load it with joblib.load
        dump(best_clf, join(path_to_save_model, "clf_neutrals.joblib"))
        np.savetxt(join(path_to_save_model, "train_data.csv"), X_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "train_truth.csv"), Y_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "eval_data.csv"), X_test, delimiter=",")
        np.savetxt(join(path_to_save_model, "eval_truth.csv"), Y_test, delimiter=",")
        print("Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_es", help="Path to es.xml")
    parser.add_argument("--truth_es", help="Path to spanish ground truth")
    parser.add_argument("--save", help="Path to save model")
    parser.add_argument("--senti", help="Path to sentiment datasets")
    parser.add_argument("--x_train", help="Path to X_train in case we want to use presaved data")
    parser.add_argument("--y_train", help="Path to Y_train in case we want to use presaved data")
    parser.add_argument("--x_test", help="Path to X_test in case we want to use presaved data")
    parser.add_argument("--y_test", help="Path to Y_test in case we want to use presaved data")

    args = parser.parse_args()
    train_svc(args.path_es, args.truth_es, args.save, args.senti, args.x_train, args.y_train, args.x_test, args.y_test)
