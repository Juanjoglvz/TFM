import argparse
from os.path import join

import numpy as np
from joblib import dump, load
from numpy import loadtxt
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import SVC

from NLP.system.Neutrals.Preprocess import preprocess, load_kaggle, load_ml_senticon, init_photo_labels
from NLP.system.Parse_xml import parse_corpus_and_gt, parse_ml_senticon


def train_svc(path_to_corpus_es, path_to_gt_es, path_to_save_model, path_to_sentiments, weights,
              x_train_path, y_train_path, x_test_path, y_test_path, path_to_neutrals, path_to_vocabulary):

    if path_to_corpus_es and path_to_gt_es:
        if path_to_sentiments:
            load_kaggle(path_to_sentiments)
            ml_senticon_es = parse_ml_senticon(path_to_sentiments, "es")
            ml_senticon_ca = parse_ml_senticon(path_to_sentiments, "ca")
            load_ml_senticon(ml_senticon_es, ml_senticon_ca)

        corpus_es, _, _, total_ground_truth_es, photos_es = parse_corpus_and_gt(
            join(path_to_corpus_es, "es.xml"), join(path_to_gt_es, "truth-es.txt"))
        corpus_ca, _, _, total_ground_truth_ca, photos_ca = parse_corpus_and_gt(
            join(path_to_corpus_es, "ca.xml"), join(path_to_gt_es, "truth-ca.txt")
        )
        n_spanish = len(total_ground_truth_es)
        corpus_es.update(corpus_ca)
        total_ground_truth_es.update(total_ground_truth_ca)
        ground_truth_list = []
        for key, value in total_ground_truth_es.items():
            ground_truth_list.append({key: value})
        photos_es.update(photos_ca)
        init_photo_labels("F:\\MultiStanceCat-IberEval-training-20180404\\output_labels")
        X_train, X_test, Y_train, Y_test, true_y, vocabulary, idf = preprocess(corpus_es, ground_truth_list,
                                                                               n_spanish, path_to_vocabulary, photos_es)

    elif x_train_path and y_train_path:
        print("Reading data from {}".format(x_train_path))
        X_train = loadtxt(x_train_path, dtype=float, delimiter=',')
        Y_train = loadtxt(y_train_path, dtype=float, delimiter=',')
        X_test = loadtxt(x_test_path, dtype=float, delimiter=',')
        Y_test = loadtxt(y_test_path, dtype=float, delimiter=',')

    if path_to_neutrals:
        clf_neutrals = load(path_to_neutrals)


    retval = []

    # Train various classifiers with best params
    best_clf = None
    best_f1 = 0
    print("Starting the training")
    names = ["SVM 1"]

    classifiers = [SVC(kernel="linear", C=1, max_iter=100000, probability=True)]

    for name, clf in zip(names, classifiers):
        print(name)
        current = {}

        Y_train_polar = Y_train[Y_train != 1]
        X_train_polar = X_train[Y_train != 1, :]
        clf.fit(X_train_polar, Y_train_polar)

        neutrals_predict = clf_neutrals.predict(X_test)
        #X_test = X_test[neutrals_predict == 0, :]
        #Y_test = Y_test[neutrals_predict == 0]
        Y_pred = clf.predict(X_test)

        Y_pred[neutrals_predict == 1] = 1

        cm = confusion_matrix(Y_test, Y_pred, labels=[0,1,2])
        print("\t\t\t\tFavor\tNeutral\tAgainst")
        print("Favor\t\t\t{}\t{}\t{}".format(cm[0, 0], cm[0, 1], cm[0, 2]))
        print("Neutral\t\t\t\t{}\t{}\t{}".format(cm[1, 0], cm[1, 1], cm[1, 2]))
        print("Against\t\t\t\t{}\t{}\t{}".format(cm[2, 0], cm[2, 1], cm[2, 2]))

        score = f1_score(Y_test, Y_pred, labels=[0, 1, 2], average="macro")
        print("Macro F1: {}".format(score))
        if score > best_f1:
            best_clf = clf
            best_f1 = score
        current["name"] = name
        current["cm"] = cm
        current["score"] = score
        retval.append(current)

    print(best_clf)

    # Fit a classifier
    #clf = LinearSVC(verbose=1, random_state=7, tol=1e-5, max_iter=100000)
    #clf.fit(X_train, Y_train)

    if path_to_save_model:
        print("saving model and data")
        # save model with dump. Load it with joblib.load
        dump(best_clf, join(path_to_save_model, "polar", "clf.joblib"))
        np.savetxt(join(path_to_save_model, "polar", "train_data.csv"), X_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "polar", "train_truth.csv"), Y_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "polar", "eval_data.csv"), X_test, delimiter=",")
        np.savetxt(join(path_to_save_model, "polar", "eval_truth.csv"), Y_test, delimiter=",")
        if vocabulary:
            for i in range(len(vocabulary)):
                with open(join(path_to_save_model, "polar", "vocabulary{}.csv".format(i)), "w+") as f:
                    for key in vocabulary[i].keys():
                        f.write("{}, {}\n".format(key, vocabulary[i][key]))
        if idf:
            for i in range(len(idf)):
                with open(join(path_to_save_model, "polar", "idf{}.csv".format(i)), "w+") as f:
                    for value in idf[i]:
                        f.write("{}\n".format(value))
        print("Saved")

    return retval


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
    parser.add_argument("--neutrals", help="Path to neutrals model")
    parser.add_argument("--vocabulary", help="Path to vocabulary for neutrals preprocessing")

    args = parser.parse_args()
    train_svc(args.path_es, args.truth_es, args.save, args.senti, None, args.x_train, args.y_train, args.x_test, args.y_test, args.neutrals, args.vocabulary)
