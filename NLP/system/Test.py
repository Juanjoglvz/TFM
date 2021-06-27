import argparse
from os.path import join as path_join

import numpy as np
from joblib import dump, load
from numpy import loadtxt
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import SVC

from NLP.system.Neutrals.Preprocess import preprocess, load_kaggle, load_ml_senticon, init_photo_labels
from NLP.system.Parse_xml import parse_corpus_and_gt, parse_ml_senticon


def train_svc(path_to_corpus_es, path_to_gt_es, path_to_sentiments, path_to_models, path_to_vocabulary,
              language):
    # Neutrals test
    if path_to_sentiments:
        load_kaggle(path_to_sentiments)
        ml_senticon = parse_ml_senticon(path_to_sentiments, language)
        if language == "es":
            load_ml_senticon(ml_senticon, None)
        elif language == "ca":
            load_ml_senticon(None, ml_senticon)
    corpus, _, _, total_ground_truth, photos = parse_corpus_and_gt(
        path_join(path_to_corpus_es, f"{language}.xml"), path_join(path_to_gt_es, f"truth-{language}.txt"))

    n_spanish = len(total_ground_truth) if language == "es" else 0

    ground_truth_list = []
    for key, value in total_ground_truth.items():
        ground_truth_list.append({key: value})

    init_photo_labels("F:\\MultiStanceCat-IberEval-training-20180404\\output_labels")
    X_train, X_test, Y_train, Y_test, true_y, vocabulary, idf = preprocess(corpus, ground_truth_list,
                                                                           n_spanish,
                                                                           path_join(path_to_vocabulary, "neutrals"),
                                                                                     photos)

    Y_test_neutrals = Y_test
    Y_test_neutrals[Y_test_neutrals == 2] = 0

    clf_neutrals = load(path_join(path_to_models, "neutrals", "clf_neutrals.joblib"))
    neutrals_predict = clf_neutrals.predict(X_test)
    print("Neutrals classifier")
    cm = confusion_matrix(Y_test_neutrals, neutrals_predict, labels=[0, 1, 2])
    print("\t\t\t\tNot neutral\tNeutral")
    print("Not neutral\t\t\t{}\t{}".format(cm[0, 0], cm[0, 1]))
    print("Neutral\t\t\t\t{}\t{}".format(cm[1, 0], cm[1, 1]))

    score = f1_score(Y_test, neutrals_predict, labels=[0, 1, 2], average="macro")
    print("Macro F1: {}".format(score))


    # Polar test
    if path_to_sentiments:
        load_kaggle(path_to_sentiments)
        ml_senticon = parse_ml_senticon(path_to_sentiments, language)
        if language == "es":
            load_ml_senticon(ml_senticon, None)
        elif language == "ca":
            load_ml_senticon(None, ml_senticon)

    n_spanish = len(total_ground_truth) if language == "es" else 0

    ground_truth_list = []
    for key, value in total_ground_truth.items():
        ground_truth_list.append({key: value})

    init_photo_labels("F:\\MultiStanceCat-IberEval-training-20180404\\output_labels")
    X_train, X_test, Y_train, Y_test, true_y, vocabulary, idf = preprocess(corpus, ground_truth_list,
                                                                           n_spanish,
                                                                           path_join(path_to_vocabulary, "polar"),
                                                                                     photos)


    clf_polar = load(path_join(path_to_models, "polar", "clf.joblib"))

    Y_pred = clf_polar.predict(X_test)
    Y_pred[neutrals_predict == 1] = 1

    print("Polar classifier")
    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1, 2])
    print("\t\t\t\tFavor\tNeutral\tAgainst")
    print("Favor\t\t\t{}\t{}\t{}".format(cm[0, 0], cm[0, 1], cm[0, 2]))
    print("Neutral\t\t\t\t{}\t{}\t{}".format(cm[1, 0], cm[1, 1], cm[1, 2]))
    print("Against\t\t\t\t{}\t{}\t{}".format(cm[2, 0], cm[2, 1], cm[2, 2]))

    score = f1_score(Y_test, Y_pred, labels=[0, 1, 2], average="macro")
    print("Macro F1: {}".format(score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_es", help="Path to es.xml")
    parser.add_argument("--truth_es", help="Path to spanish ground truth")
    parser.add_argument("--senti", help="Path to sentiment datasets")
    parser.add_argument("--models", help="Path to both models")
    parser.add_argument("--vocabulary", help="Path to vocabulary for neutrals preprocessing")
    parser.add_argument("--language", help="Either <es> or <cat>")

    args = parser.parse_args()
    train_svc(args.path_es, args.truth_es, args.senti, args.models, args.vocabulary, args.language)
