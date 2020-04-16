from NLP.uc3m_reproduction.Parse_xml import parse_corpus_and_gt
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import dump
from os.path import join
from copy import deepcopy


def preprocess(corpus, ground_truth):
    preprocessed_corpus = []
    true_y = []

    for identifier, doc in corpus.items():
        # doc = corpus["af9f7be8eddca053f705decaf6b12805"]
        # tokenize the tweet into words
        tokens = word_tokenize(doc)
        # convert to lowercase
        tokens = [w.lower() for w in tokens]
        # remove http words
        tokens = [w for w in tokens if "http" not in w]
        # remove punctuation
        tokens = [w for w in tokens if w.isalnum()]
        # print(tokens)
        preprocessed_text = ""
        for t in tokens:
            preprocessed_text += t + " "
        preprocessed_corpus.append(preprocessed_text)
        # Append Ground truth
        true_y.append(ground_truth[identifier])
    # Split train and test
    preprocessed_corpus = np.array(preprocessed_corpus)
    true_y = np.array(true_y)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    for train_index, test_index in splitter.split(preprocessed_corpus, true_y):
        X_train, X_test = preprocessed_corpus[train_index], preprocessed_corpus[test_index]
        Y_train, Y_test = true_y[train_index], true_y[test_index]
    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train.toarray(), X_test.toarray(), Y_train, Y_test


def train_svc(path_to_corpus, path_to_gt, path_to_save_model, name):
    corpus, ground_truth = parse_corpus_and_gt(path_to_corpus, path_to_gt)
    X_train, X_test, Y_train, Y_test = preprocess(corpus, ground_truth)


    # Fit classifier
    print("Starting the training")
    clf = LinearSVC(verbose=1, random_state=7, tol=1e-5, class_weight="balanced", C=1)
    clf.fit(X_train, Y_train)

    if path_to_save_model:
        print("saving model and data")
        # save model with dump. Load it with joblib.load
        dump(clf, join(path_to_save_model, "svc_uc3m{}.joblib".format(name)))
        np.savetxt(join(path_to_save_model, "train_data{}.csv".format(name)), X_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "train_truth{}.csv".format(name)), Y_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "eval_data{}.csv".format(name)), X_test, delimiter=",")
        np.savetxt(join(path_to_save_model, "eval_truth{}.csv".format(name)), Y_test, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to es.xml/ca.xml")
    parser.add_argument("--truth", help="Path to ground truth of the corresponding corpus")
    parser.add_argument("--save", help="Path to save model")
    parser.add_argument("--name", help="Name to append to model")

    args = parser.parse_args()
    train_svc(args.path, args.truth, args.save, args.name)
