from NLP.CrisCa_reproduction.Parse_xml import parse_corpus_and_gt
import argparse
import numpy as np
from NLP.Elirf_reproduction.TweetMotifTokenizer import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from joblib import dump
from os.path import join
from copy import deepcopy


def preprocess(corpus, ground_truth):
    preprocessed_corpus = []
    true_y = []

    for identifier, doc in corpus.items():
        # doc = corpus["af9f7be8eddca053f705decaf6b12805"]
        # tokenize the tweet into words
        tokens = tokenize(doc)
        # convert to lowercase
        tokens = [w.lower() for w in tokens]
        # Normalize Twitter-specific tokens
        new_tokens = []
        for token in tokens:
            if "http" in token:
                new_tokens.append("url")
            elif "#" in token:
                new_tokens.append("#hashtag")
            elif "@" in token:
                new_tokens.append("@mention")
            else:
                new_tokens.append(token)
        tokens = new_tokens
        # remove punctuation
        exceptions = ["#hashtag", "@mention"]
        tokens = [w for w in tokens if w.isalnum() or w in exceptions]
        # print(tokens)
        preprocessed_text = ""
        for t in tokens:
            preprocessed_text += t + " "
        preprocessed_corpus.append(preprocessed_text)
        # Append Ground truth
        true_y.append(ground_truth[identifier])

    # Split train and test
    X_train, X_test, Y_train, Y_test = train_test_split(preprocessed_corpus, true_y, test_size=0.2, random_state=7)
    # Vectorize the text
    vectorizer = CountVectorizer(ngram_range=(1,4))
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train.toarray(), X_test.toarray(), Y_train, Y_test


def train_svc(path_to_corpus_es, path_to_gt_es, path_to_save_model):
    corpus_es, ground_truth_es = parse_corpus_and_gt(path_to_corpus_es, path_to_gt_es)
    X_train, X_test, Y_train, Y_test = preprocess(corpus_es, ground_truth_es)

    # Fit classifier
    print("Starting the training")
    clf = LinearSVC(verbose=1, random_state=7, tol=1e-5, max_iter=100000)
    clf.fit(X_train, Y_train)

    if path_to_save_model:
        print("saving model and data")
        # save model with dump. Load it with joblib.load
        dump(clf, join(path_to_save_model, "svc_elirf.joblib"))
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

    args = parser.parse_args()
    train_svc(args.path_es, args.truth_es, args.save)
