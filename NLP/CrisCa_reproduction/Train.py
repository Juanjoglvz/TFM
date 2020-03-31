from NLP.CrisCa_reproduction.Parse_xml import parse_corpus_and_gt
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from joblib import dump
from os.path import join
from copy import deepcopy


def preprocess(corpus, ground_truth):
    stemmed_corpus = []
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
        # Stemming with Snowball stemmer
        stemmer = SnowballStemmer('spanish')
        stemmed_tokens = [stemmer.stem(w) for w in tokens]
        stemmed_text = ""
        for st in stemmed_tokens:
            stemmed_text += st + " "
        # print(stemmed_text)
        stemmed_corpus.append(stemmed_text)
        # Append Ground truth
        true_y.append(ground_truth[identifier])
    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(stemmed_corpus)
    return X.toarray(), true_y

def train_svc(path_to_corpus_es, path_to_corpus_ca, path_to_gt_es,
              path_to_gt_ca, path_to_save_model):
    corpus_es, ground_truth_es = parse_corpus_and_gt(path_to_corpus_es, path_to_gt_es)
    corpus_ca, ground_truth_ca = parse_corpus_and_gt(path_to_corpus_ca, path_to_gt_ca)
    corpus = deepcopy(corpus_es)
    corpus.update(corpus_ca)
    ground_truth = deepcopy(ground_truth_es)
    ground_truth.update(ground_truth_ca)
    X, true_y = preprocess(corpus, ground_truth)

    X_train, X_test, Y_train, Y_test = train_test_split(X, true_y, test_size=0.2, random_state=7)
    # Fit classifier
    print("Starting the training")
    clf = LinearSVC(verbose=1, random_state=7, tol=1e-5)
    clf.fit(X_train, Y_train)

    if path_to_save_model:
        print("saving model and data")
        # save model with dump. Load it with joblib.load
        dump(clf, join(path_to_save_model, "svc_crisca.joblib"))
        np.savetxt(join(path_to_save_model, "train_data.csv"), X_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "train_truth.csv"), Y_train, delimiter=",")
        np.savetxt(join(path_to_save_model, "eval_data.csv"), X_test, delimiter=",")
        np.savetxt(join(path_to_save_model, "eval_truth.csv"), Y_test, delimiter=",")

        # Save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_es", help="Path to es.xml")
    parser.add_argument("path_ca", help="Path to ca.xml")
    parser.add_argument("--truth_es", help="Path to spanish ground truth")
    parser.add_argument("--truth_ca", help="Path to catalan ground truth")
    parser.add_argument("--save", help="Path to save model")

    args = parser.parse_args()
    train_svc(args.path_es, args.path_ca, args.truth_es, args.truth_ca, args.save)
