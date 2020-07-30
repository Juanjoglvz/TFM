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
    # Split train and test
    X_train, X_test, Y_train, Y_test = train_test_split(stemmed_corpus, true_y, test_size=0.2, random_state=7)
    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train.toarray(), X_test.toarray(), Y_train, Y_test, vectorizer.vocabulary_

def train_svc(path_to_corpus, path_to_gt, path_to_save_model):
    corpus, ground_truth = parse_corpus_and_gt(path_to_corpus, path_to_gt)
    X_train, X_test, Y_train, Y_test, vocabulary = preprocess(corpus, ground_truth)


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
        with open(join(path_to_save_model, "vocabulary.txt"), "w+") as f:
            for key in vocabulary.keys():
                f.write("{}, {}\n".format(key, vocabulary[key]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to all.xml")
    parser.add_argument("--truth", help="Path to ground truth")
    parser.add_argument("--save", help="Path to save model")

    args = parser.parse_args()
    train_svc(args.path, args.truth, args.save)
