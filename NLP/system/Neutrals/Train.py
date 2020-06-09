import argparse
from os.path import join
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from NLP.system.Neutrals.TweetMotifTokenizer import tokenize
from NLP.system.Neutrals.Parse_xml import parse_corpus_and_gt
from NLP.system.Graphics import barplot


def is_exception(ch):
    if ch[0] == "#" or ch[0] == "@":
        return True
    else:
        return False


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
        tokens = [w for w in tokens if w.isalnum() or is_exception(w)]
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
    return X_train.toarray(), X_test.toarray(), Y_train, Y_test, true_y


def train_svc(path_to_corpus_es, path_to_gt_es, path_to_save_model):
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

    args = parser.parse_args()
    train_svc(args.path_es, args.truth_es, args.save)
