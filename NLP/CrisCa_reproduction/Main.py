from CrisCa_reproduction.Parse_xml import parse_corpus_and_gt
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import dump


def train_svc(path_to_corpus, path_to_gt, path_to_save_model):
    corpus, ground_truth = parse_corpus_and_gt(path_to_corpus, path_to_gt)

    stemmed_corpus = []
    true_y = []

    for id, doc in corpus.items():
    #doc = corpus["af9f7be8eddca053f705decaf6b12805"]
        # tokenize the tweet into words
        tokens = word_tokenize(doc)
        # convert to lowercase
        tokens = [w.lower() for w in tokens]
        # remove http words
        tokens = [w for w in tokens if "http" not in w]
        # remove punctuation
        tokens = [w for w in tokens if w.isalnum()]
        #print(tokens)
        # Stemming with Snowball stemmer
        stemmer = SnowballStemmer('spanish')
        stemmed_tokens = [stemmer.stem(w) for w in tokens]
        stemmed_text = ""
        for st in stemmed_tokens:
            stemmed_text += st + " "
        #print(stemmed_text)
        stemmed_corpus.append(stemmed_text)
        #Append Ground truth
        true_y.append(ground_truth[id])
    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(stemmed_corpus)
    X = X.toarray()
    # Fit classifier
    print("Starting the training")
    clf = LinearSVC(verbose=1, random_state=7, tol=1e-5)
    clf.fit(X, true_y)

    if path_to_save_model:
        # save model with dump. Load it with joblib.load
        dump(clf, path_to_save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to all.xml")
    parser.add_argument("--truth", help="Path to ground truth")
    parser.add_argument("--save", help="Path to save model")

    args = parser.parse_args()
    train_svc(args.path, args.truth, args.save)
