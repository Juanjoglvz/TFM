import argparse
import numpy as np
from joblib import load
from NLP.Elirf_reproduction.TweetMotifTokenizer import tokenize
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from NLP.Elirf_reproduction.Parse_xml import parse_corpus_and_gt


def load_vocabulary(path):
    voc = {}
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.split(", ")
            voc[line[0]] = int(line[1])
    return voc


def preprocess(corpus, ground_truth, vocabulary):
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

    # Vectorize the text
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    X_test = vectorizer.fit_transform(preprocessed_corpus)
    return X_test.toarray(), true_y


def test_svc(path_to_corpus, path_to_gt, path_to_model, path_to_vocabulary):
    corpus, ground_truth = parse_corpus_and_gt(path_to_corpus, path_to_gt)
    vocabulary = load_vocabulary(path_to_vocabulary)
    X_test, Y_test = preprocess(corpus, ground_truth, vocabulary)


    clf = load(path_to_model)
    Y_pred = clf.predict(X_test)

    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1, 2])
    print("\t\t\t\tFavor\tNeutral\tAgainst")
    print("Favor\t\t\t{}\t{}\t{}".format(cm[0, 0], cm[0, 1], cm[0, 2]))
    print("Neutral\t\t\t\t{}\t{}\t{}".format(cm[1, 0], cm[1, 1], cm[1, 2]))
    print("Against\t\t\t\t{}\t{}\t{}".format(cm[2, 0], cm[2, 1], cm[2, 2]))

    prec = precision_score(Y_test, Y_pred, labels=[0,1,2], average=None, zero_division=0)
    for i in range(len(prec)):
        print("Precision for class {}: {}".format(i, prec[i]))
    fscore = f1_score(Y_test, Y_pred, labels=[0,1,2], average="macro")
    print("F_score: {}".format(fscore))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Path to test data.")
    parser.add_argument("--truth", help="Path to ground truth.")
    parser.add_argument("--vocabulary", help="Path to vectorizer vocabulary")
    parser.add_argument("--model", help="Path to model")

    args = parser.parse_args()

    test_svc(args.corpus, args.truth, args.model, args.vocabulary)
