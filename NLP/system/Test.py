import argparse
import numpy as np
import spacy
from joblib import load
from NLP.system.TweetMotifTokenizer import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score
from NLP.system.Parse_xml import parse_corpus_and_gt, parse_ml_senticon
from NLP.system.Neutrals.Preprocess import load_ml_senticon, load_kaggle, \
    is_polarized_kaggle , get_factor_ml_senticon, is_exception, load_vocabularies


def load_vocabulary(path):
    voc = {}
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.split(", ")
            voc[line[0]] = int(line[1])
    return voc


def preprocess(corpus, ground_truth, path_to_vocabulary, senticon):
    preprocessed_corpus = []
    true_y = []
    hashtags_total = []
    mentions_total = []
    sents_total = []
    domains = []
    n_hashtags_total = []
    n_mentions_total = []
    n_positive_words_total = []
    n_negative_words_total = []

    lemmatizer = spacy.load("es_core_news_sm")


    for identifier, gt in ground_truth.items():
        # for identifier in ["e079379e7b64ca8b52e58d87bebd36f9", "6f510b6acc3fab195959d88db9ee34a5", "ad88732860e9e8f2f7533a9b331d9eb9", "80f85f7fd8e3858b774b9cafbb701ce1", "6696259fb7704bc2072b696adc10ea5f", "91d9d568b6f08af5264fbe52bc849f88"]:
        doc = corpus[identifier]
        hashtags = []
        mentions = []
        sents = []
        n_hashtags = 0
        n_mentions = 0
        n_positive_words = 0
        n_negative_words = 0
        # tokenize the tweet into words
        tokens = tokenize(doc)
        # convert to lowercase
        tokens = [w.lower() for w in tokens]

        # Handle each token
        new_tokens = []
        for token in tokens:
            if "http" in token:
                new_tokens.append("url")
            elif "#" in token:
                n_hashtags += 1
                hashtags.append(token)
            elif "@" in token:
                n_mentions += 1
                mentions.append(token)
            # Normal word (and punctuations)
            else:
                if is_polarized_kaggle(token) == 1:
                    n_positive_words += 1
                elif is_polarized_kaggle(token) == -1:
                    n_negative_words += 1
                new_tokens.append(token)

        tokens = new_tokens

        # remove punctuation
        tokens = [w for w in tokens if w.isalnum() or is_exception(w)]

        # Stemming
        # stemmer = SnowballStemmer('spanish')
        # tokens = [stemmer.stem(w) for w in tokens]

        # Lemmatizing
        new_tokens = []
        preprocessed_text = ""
        for t in tokens:
            preprocessed_text += t + " "
        for l in lemmatizer(preprocessed_text):
            if l.lemma_ in senticon.keys():
                sents.append(l.lemma_)
            new_tokens.append(l.lemma_)
        tokens = new_tokens
        # convert to text
        preprocessed_text = ""
        for t in tokens:
            preprocessed_text += t + " "
        preprocessed_corpus.append(preprocessed_text)

        preprocessed_hashtags_text = ""
        for t in hashtags:
            preprocessed_hashtags_text += t + " "
        hashtags_total.append(preprocessed_hashtags_text)

        preprocessed_mentions_text = ""
        for t in mentions:
            preprocessed_mentions_text += t + " "
        mentions_total.append(preprocessed_mentions_text)

        preprocessed_sents_text = ""
        for t in sents:
            preprocessed_sents_text += t + " "
        sents_total.append(preprocessed_sents_text)
        # Append Ground truth
        true_y.append(gt)
        # Append n_hashtags and n_mentions
        n_hashtags_total.append(n_hashtags)
        n_mentions_total.append(n_mentions)
        n_positive_words_total.append(n_positive_words)
        n_negative_words_total.append(n_negative_words)

    print("Iteration finished")

    X = preprocessed_corpus
    y = true_y

    X_hashtags = hashtags_total
    X_mentions = mentions_total
    X_sents = sents_total


    # Convert to numpy array
    preprocessed_corpus = np.array(preprocessed_corpus)
    hashtags_total = np.array(hashtags_total)
    mentions_total = np.array(mentions_total)
    sents_total = np.array(sents_total)
    true_y = np.array(true_y)
    n_mentions_total = np.array(n_mentions_total)
    n_hashtags_total = np.array(n_hashtags_total)
    n_negative_words_total = np.array(n_negative_words_total)
    n_positive_words_total = np.array(n_positive_words_total)

    # Get BOW, BOH, BOM
    if path_to_vocabulary:
        vocabulary0, vocabulary1, vocabulary2, vocabulary3 = load_vocabularies(path_to_vocabulary)
    else:
        vocabulary0 = None
        vocabulary1 = None
        vocabulary2 = None
        vocabulary3 = None
    total_vocabulary = []
    vectorizer = TfidfVectorizer(vocabulary=vocabulary0)
    X = vectorizer.fit_transform(X).toarray()

    vectorizer = TfidfVectorizer(vocabulary=vocabulary1)
    X_hashtags = vectorizer.fit_transform(X_hashtags).toarray()

    vectorizer = TfidfVectorizer(vocabulary=vocabulary2)
    X_mentions = vectorizer.fit_transform(X_mentions).toarray()

    vectorizer2 = CountVectorizer(vocabulary=vocabulary3)
    X_sents = vectorizer2.fit_transform(X_sents).toarray()

    # Add sentiment factor
    for word, column in vectorizer2.vocabulary_.items():
        factor = float(senticon[word])
        X_sents[:, column] = X_sents[:, column] * factor

    # Merge features
    X = np.concatenate((X, X_hashtags, X_mentions, X_sents), axis=1)


    # Add extra features
    X = np.c_[X, n_hashtags_total, n_mentions_total,
                    n_positive_words_total, n_negative_words_total]

    return X, y


def test_svc(path_to_corpus, path_to_gt, path_to_model, path_to_sentiments, path_to_neutrals, path_to_vocabulary):
    corpus_es, ground_truth_es, ground_truth_es_neutrals, total_ground_truth_es = parse_corpus_and_gt(path_to_corpus,
                                                                                                      path_to_gt)
    load_kaggle(path_to_sentiments)
    ml_senticon = parse_ml_senticon(path_to_sentiments)
    senticon = load_ml_senticon(ml_senticon)
    X_test, Y_test = preprocess(corpus_es, total_ground_truth_es, path_to_vocabulary, senticon)

    clf_neutrals = load(path_to_neutrals)
    clf = load(path_to_model)

    neutrals_predict = clf_neutrals.predict(X_test)
    Y_pred = clf.predict(X_test)

    Y_pred[neutrals_predict == 1] = 1

    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1, 2])
    print("\t\t\t\tFavor\tNeutral\tAgainst")
    print("Favor\t\t\t{}\t{}\t{}".format(cm[0, 0], cm[0, 1], cm[0, 2]))
    print("Neutral\t\t\t\t{}\t{}\t{}".format(cm[1, 0], cm[1, 1], cm[1, 2]))
    print("Against\t\t\t\t{}\t{}\t{}".format(cm[2, 0], cm[2, 1], cm[2, 2]))

    prec = precision_score(Y_test, Y_pred, labels=[0,1,2], average=None)
    for i in range(len(prec)):
        print("Precision for class {}: {}".format(i, prec[i]))
    fscore = f1_score(Y_test, Y_pred, labels=[0, 1, 2], average="macro")
    print("F_score: {}".format(fscore))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Path to test data.")
    parser.add_argument("--truth", help="Path to ground truth.")
    parser.add_argument("--model", help="Path to model")
    parser.add_argument("--senti", help="Path to sentiment datasets")
    parser.add_argument("--neutrals", help="Path to neutrals model")
    parser.add_argument("--vocabulary", help="Path to vocabulary for neutrals preprocessing")

    args = parser.parse_args()

    test_svc(args.corpus, args.truth, args.model, args.senti, args.neutrals, args.vocabulary)
