import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from NLP.system.Neutrals.TweetMotifTokenizer import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import join
from nltk.stem import SnowballStemmer
import spacy


# Global variables
positive_words_es = []
negative_words_es = []
senticon = {}



# Helper functions

def is_exception(ch):
    if ch[0] == "#" or ch[0] == "@":
        return True
    else:
        return False


def load_kaggle(path):
    # Spanish words
    print("Loading kaggle positive/negative words")
    with open(join(path, "positive_words_es.txt"), "r") as f:
        for line in f.readlines():
            positive_words_es.append(line.rstrip())
    with open(join(path, "negative_words_es.txt"), "r") as f:
        for line in f.readlines():
            negative_words_es.append(line.rstrip())
    # Catalan words


def load_ml_senticon(ml_senticon):
    global senticon
    senticon = ml_senticon


def is_polarized_kaggle(word):
    global positive_words_es
    global negative_words_es
    if word in positive_words_es:
        return 1
    elif word in negative_words_es:
        return -1
    else:
        return 0

def get_factor_ml_senticon(word):
    global senticon
    return senticon[word]


# Preprocess
def preprocess(corpus, ground_truth):
    print("Starting the preprocessing")
    preprocessed_corpus = []
    true_y = []
    hashtags_total = []
    mentions_total = []
    sents_total = []
    n_hashtags_total = []
    n_mentions_total = []
    n_positive_words_total = []
    n_negative_words_total = []

    lemmatizer = spacy.load("es_core_news_sm")

    for identifier, doc in corpus.items():
    #for identifier in ["e079379e7b64ca8b52e58d87bebd36f9", "6f510b6acc3fab195959d88db9ee34a5", "ad88732860e9e8f2f7533a9b331d9eb9", "80f85f7fd8e3858b774b9cafbb701ce1", "6696259fb7704bc2072b696adc10ea5f", "91d9d568b6f08af5264fbe52bc849f88"]:
        #doc = corpus[identifier]
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
        #stemmer = SnowballStemmer('spanish')
        #tokens = [stemmer.stem(w) for w in tokens]

        # Lemmatizing
        new_tokens = []
        for t in tokens:
            for x in lemmatizer(t):
                new_tokens.append(x.lemma_)
        tokens = new_tokens

        for t in tokens:
            if t in senticon.keys():
                sents.append(t)

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
        true_y.append(ground_truth[identifier])
        # Append n_hashtags and n_mentions
        n_hashtags_total.append(n_hashtags)
        n_mentions_total.append(n_mentions)
        n_positive_words_total.append(n_positive_words)
        n_negative_words_total.append(n_negative_words)

    print("Iteration finished")

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

    # Split train and test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    for train_index, test_index in splitter.split(preprocessed_corpus, true_y):
        # Data
        X_train, X_test = preprocessed_corpus[train_index], preprocessed_corpus[test_index]
        X_hashtags_train, X_hashtags_test = hashtags_total[train_index], hashtags_total[test_index]
        X_mentions_train, X_mentions_test = mentions_total[train_index], mentions_total[test_index]
        X_sents_train, X_sents_test = sents_total[train_index], sents_total[test_index]

        # Ground truth
        Y_train, Y_test = true_y[train_index], true_y[test_index]

        # Extra features
        n_mentions_total_train, n_mentions_total_test = n_mentions_total[train_index], n_mentions_total[test_index]
        n_hashtags_total_train, n_hashtags_total_test = n_hashtags_total[train_index], n_hashtags_total[test_index]
        n_positive_words_total_train, n_positive_words_total_test = \
            n_positive_words_total[train_index], n_positive_words_total[test_index]
        n_negative_words_total_train, n_negative_words_total_test = \
            n_negative_words_total[train_index], n_negative_words_total[test_index]

    # Get BOW, BOH, BOM
    total_vocabulary = []
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    total_vocabulary.append(vectorizer.vocabulary_)

    X_hashtags_train = vectorizer.fit_transform(X_hashtags_train).toarray()
    X_hashtags_test = vectorizer.transform(X_hashtags_test).toarray()
    total_vocabulary.append(vectorizer.vocabulary_)

    X_mentions_train = vectorizer.fit_transform(X_mentions_train).toarray()
    X_mentions_test = vectorizer.transform(X_mentions_test).toarray()
    total_vocabulary.append(vectorizer.vocabulary_)

    X_sents_train = vectorizer.fit_transform(X_sents_train).toarray()
    X_sents_test = vectorizer.transform(X_sents_test).toarray()
    total_vocabulary.append(vectorizer.vocabulary_)

    # Add sentiment factor
    for word, column in vectorizer.vocabulary_.items():
        factor = float(senticon[word])
        X_sents_train[:, column] = X_sents_train[:, column] * factor
        X_sents_test[:, column] = X_sents_test[:, column] * factor

    # Merge features
    X_train = np.concatenate((X_train, X_hashtags_train, X_mentions_train, X_sents_train), axis=1)

    X_test = np.concatenate((X_test, X_hashtags_test, X_mentions_test, X_sents_test), axis=1)

    # Add extra features
    X_train = np.c_[X_train, n_hashtags_total_train, n_mentions_total_train,
                    n_positive_words_total_train, n_negative_words_total_train]
    X_test = np.c_[X_test, n_hashtags_total_test, n_mentions_total_test,
                   n_positive_words_total_test, n_negative_words_total_test]

    return X_train, X_test, Y_train, Y_test, true_y, total_vocabulary
