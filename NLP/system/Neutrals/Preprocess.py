import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from NLP.system.Neutrals.TweetMotifTokenizer import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import join

# Global variables
positive_words_es = []
negative_words_es = []


# Helper functions

def is_exception(ch):
    if ch[0] == "#" or ch[0] == "@":
        return True
    else:
        return False


def load_kaggle(path):
    print("Loading kaggle positive/negative words")
    with open(join(path, "positive_words_es.txt"), "r") as f:
        for line in f.readlines():
            positive_words_es.append(line.rstrip())
    with open(join(path, "negative_words_es.txt"), "r") as f:
        for line in f.readlines():
            negative_words_es.append(line.rstrip())


def is_polarized_kaggle(word):
    global positive_words_es
    global negative_words_es
    if word in positive_words_es:
        return 1
    elif word in negative_words_es:
        return -1
    else:
        return 0



# Preprocess
def preprocess(corpus, ground_truth):
    preprocessed_corpus = []
    true_y = []
    n_hashtags_total = []
    n_mentions_total = []
    n_positive_words_total = []
    n_negative_words_total = []

    for identifier, doc in corpus.items():
        n_hashtags = 0
        n_mentions = 0
        n_positive_words = 0
        n_negative_words = 0
        # doc = corpus["af9f7be8eddca053f705decaf6b12805"]
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
            elif "@" in token:
                n_mentions += 1
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
        # convert to text
        preprocessed_text = ""
        for t in tokens:
            preprocessed_text += t + " "
        preprocessed_corpus.append(preprocessed_text)

        # Append Ground truth
        true_y.append(ground_truth[identifier])
        # Append n_hashtags and n_mentions
        n_hashtags_total.append(n_hashtags)
        n_mentions_total.append(n_mentions)
        n_positive_words_total.append(n_positive_words)
        n_negative_words_total.append(n_negative_words)


    # Convert to numpy array
    preprocessed_corpus = np.array(preprocessed_corpus)
    true_y = np.array(true_y)
    n_mentions_total = np.array(n_mentions_total)
    n_hashtags_total = np.array(n_hashtags_total)
    n_negative_words_total = np.array(n_negative_words_total)
    n_positive_words_total = np.array(n_positive_words_total)

    # Split train and test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    for train_index, test_index in splitter.split(preprocessed_corpus, true_y):
        X_train, X_test = preprocessed_corpus[train_index], preprocessed_corpus[test_index]
        Y_train, Y_test = true_y[train_index], true_y[test_index]
        n_mentions_total_train, n_mentions_total_test = n_mentions_total[train_index], n_mentions_total[test_index]
        n_hashtags_total_train, n_hashtags_total_test = n_hashtags_total[train_index], n_hashtags_total[test_index]
        n_positive_words_total_train, n_positive_words_total_test = \
            n_positive_words_total[train_index], n_positive_words_total[test_index]
        n_negative_words_total_train, n_negative_words_total_test = \
            n_negative_words_total[train_index], n_negative_words_total[test_index]

    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    # Add extra features
    X_train = np.c_[X_train, n_hashtags_total_train, n_mentions_total_train,
                    n_positive_words_total_train, n_negative_words_total_train]
    X_test = np.c_[X_test, n_hashtags_total_test, n_mentions_total_test,
                   n_positive_words_total_test, n_negative_words_total_test]

    return X_train, X_test, Y_train, Y_test, true_y
