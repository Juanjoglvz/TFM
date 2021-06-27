import lime
import spacy
import numpy as np
import random

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import load
from lime.lime_text import LimeTextExplainer

from NLP.system.Neutrals.Preprocess import load_kaggle, load_ml_senticon, get_factor_ml_senticon, \
    is_polarized_kaggle, is_exception
from NLP.system.Parse_xml import parse_corpus_and_gt, parse_ml_senticon
from NLP.system.TweetMotifTokenizer import tokenize

from os.path import join


class SystemTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabularies, idfs, senticon):
        self.bow_vectorizer = TfidfVectorizer(vocabulary=vocabularies[0])
        self.bow_vectorizer.idf_ = idfs[0]
        self.boh_vectorizer = TfidfVectorizer(vocabulary=vocabularies[1])
        self.boh_vectorizer.idf_ = idfs[1]
        self.bom_vectorizer = TfidfVectorizer(vocabulary=vocabularies[2])
        self.bom_vectorizer.idf_ = idfs[2]
        self.sents_vectorizer = CountVectorizer(vocabulary=vocabularies[3])
        self.senticon = senticon
        self.lemmatizer = spacy.load("es_core_news_sm")

    def fit(self):
        return self

    def transform(self, x, y=None):
        preprocessed_corpus = []
        hashtags_total = []
        mentions_total = []
        sents_total = []
        n_hashtags_total = []
        n_mentions_total = []
        n_positive_words_total = []
        n_negative_words_total = []

        for doc in x:
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

            # Lemmatizing
            new_tokens = []
            preprocessed_text = ""
            for t in tokens:
                preprocessed_text += t + " "
            for l in self.lemmatizer(preprocessed_text):
                if l.lemma_ in self.senticon.keys():
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
            # Append n_hashtags and n_mentions
            n_hashtags_total.append(n_hashtags)
            n_mentions_total.append(n_mentions)
            n_positive_words_total.append(n_positive_words)
            n_negative_words_total.append(n_negative_words)

        # Convert to numpy array
        preprocessed_corpus = np.array(preprocessed_corpus)
        hashtags_total = np.array(hashtags_total)
        mentions_total = np.array(mentions_total)
        sents_total = np.array(sents_total)
        n_mentions_total = np.array(n_mentions_total)
        n_hashtags_total = np.array(n_hashtags_total)
        n_negative_words_total = np.array(n_negative_words_total)
        n_positive_words_total = np.array(n_positive_words_total)

        # Apply vectorizers
        bow = self.bow_vectorizer.transform(preprocessed_corpus).toarray()
        boh = self.boh_vectorizer.transform(hashtags_total).toarray()
        bom = self.bom_vectorizer.transform(mentions_total).toarray()
        sents = self.sents_vectorizer.transform(sents_total).toarray()

        # Add sentiment factor
        for word, column in self.sents_vectorizer.vocabulary_.items():
            factor = float(self.senticon[word])
            sents[:, column] = sents[:, column] * factor

        # Merge features
        X = np.concatenate((bow, boh, bom, sents), axis=1)
        X = np.c_[X, n_hashtags_total, n_mentions_total,
                        n_positive_words_total, n_negative_words_total]

        return X


def main():
    path_to_sentiments = "F:/Sentimientos"
    path_to_corpus_es = "F:/MultiStanceCat-IberEval-training-20180404/es.xml"
    path_to_gt_es = "F:/MultiStanceCat-IberEval-training-20180404/truth-es.txt"

    # Load all data and auxiliary corpus
    load_kaggle(path_to_sentiments)
    ml_senticon = parse_ml_senticon(path_to_sentiments)
    ml_senticon = load_ml_senticon(ml_senticon)

    # Parse data
    corpus_es, ground_truth_es, ground_truth_es_neutrals, total_ground_truth_es = parse_corpus_and_gt(path_to_corpus_es,
                                                                                                      path_to_gt_es)

    # Load weights for preprocessing
    vocabularies = []
    for i in range(4):
        voc = {}
        with open(join("F:/MultiStanceCat-IberEval-training-20180404/system/neutrals", "vocabulary{}.csv".format(i)), "r") as f:
            for line in f.readlines():
                line = line.split(", ")
                voc[line[0]] = int(line[1])
            vocabularies.append(voc)

    idfs = []
    for i in range(3):
        voc = []
        with open(join("F:/MultiStanceCat-IberEval-training-20180404/system/neutrals", "idf{}.csv".format(i)), "r") as f:
            for line in f.readlines():
                voc.append(float(line))
            idfs.append(np.array(voc))

    # Load model
    model = load("F:/MultiStanceCat-IberEval-training-20180404/system/neutrals/clf_neutrals.joblib")

    # Create pipeline
    pipeline = Pipeline(steps=[
        ("Preprocessing", SystemTransformer(vocabularies, idfs, ml_senticon)),
        ("Model", model)
    ])

    # Prepare data to pass to pipeline
    class_names = ["Not Neutral", "Neutral"]
    explainer = LimeTextExplainer(class_names=class_names)

    # idx = 5
    data = list(corpus_es.values())
    target = list(ground_truth_es_neutrals.values())
    for i in range(8):
        idx = random.randint(0, 4437)
        exp = explainer.explain_instance(data[idx], pipeline.predict_proba, num_features=6)
        exp.save_to_file('./{}.html'.format(idx))

        print('Document id: %d' % idx)
        print('Probability(neutral) =', pipeline.predict_proba([data[idx]])[0, 1])
        print('True class: %s' % class_names[target[idx]])


if __name__ == "__main__":
        main()
