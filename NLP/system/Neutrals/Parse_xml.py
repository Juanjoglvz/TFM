import xml.etree.ElementTree as ET
import argparse
from os.path import join
from nltk.stem import SnowballStemmer


def parse_corpus_and_gt(file, truth):
    with_truth = True if truth else False
    corpus = {}

    tree = ET.parse(file)
    root = tree.getroot()

    for doc in root.iter("document"):
        id = doc.attrib["id"]
        text = doc.find("text")

        # Include previous and next tweet as part of it.
        # Copy tweet twice for more weight
        next = doc.find("next")
        prev = doc.find("prev")

        final_text = ""
        if prev.text is not None:
            final_text += prev.text + " "
        final_text += text.text + " "
        final_text += text.text + " "
        if next.text is not None:
            final_text += next.text

        corpus[id] = final_text

    if with_truth:
        ground_truth = {}
        with open(truth, "r") as f:
            n = 0
            for line in f.readlines():
                line = line.split(":::")
                # Favor and against same class as we are interested in N/not N
                if "FAVOR" in line[1]:
                    ground_truth[line[0]] = 0
                elif "NEUTRAL" in line[1]:
                    ground_truth[line[0]] = 1
                elif "AGAINST" in line[1]:
                    ground_truth[line[0]] = 0
        return corpus, ground_truth
    return corpus, None


def parse_ml_senticon(path):
    print("Loading ML_senticon lexicon")
    ret = {}

    tree = ET.parse(join(path, "senticon.es.xml"))
    root = tree.getroot()
    #stemmer = SnowballStemmer('spanish')
    for layer in root.iter("layer"):
        for lemma in layer.iter("lemma"):
            word = lemma.text
            if "_" not in word:
                word = word.lstrip().rstrip()
                #word = stemmer.stem(word)
                polarity = lemma.attrib["pol"]
                ret[word] = polarity
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to es.xml")
    parser.add_argument("--truth", help="Path to ground truth")

    args = parser.parse_args()
    corpus, truth = parse_corpus_and_gt(args.path, args.truth)
    print(corpus)
    print(truth)
