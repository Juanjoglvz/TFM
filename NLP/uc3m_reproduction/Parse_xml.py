import xml.etree.ElementTree as ET
import argparse


def parse_corpus_and_gt(file, truth):
    with_truth = True if truth else False
    corpus = {}

    tree = ET.parse(file)
    root = tree.getroot()

    for doc in root.iter("document"):
        id = doc.attrib["id"]
        text = doc.find("text")

        # Include previous and next tweet as part of it
        next = doc.find("next")
        prev = doc.find("prev")

        final_text = ""
        if prev.text:
            final_text = prev.text + " "
        final_text += text.text + " "
        if next.text:
            final_text += next.text

        corpus[id] = final_text

    if with_truth:
        ground_truth = {}
        with open(truth, "r") as f:
            n = 0
            for line in f.readlines():
                line = line.split(":::")
                if "FAVOR" in line[1] or "favor" in line[1]:
                    ground_truth[line[0]] = 0
                elif "NEUTRAL" in line[1]:
                    ground_truth[line[0]] = 1
                elif "AGAINST" in line[1]:
                    ground_truth[line[0]] = 2
        return corpus, ground_truth
    return corpus, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to es.xml")
    parser.add_argument("--truth", help="Path to ground truth")

    args = parser.parse_args()
    corpus, truth = parse_corpus_and_gt(args.path, args.truth)
    print(corpus)
    print(truth)
