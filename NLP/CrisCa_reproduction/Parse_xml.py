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

        # Include previous and next tweet as part of it.
        # Copy tweet twice for more weight
        next = doc.find("next")
        prev = doc.find("prev")

        final_text = ""
        if prev.text is not None:
            pass
            #final_text += prev.text + " "
        final_text += text.text + " "
        #final_text += text.text + " "
        if next.text is not None:
            #final_text += next.text
            pass

        corpus[id] = final_text

    if with_truth:
        ground_truth = {}
        with open(truth, "r") as f:
            n = 0
            for line in f.readlines():
                line = line.split(":::")
                if "FAVOR" in line[1]:
                    ground_truth[line[0]] = 0
                elif "NEUTRAL" in line[1]:
                    ground_truth[line[0]] = 1
                elif "AGAINST" in line[1]:
                    ground_truth[line[0]] = 2
        return corpus, ground_truth
    return corpus, None
# Interesante: la suma de los dos corpus es 9121, pero al juntarlos y hacer diccionarios queda 9097
# Parece que hay repes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to all.xml")
    parser.add_argument("--truth", help="Path to ground truth")

    args = parser.parse_args()
    print(parse_corpus_and_gt(args.path, args.truth))
