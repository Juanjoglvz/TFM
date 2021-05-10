import os
import xml.etree.ElementTree as ET

photos_dic = {}
gt_dic = {}
positive_images = []
neutral_images = []
negative_images = []


def main():
    tree = ET.parse("F:/MultiStanceCat-IberEval-training-20180404/es.xml")
    root = tree.getroot()
    for doc in root.iter("document"):
        id = doc.attrib["id"]
        for photo in doc.find("photos"):
            photo_text = photo.text.split(".")[0]
            if photo_text in photos_dic.keys():
                photos_dic[photo_text].append(id)
            else:
                photos_dic[photo_text] = [id]

    with open("F:/MultiStanceCat-IberEval-training-20180404/truth-es.txt", "r") as f:
        for line in f.readlines():
            line = line.split(":::")
            # Favor and against same class as we are interested in N/not N
            if "FAVOR" in line[1]:
                gt_dic[line[0]] = 0
            elif "NEUTRAL" in line[1]:
                gt_dic[line[0]] = 1
            elif "AGAINST" in line[1]:
                gt_dic[line[0]] = 2

    for photo, doc in photos_dic.items():
        for d in doc:
            sent = gt_dic[d]
            if sent == 0:
                positive_images.append(photo)
            if sent == 1:
                neutral_images.append(photo)
            if sent == 2:
                negative_images.append(photo)

    print(len(photos_dic), len(positive_images), len(neutral_images), len(negative_images))


if __name__ == "__main__":
    main()
