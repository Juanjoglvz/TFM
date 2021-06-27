from os.path import join as path_join
import operator
import heapq


def get_top_5_labels(labels: dict) -> list:
    return heapq.nlargest(5, labels.items(), key=operator.itemgetter(1))


def main():
    path = "./polar_explanations"
    class_1_effect_labels = {}
    class_0_effect_labels = {}
    effects = {}
    with open(path_join(path, "explanations.txt"), "r") as f:
        for line in f.readlines():
            i, model_prediction, true_class, labels = line.split(";")

            labels = labels.strip()[1:-1].split(")")
            for label in labels[:-1]:
                label = label[1:]
                feature, effect = label.split(",")
                effects[feature] = effect

                if float(effect) > 0:
                    if feature not in class_1_effect_labels.keys():
                        class_1_effect_labels[feature] = 1
                    else:
                        class_1_effect_labels[feature] += 1
                else:
                    if feature not in class_0_effect_labels.keys():
                        class_0_effect_labels[feature] = 1
                    else:
                        class_0_effect_labels[feature] += 1

    class_1_top_5 = get_top_5_labels(class_1_effect_labels)
    class_0_top_5 = get_top_5_labels(class_0_effect_labels)
    print(class_1_top_5)
    print(class_0_top_5)

    for label in class_1_top_5:
        print(f"Label: {label[0]} with effect {effects[label[0]]} appearead {label[1]} times.")


if __name__ == "__main__":
    main()
