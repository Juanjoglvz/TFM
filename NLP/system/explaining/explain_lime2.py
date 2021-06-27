import pandas as pd
import numpy as np
from joblib import load
from os.path import join as path_join
import datetime

import lime
import lime.lime_tabular


def load_vocabularies(path: str) -> dict:
    """
    Loads vocabularies from pre-processing to reconstruct feature lables
    :param path: Path to dir where vocabularies are
    :return: returns a vocabulary with mappings label:column_index
    """
    total_vocabulary = {}
    for i in range(4):
        offset = len(total_vocabulary)
        with open(path_join(path, f"vocabulary{i}.csv"), "r") as f:
            for line in f.readlines():
                word, n = line.split(", ")
                if i == 1:
                    word = "#" + word
                elif i == 2:
                    word = "@" + word
                elif i == 3:
                    word = "photo_" + word
                total_vocabulary[word] = int(n) + offset
    return total_vocabulary


def transform_vocabulary_to_list(vocabulary: dict) -> list:
    result = [""] * len(vocabulary)
    for key in vocabulary.keys():
        result[vocabulary[key]] = key
    return result


def load_dataset(path: str, name: str, labels: list) -> tuple:
    """
    Loads a part of the dataset (train or eval) by reconstructing the data and the ground truth
    :param path: Path to data directory
    :param name: Name of dataset part (train or eval)
    :param labels: Feature names to construct the DataFrame
    :return: returns a pandas dataframe with the data
    """
    labels = labels + ["sent_final_factors", "sent_factors_library",
                       "n_hashtags_total", "n_mentions_total",
                       "n_positive_words_total", "n_negative_words_total", "language_feature"]
    data = pd.read_csv(path_join(path, f"{name}_data.csv"), names=labels)
    truth = pd.read_csv(path_join(path, f"{name}_truth.csv"), names=["truth"])

    return data, truth


def main():
    print(datetime.datetime.now())
    # labels = transform_vocabulary_to_list(
    #     load_vocabularies("F:\\MultiStanceCat-IberEval-training-20180404\\system_with_cat\\polar"))
    #
    # x_train, y_train = load_dataset("F:\\MultiStanceCat-IberEval-training-20180404\\system_with_cat\\polar", "train",
    #                                 labels)
    # x_test, y_test = load_dataset("F:\\MultiStanceCat-IberEval-training-20180404\\system_with_cat\\polar", "eval",
    #                               labels)
    # x_train.to_csv("x_train_polar.csv", index=False)
    # x_test.to_csv("x_test_polar.csv", index=False)
    # y_train.to_csv("y_train_polar.csv", index=False)
    # y_test.to_csv("y_test_polar.csv", index=False)

    x_train = pd.read_csv("x_train_polar.csv")
    x_test = pd.read_csv("x_test_polar.csv")
    y_train = pd.read_csv("y_train_polar.csv")
    y_test = pd.read_csv("y_test_polar.csv")

    x_train = x_train[y_train["truth"] != 1]
    y_train = y_train[y_train["truth"] != 1]
    x_test = x_test[y_test["truth"] != 1]
    y_test = y_test[y_test["truth"] != 1]

    print("Loaded data. Let's explain!!")

    clf_polar = load("F:\\MultiStanceCat-IberEval-training-20180404\\system_with_cat\\polar\\clf.joblib")

    explainer = lime.lime_tabular.LimeTabularExplainer(x_train.to_numpy(), feature_names=x_train.columns,
                                                       class_names=np.array(["FAVOUR", "AGAINST"]),
                                                       discretize_continuous=True)

    explanations = []
    for n in range(153):
        print(n)
        current_explanation = {}
        i = np.random.randint(0, len(x_test))
        exp = explainer.explain_instance(x_test.iloc[i], clf_polar.predict_proba, num_features=5)
        current_explanation["i"] = i
        current_explanation['labels'] = exp.as_list()
        current_explanation["model_prediction"] = clf_polar.predict_proba([x_test.iloc[i]])[0, 1]
        current_explanation["true_class"] = y_test.iloc[i][0]
        # exp.save_to_file('./neutrals_explanations/{}.html'.format(i))
        explanations.append(current_explanation)

    with open("polar_explanations/explanations.txt", "w+") as f:
        for exp in explanations:
            f.write(f"{exp['i']};{exp['model_prediction']};{exp['true_class']};[")
            for label in exp['labels']:
                f.write(f"({label[0]},{label[1]})")
            f.write("]\n")
    print(datetime.datetime.now())


if __name__ == "__main__":
    main()
