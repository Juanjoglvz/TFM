import numpy as np
import matplotlib.pyplot as plt



def barplot(data, labels, path_to_save):
    heights = []

    for i in labels:
        heights.append(0)

    for i in range(len(labels)):
        for x in data:
            if x == i:
                heights[i] += 1

    plt.figure()
    plt.bar([i for i in range(len(labels))], heights, tick_label=labels)
    plt.savefig(path_to_save, format="pdf")

