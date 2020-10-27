import seaborn
import numpy as np

from typing import List
from collections import Counter
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

seaborn.set(rc={'figure.figsize': (15, 15)})


def noninteractive_visualisation(embeddings: np.ndarray, labels: List[int], *, verbose: bool = False) -> None:
    if verbose:
        print("Started dimensionality reduction for visualisation")
    tsne = TSNE(perplexity=20, random_state=42, verbose=False)
    embeddings = tsne.fit_transform(embeddings)

    palette = seaborn.hls_palette(len(set(labels)) - 1, s=.9)
    counter = Counter(labels)
    noise_x, noise_y, clusters_x, clusters_y, cluster_labels = [], [], [], [], []
    for i, label in enumerate(labels):
        if label == -1 or counter[label] == 1:
            noise_x.append(embeddings[i, 0])
            noise_y.append(embeddings[i, 1])
        else:
            clusters_x.append(embeddings[i, 0])
            clusters_y.append(embeddings[i, 1])
            cluster_labels.append(label)

    plt.scatter(noise_x, noise_y, c="gray", marker='^')
    seaborn.scatterplot(x=clusters_x, y=clusters_y, hue=cluster_labels, legend='full', palette=palette)
    plt.title("Covid-19 articles")
    plt.savefig("covid_visualisation.png")
    plt.show()
