import hdbscan
import numpy as np

from math import sqrt

from kneed import KneeLocator
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


def get_cluster_labels(embeddings: np.ndarray, method: str) -> np.ndarray:
    try:
        methods = {
            'dbscan': _get_cluster_labels_dbscan,
            'hdbscan': _get_cluster_labels_hdbscan,
            'kmeans': _get_cluster_labels_kmeans
        }
        function = methods[method.lower()]
    except KeyError as key:
        raise KeyError(f"Invalid clustering method name: {key}! Available methods are: {tuple(methods.keys())}")
    return function(embeddings)


def find_epsilon(embeddings: np.ndarray) -> float:
    k = round(sqrt(len(embeddings)))
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1, algorithm='auto').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    distances = [np.mean(d) for d in np.sort(distances, axis=0)]
    kneedle = KneeLocator(distances, list(range(len(distances))), online=True)
    epsilon = np.min(list(kneedle.all_elbows))
    if epsilon == 0.0:
        epsilon = np.mean(distances)
    return float(epsilon)


def _get_cluster_labels_hdbscan(embeddings: np.ndarray) -> np.ndarray:
    distance_matrix = pairwise_distances(embeddings, n_jobs=-1, metric='cosine')
    distance_matrix = np.array(distance_matrix, dtype='double')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2,
                                min_samples=2,
                                metric='precomputed')
    clusterer.fit(distance_matrix)
    return clusterer.labels_


def _get_cluster_labels_dbscan(embeddings: np.ndarray) -> np.ndarray:
    epsilon = find_epsilon(embeddings)
    cluster_labels = DBSCAN(eps=epsilon,
                            min_samples=1,
                            n_jobs=1) \
        .fit_predict(embeddings)
    return cluster_labels


def _get_cluster_labels_kmeans(embeddings: np.ndarray) -> np.ndarray:
    model = MiniBatchKMeans(n_clusters=50)
    cluster_labels = model.fit_predict(embeddings)
    return cluster_labels
