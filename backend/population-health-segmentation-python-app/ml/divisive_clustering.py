#Divisive Clustering (Top-Down)
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


class DivisiveClustering:

    def __init__(
        self,
        n_clusters=8,
        random_state=42
    ):

        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, X):

        n_samples = X.shape[0]

        labels = np.zeros(
            n_samples,
            dtype=int
        )

        clusters = {
            0: np.arange(n_samples)
        }

        next_cluster_id = 1

        while len(clusters) < self.n_clusters:

            # Find largest cluster

            largest_cluster = max(
                clusters,
                key=lambda k: len(
                    clusters[k]
                )
            )

            indices = clusters[
                largest_cluster
            ]

            if len(indices) < 2:
                break

            # Split using KMeans

            kmeans = KMeans(
                n_clusters=2,
                random_state=self.random_state,
                n_init=20
            )

            split_labels = (
                kmeans.fit_predict(
                    X[indices]
                )
            )

            cluster_a = indices[
                split_labels == 0
            ]

            cluster_b = indices[
                split_labels == 1
            ]

            del clusters[
                largest_cluster
            ]

            clusters[
                largest_cluster
            ] = cluster_a

            clusters[
                next_cluster_id
            ] = cluster_b

            next_cluster_id += 1

        final_labels = np.zeros(
            n_samples,
            dtype=int
        )

        for cluster_id, idx in (
            clusters.items()
        ):
            final_labels[idx] = cluster_id

        self.labels_ = final_labels

        return final_labels