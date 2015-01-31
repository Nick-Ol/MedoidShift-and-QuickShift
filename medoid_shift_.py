""" Medoid shift implementation based on the article Mode-seeking Medoidshifts
by  Y. A. Sheikh, E. A. Khan, and T. Kanade, 2007.
Complexity in O(n_samples**3 + n_features*n_samples**2),
caution if n_samples too large.

Any contribution is welcomed
"""

# Author: Clement Nicolle <clement.nicolle@student.ecp.fr>

import numpy as np
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt


def compute_distance_matrix(data, metric):
    """Compute the distance between each pair of points.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.

    metric : string
        Metric used to compute the distance. See pairwise_distances doc to
        look at all the possible values.

    Returns
    -------
    distance_matrix : array-like, shape=[n_samples, n_samples]
        Distance between each pair of points.
    """

    return pairwise_distances(data, metric=metric)


def compute_weight_matrix(dist_matrix, window_type, bandwidth):
    """Compute the weight of each pair of points, according to the window
    chosen.

    Parameters
    ----------
    dist_matrix : array-like, shape=[n_samples, n_samples]
        Distance matrix.

    window_type : string
        Type of window to compute the weights matrix. Can be
        "flat" or "normal".

    bandwidth : float
        Value of the bandwidth for the window.

    Returns
    -------
    weight_matrix : array-like, shape=[n_samples, n_samples]
        Weight for each pair of points.
    """

    if window_type == 'flat':
        # 1* to convert boolean in int
        weight_matrix = 1*(dist_matrix <= bandwidth)
    elif window_type == 'normal':
        weight_matrix = np.exp(-dist_matrix**2 / (2 * bandwidth**2))
    else:
        raise ValueError("Unknown window type")
    return weight_matrix


def compute_medoids(dist_matrix, weight_matrix):
    """For each point, compute the associated medoid.

    Parameters
    ----------
    dist_matrix : array-like, shape=[n_samples, n_samples]
        Distance matrix.

    weight_matrix : array-like, shape=[n_samples, n_samples]
        Weight for each pair of points.

    Returns
    -------
    medoids : array, shape=[n_samples]
        i-th value is the index of the medoid for i-th point.
    """

    S = np.dot(dist_matrix, weight_matrix)
    # new medoid for point i lowest coef in the i-th column of S
    return np.argmin(S, axis=0)


def compute_stationary_medoids(data, window_type, bandwidth, metric):
    """Return the indices of the own medoids.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.

    window_type : string
        Type of window to compute the weights matrix. Can be
        "flat" or "normal".

    bandwidth : float
        Value of the bandwidth for the window.

    metric : string
        Metric used to compute the distance. See pairwise_distances doc to
        look at all the possible values.

    Returns
    -------
    medoids : array, shape=[n_samples]
        i-th value is the index of the medoid for i-th point.

    stationary_pts : array, shape=[n_stationary_pts]
        Indices of the points which are their own medoids.
    """
    dist_matrix = compute_distance_matrix(data, metric)
    weight_matrix = compute_weight_matrix(dist_matrix, window_type, bandwidth)
    medoids = compute_medoids(dist_matrix, weight_matrix)
    stationary_idx = []
    for i in range(len(medoids)):
        if medoids[i] == i:
            stationary_idx.append(i)
    return medoids, np.asarray(stationary_idx)


def medoid_shift(data, window_type, bandwidth, metric):
    """Perform medoid shiftclustering of data with corresponding parameters.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.

    window_type : string
        Type of window to compute the weights matrix. Can be
        "flat" or "normal".

    bandwidth : float
        Value of the bandwidth for the window.

    metric : string
        Metric used to compute the distance. See pairwise_distances doc to
        look at all the possible values.

    Returns
    -------
    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.

    labels : array, shape=[n_samples]
        Cluster labels for each point.

    cluster_centers_idx : array, shape=[n_clusters]
        Index in data of cluster centers.
    """

    if bandwidth is None:
        bandwidth = estimate_bandwidth(data)

    medoids, stat_idx = compute_stationary_medoids(data, window_type,
                                                   bandwidth, metric)
    new_data = data[stat_idx]
    new_medoids, new_stat_idx = compute_stationary_medoids(new_data,
                                                           window_type,
                                                           bandwidth, metric)
    if len(new_stat_idx) == len(new_data):
        cluster_centers = new_data
        labels = []
        labels_val = {}
        lab = 0
        for i in stat_idx:
            labels_val[i] = lab
            lab += 1
        for i in range(len(data)):
            next_med = medoids[i]
            while next_med not in stat_idx:
                next_med = medoids[next_med]
            labels.append(labels_val[next_med])
        return cluster_centers, np.asarray(labels), stat_idx

    else:
        cluster_centers, next_labels, next_clusters_centers_idx = \
            medoid_shift(new_data, window_type, bandwidth, metric)
        clusters_centers_idx = stat_idx[next_clusters_centers_idx]
        labels = []
        for i in range(len(data)):
            next_med = medoids[i]
            while next_med not in stat_idx:
                next_med = medoids[next_med]
            # center associated to the medoid in next iteration
            next_med_new_idx = np.where(stat_idx == next_med)[0][0]
            labels.append(next_labels[next_med_new_idx])
        return cluster_centers, np.asarray(labels), clusters_centers_idx


def visualize2D(data, labels, clusters_centers_idx):
    """Plot clustering result if points in 2D

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.

    labels : array, shape=[n_samples]
        Cluster labels for each point.

    cluster_centers_idx : array, shape=[n_clusters]
        Index in data of cluster centers.
    """

    n_samples = len(data)
    K = len(clusters_centers_idx)
    colors = []
    # generate random colors vector :
    for i in range(K):
        colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(0, n_samples):
        cluster = int(labels[i])
        ax.scatter(data[i, 0], data[i, 1], color=colors[cluster])
    for j in range(0, K):
        ax.scatter(data[clusters_centers_idx[j], 0],
                   data[clusters_centers_idx[j], 1],
                   color='k', marker='x', s=100)
        # clusters centers as large black X


class MedoidShift():
    """ Compute the Medoid shift algorithm with flat or normal window

    data : array-like, shape=[n_samples, n_features]
        Input points.

    window_type : string
        Type of window to compute the weights matrix. Can be
        "flat" or "normal".

    bandwidth : float
        Value of the bandwidth for the window.

    metric : string
        Metric used to compute the distance. See pairwise_distances doc to
        look at all the possible values.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ :
        Labels of each point
    cluster_centers_idx_ : array, shape=[n_clusters]
        Index in data of cluster centers.
    """

    def __init__(self, bandwidth=None,
                 window_type="flat", metric="euclidean"):

        self.bandwidth = bandwidth
        self.window_type = window_type
        self.metric = metric

    def fit(self, data):
        """Perform clustering.

         Parameters
        -----------
        data : array-like, shape=[n_samples, n_features]
            Samples to cluster.
        """

        self.cluster_centers_, self.labels_, self.cluster_centers_idx_ = \
            medoid_shift(data, window_type=self.window_type,
                         bandwidth=self.bandwidth, metric=self.metric)

        return self
