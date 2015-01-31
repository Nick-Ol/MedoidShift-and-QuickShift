""" Quick shift implementation based on the article Quick Shift and Kernel
Methods for Mode Seeking by  A. Vedaldi and S. Soatto, 2008.
All points are connected into a single tree where the root is the point
whith maximal estimated density. Thus, we need at final a threshold parameter
tau, to break the branches that are longer than tau.
Complexity in O(n_features*n_samples**2).

Any contribution is welcomed
"""

# Author: Clement Nicolle <clement.nicolle@student.ecp.fr>

from __future__ import division
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


def compute_medoids(dist_matrix, weight_matrix, tau):
    """For each point, compute the associated medoid.

    Parameters
    ----------
    dist_matrix : array-like, shape=[n_samples, n_samples]
        Distance matrix.

    weight_matrix : array-like, shape=[n_samples, n_samples]
        Weight for each pair of points.

    tau : float
        Threshold parameter. Distance should not be over tau so that two points
        may be connected to each other.

    Returns
    -------
    medoids : array, shape=[n_samples]
        i-th value is the index of the medoid for i-th point.
    """

    P = sum(weight_matrix)
    # P[i,j] = P[i] - P[j]
    P = P[:, np.newaxis] - P
    dist_matrix[dist_matrix == 0] = tau/2
    S = P * (1/dist_matrix)  # pointwise product
    S[dist_matrix > tau] = -1

    # new medoid for point j highest coef in the j-th column of S
    return np.argmax(S, axis=0)


def compute_stationary_medoids(data, tau, window_type, bandwidth, metric):
    """Return the indices of the own medoids.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.

    tau : float
        Threshold parameter. Distance should not be over tau so that two points
        may be connected to each other.

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
    medoids = compute_medoids(dist_matrix, weight_matrix, tau)
    stationary_idx = []
    for i in range(len(medoids)):
        if medoids[i] == i:
            stationary_idx.append(i)
    return medoids, np.asarray(stationary_idx)


def quick_shift(data, tau, window_type, bandwidth, metric):
    """Perform medoid shiftclustering of data with corresponding parameters.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input points.

    tau : float
        Threshold parameter. Distance should not be over tau so that two points
        may be connected to each other.

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

    if tau is None:
        tau = estimate_bandwidth(data)
    if bandwidth is None:
        bandwidth = estimate_bandwidth(data)

    medoids, cluster_centers_idx = compute_stationary_medoids(data, tau,
                                                              window_type,
                                                              bandwidth,
                                                              metric)
    cluster_centers = data[cluster_centers_idx]
    labels = []
    labels_val = {}
    lab = 0
    for i in cluster_centers_idx:
        labels_val[i] = lab
        lab += 1
    for i in range(len(data)):
        next_med = medoids[i]
        while next_med not in cluster_centers_idx:
            next_med = medoids[next_med]
        labels.append(labels_val[next_med])
    return cluster_centers, np.asarray(labels), cluster_centers_idx


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


class QuickShift():
    """ Compute the Quick shift algorithm with flat or normal window

    data : array-like, shape=[n_samples, n_features]
        Input points.

    tau : float
        Threshold parameter. Distance should not be over tau so that two points
        may be connected to each other.

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

    def __init__(self, tau=None, bandwidth=None,
                 window_type="flat", metric="euclidean"):

        self.tau = tau
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
            quick_shift(data, tau=self.tau, window_type=self.window_type,
                        bandwidth=self.bandwidth, metric=self.metric)

        return self
