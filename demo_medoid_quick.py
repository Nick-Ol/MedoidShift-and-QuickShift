"""
Comparison of Medoidshift and Quickshift algorithms with Meanshift and KMeans
"""


import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from medoid_shift_ import MedoidShift
from quick_shift_ import QuickShift

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plt.figure(figsize=(17, 9.5))
plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
for i_dataset, dataset in enumerate([noisy_circles, noisy_moons, blobs,
                                     no_structure]):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # create clustering estimators
    mean_s = cluster.MeanShift()
    two_means = cluster.KMeans(n_clusters=2)
    medoid_s_flat = MedoidShift(window_type="flat")
    medoid_s_norm = MedoidShift(window_type="normal")
    quick_s_flat = QuickShift(window_type="flat")
    quick_s_norm = QuickShift(window_type="normal")

    for name, algorithm in [
        ('MiniBatchKMeans', two_means),
        ('MeanShift', mean_s),
        ('MedoidShift flat', medoid_s_flat),
        ('MedoidShift normal', medoid_s_norm),
        ('QuickShift flat', quick_s_flat),
        ('QuickShift normal', quick_s_norm)
            ]:
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        y_pred = algorithm.labels_.astype(np.int)

        # plot
        plt.subplot(4, 6, plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        centers = algorithm.cluster_centers_
        center_colors = colors[:len(centers)]
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()
