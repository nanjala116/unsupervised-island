"""K-Means Clustering Module
This module implements a custom K-Means clustering algorithm.
It provides a function to perform K-Means clustering with a specified convergence criterion.
"""

import numpy as np

def kmeans_custom(eps, K, Xmat, c_init):
    """
    Custom K-Means clustering implementation.
    """
    n, D = Xmat.shape
    c = c_init.copy() # Use a copy to avoid modifying the original
    c_old = np.zeros(c.shape)
    dist2 = np.zeros((K, n))

    # Loop until centroids stabilize
    while np.sum(np.abs(c - c_old)) > eps:
        c_old = c.copy()
        # E-step: Assign points to the nearest centroid
        for i in range(K):
            dist2[i, :] = np.sum((Xmat - c[:, i].T)**2, axis=1)
        label = np.argmin(dist2, axis=0)

        # M-step: Recompute centroids
        for i in range(K):
            entries = np.where(label == i)[0]
            if len(entries) > 0:
                c[:, i] = np.mean(Xmat[entries, :], axis=0)
            # If a centroid has no points, it remains unchanged (or could be re-initialized)
            
    return c, label
