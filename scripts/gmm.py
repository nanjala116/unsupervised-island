
""" gmm.py:
Gaussian Mixture Model Module
This module implements a custom Expectation-Maximization algorithm for Gaussian Mixture Models.
It provides functions to calculate the PDF of a multivariate normal distribution,
"""

import numpy as np
from scipy.stats import multivariate_normal

def normal_pdf(m, Xmat, var=1.0):
    """Calculates the PDF of a multivariate normal distribution."""
    D = Xmat.shape[1]
    # Create a covariance matrix with variance `var` on the diagonal
    cov_matrix = np.eye(D) * var
    try:
        mvn = multivariate_normal(mean=m, cov=cov_matrix)
        return mvn.pdf(Xmat)
    except np.linalg.LinAlgError:
        # Handle cases where covariance might not be positive definite
        return np.zeros(Xmat.shape[0])

def normal_train(p, Xmat):
    """Updates the mean based on posterior probabilities."""
    # Add a small epsilon to avoid division by zero if a cluster has no points
    sum_p = np.sum(p) + 1e-9 
    m = (Xmat.T @ p.T) / sum_p
    return m

def exp_max_custom(Iter, K, pdf, train, Xmat, W_Init, P_Init, var=1.0):
    """
    Custom Expectation-Maximization for a Gaussian Mixture Model.
    """
    n, D = Xmat.shape
    p = np.zeros((K, n))
    W, P = W_Init.copy(), P_Init.copy()

    for i in range(Iter):
        # E-Step: Calculate posterior probabilities
        for k in range(K):
            # Pass the variance parameter to the pdf function
            p[k, :] = W[0, k] * pdf(P[:, k], Xmat, var=var)
        
        # Normalize probabilities to sum to 1 for each data point
        p_sum = np.sum(p, axis=0)
        p_sum[p_sum == 0] = 1.0 # Avoid division by zero
        p = p / p_sum

        # M-Step: Update weights and parameters (means)
        W = np.mean(p, axis=1).reshape(1, K)
        for k in range(K):
            P[:, k] = train(p[k, :], Xmat)
            
    # Final assignment of labels
    labels = np.argmax(p, axis=0)
    return W, P, labels
