"""polynomial_regression_cv.py:
Cross-Validation for Polynomial Regression
This module implements K-fold cross-validation for polynomial regression.
It uses functions from the polynomial regression module to train and test models.
"""

import numpy as np
from polynomial_regression import train, test_coefficients

def linear_cv(K, n, X, y):
    loss = []
    fold_size = int(n / K)
    for k in range(K):
        start = k * fold_size
        end = (k + 1) * fold_size
        test_ind = np.arange(start, end).astype('int')
        train_ind = np.setdiff1d(np.arange(n), test_ind)

        X_train, y_train = X[train_ind, :], y[train_ind]
        X_test, y_test = X[test_ind, :], y[test_ind]

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            continue

        betahat = train(X_train, y_train)
        loss.append(test_coefficients(len(y_test), betahat, X_test, y_test))
    
    cv = np.mean(loss) if loss else float('inf')
    return cv
