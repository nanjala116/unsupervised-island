"""polynomial_regression.py:
Polynomial Regression Module
This module provides functions to generate data, create model matrices,
train a polynomial regression model, and test the coefficients.
"""

import numpy as np
from numpy.linalg import norm, solve

def generate_data(p, beta, sig, n):
    u = np.random.rand(n, 1)
    y = (u ** np.arange(0, p + 1)) @ beta + sig * np.random.randn(n, 1)
    return u, y

def model_matrix(p, u):
    n = u.shape[0]
    X = np.ones((n, 1))
    p_range = np.arange(1, p + 1)
    for p_current in p_range:
        X = np.hstack((X, u**(p_current)))
    return X

def train(X, y):
    betahat = solve(X.T @ X, X.T @ y)
    return betahat

def test_coefficients(n, betahat, X, y):
    y_hat = X @ betahat
    loss = (norm(y - y_hat)**2 / n)
    return loss
