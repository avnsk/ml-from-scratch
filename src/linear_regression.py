import numpy as np
import matplotlib.pyplot as plt
import argparse
import utils as utils


def fit_linear_regression(X, Y):
    """
    w = (X^T X)^(-1) X^T y
    """
    return np.linalg.inv(X.T @ X) @ (X.T @ Y)


def fit_linear_regression_gradient_decent(X, Y, lr=0.01, steps=1000):
    losses = []
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    for _ in range(steps):
        y_pred = utils.predict(X, w)
        error = y_pred - Y

        # ∇L(w)=2/n*(​X^T(Xw−y))
        grad = (2 / n_samples) * (X.T @ error)
        w -= lr * grad
        losses.append(utils.mse(Y, y_pred))
    return w, losses
