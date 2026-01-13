import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def linear_combination(x: float, w: float, b: float) -> float:
    return w * x + b


def predict_proba(X, w: float):
    z = X @ w
    return sigmoid(z)


def binary_cross_entropy(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def fit_logistic_regression(X, y, lr=0.001, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []
    for _ in range(epochs):
        y_predict = predict_proba(X, w)
        loss = binary_cross_entropy(y, y_predict)
        losses.append(loss)
        grad = (1 / n_samples) * X.T @ (y_predict - y)
        w -= lr * grad
    return w, losses


if __name__ == "__main__":
    X = np.c_[np.ones(4), [0, 1, 2, 3]]
    y = np.array([0, 0, 1, 1])
    w, losses = fit_logistic_regression(X, y)
    print("Final weights:", w)
    print("Final loss:", losses[-1])
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Logistic Regression Training Loss")
    plt.show()
