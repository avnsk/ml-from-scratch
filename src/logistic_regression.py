import numpy as np
import matplotlib.pyplot as plt
import utils as utils


def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def predict_proba(X, w: float):
    z = X @ w
    return sigmoid(z)


def predict(X, w, threshold=0.5):
    probability = predict_proba(X, w)
    return (probability >= threshold).astype(int)


def decision_boundary(w):
    return -w[0] / w[1]


def binary_cross_entropy(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def fit_logistic_regression(X, y, lr=0.001, epochs=1000, reg=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []
    for _ in range(epochs):
        y_predict = predict_proba(X, w)
        loss = binary_cross_entropy(y, y_predict) + reg * np.sum(w**2)
        losses.append(loss)
        grad = (1 / n_samples) * X.T @ (y_predict - y) + 2 * reg * w
        w -= lr * grad
    return w, losses


def plot_decision_boundary_2d(X, y, w):
    # X includes bias: [1, x1, x2]
    x1_vals = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    x2_vals = -(w[0] + w[1] * x1_vals) / w[2]

    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], color="blue", label="Class 0")
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color="red", label="Class 1")
    plt.plot(x1_vals, x2_vals, color="green", label="Decision Boundary")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("2D Logistic Regression Decision Boundary")
    plt.show()


def test_boundary(w):
    x_vals = np.linspace(-1, 4, 200)
    X_plot = np.c_[np.ones(len(x_vals)), x_vals]
    probs = predict_proba(X_plot, w)

    plt.plot(x_vals, probs, label="Sigmoid")
    plt.axhline(0.5, linestyle="--", color="gray")
    plt.axvline(decision_boundary(w), linestyle="--", color="red", label="Boundary")

    plt.scatter([0, 1, 2, 3], [0, 0, 1, 1], c="black")
    plt.xlabel("x")
    plt.ylabel("P(y=1)")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.show()


if __name__ == "__main__":
    X = np.array([[1, 1], [2, 1], [2, 2], [3, 2], [3, 3], [4, 3], [4, 4], [5, 4]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X = np.c_[np.ones(len(X)), X]  # shape: (n, 3)

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.25)
    w, losses = fit_logistic_regression(X_train, y_train, lr=0.1, epochs=2000)
    plot_decision_boundary_2d(X_train, y_train, w)

    # print("Final weights:", w)
    # print("Final loss:", losses[-1])
    # plt.plot(losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Log Loss")
    # plt.title("Logistic Regression Training Loss")
    # plt.show()
