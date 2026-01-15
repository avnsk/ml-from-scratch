import numpy as np
import matplotlib.pyplot as plt


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
    X = np.c_[np.ones(4), [0, 1, 2, 3]]
    y = np.array([0, 0, 1, 1])
    w, losses = fit_logistic_regression(X, y)
    test_boundary(w)
    # print("Final weights:", w)
    # print("Final loss:", losses[-1])
    # plt.plot(losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Log Loss")
    # plt.title("Logistic Regression Training Loss")
    # plt.show()
