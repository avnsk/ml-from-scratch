import numpy as np
import matplotlib.pyplot as plt
import utils as utils


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp + 1e-15)


def recall(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn + 1e-15)


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
    X = np.c_[np.ones(8), [0, 1, 2, 3, 4, 5, 6, 7]]
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.25)
    w, losses = fit_logistic_regression(X_train, y_train)
    test_boundary(w)
    for t in [0.3, 0.5, 0.7]:
        y_pred = predict(X, w, threshold=t)
        print(f"\nThreshold = {t}")
        print("Accuracy:", accuracy(y, y_pred))
        print("Precision:", precision(y, y_pred))
        print("Recall:", recall(y, y_pred))

        y_test_pred = predict(X_test, w)
        print("\nTest metrics:")
        print("Accuracy:", accuracy(y_test, y_test_pred))
        print("Precision:", precision(y_test, y_test_pred))
        print("Recall:", recall(y_test, y_test_pred))
    # print("Final weights:", w)
    # print("Final loss:", losses[-1])
    # plt.plot(losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Log Loss")
    # plt.title("Logistic Regression Training Loss")
    # plt.show()
