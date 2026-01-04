import numpy as np
import matplotlib.pyplot as plt

# For reproducibilty purpose.
np.random.seed(42)


def generate_data(n_samples=50, n_features=3, noise_std=0.5):

    X = np.random.rand(n_samples, n_features)
    true_w = np.array([[5], [2], [-3], [1]])
    X_bias = np.c_[np.ones(n_samples), X]
    noise = np.random.normal(0, noise_std, size=(n_samples, 1))
    y = X_bias @ true_w + noise
    return X, X_bias, y, true_w


def mse(y_true, y_pred):
    error = y_true - y_pred
    return np.mean(error**2)


def fit_linear_regression(X, Y):
    """
    w = (X^T X)^(-1) X^T y
    """
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def predict(X, w):
    return X @ w


if __name__ == "__main__":
    X, X_bias, y, true_w = generate_data()
    learned_w = fit_linear_regression(X_bias, y)
    y_pred = predict(X_bias, learned_w)

    print("True weights:")
    print(true_w)

    print("\nLearned weights:")
    print(learned_w)

    print("\nMean Squared Error:")
    print(mse(y, y_pred))

    plt.scatter(y, y_pred)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("True vs Predicted Values")
    plt.plot([y.min(), y.max()], [y.min(), y.max()])
    plt.show()
