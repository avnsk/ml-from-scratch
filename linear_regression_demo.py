import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# For reproducibilty purpose.
np.random.seed(42)


def load_dataset(csv_path):
    """
    Expects CSV with columns: x1, x2, ..., y
    """
    df = pd.read_csv(csv_path)

    X = df.drop("y", axis=1).values
    y = df["y"].values.reshape(-1, 1)

    # Add bias column
    X_bias = np.c_[np.ones(len(X)), X]
    return X_bias, y


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


def fit_linear_regression_gradient_decent(X, Y, lr=0.01, steps=1000):
    losses = []
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    for _ in range(steps):
        y_pred = predict(X, w)
        error = y_pred - Y

        # ∇L(w)=2/n*(​X^T(Xw−y))
        grad = (2 / n_samples) * (X.T @ error)
        w -= lr * grad
        losses.append(mse(Y, y_pred))
    return w, losses


def predict(X, w):
    return X @ w


def train(csv_path, method, lr, epochs):
    X, y = load_dataset(csv_path)

    if method == "normal":
        w = fit_linear_regression(X, y)
        losses = None
    else:
        w, losses = fit_linear_regression_gradient_decent(X, y, lr, epochs)

    return w, losses, X, y


def parse_args():
    parser = argparse.ArgumentParser("Linear Regression from Scratch")

    parser.add_argument("--train", type=str, help="Training CSV path")
    parser.add_argument("--predict", type=str, help="Prediction CSV path")

    parser.add_argument(
        "--method",
        choices=["normal", "gd"],
        default="normal",
        help="Training method",
    )

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--weights", type=str, help="Path to saved weights (.npy)")

    return parser.parse_args()


def plot_regression(X, y, w):
    """
    Only plots if 1 feature (+ bias)
    """
    if X.shape[1] != 2:
        print("Skipping regression plot (requires single feature).")
        return

    x_vals = X[:, 1]
    y_pred = X @ w

    plt.scatter(x_vals, y)
    plt.plot(x_vals, y_pred, color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    if args.train:
        X, y = load_dataset(args.train)

        if args.method == "normal":
            w = fit_linear_regression(X, y)
            losses = None
        else:
            w, losses = fit_linear_regression_gradient_decent(
                X, y, args.lr, args.epochs
            )

        plot_regression(X, y, w)
