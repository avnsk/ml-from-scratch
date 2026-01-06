import numpy as np
import matplotlib.pyplot as plt
import argparse
import utils as utils


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
        y_pred = utils.predict(X, w)
        error = y_pred - Y

        # ∇L(w)=2/n*(​X^T(Xw−y))
        grad = (2 / n_samples) * (X.T @ error)
        w -= lr * grad
        losses.append(utils.mse(Y, y_pred))
    return w, losses


def train(csv_path, method, lr, epochs):
    X, y = utils.load_dataset(csv_path)

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
        X, y = utils.load_dataset(args.train)

        if args.method == "normal":
            w = fit_linear_regression(X, y)
            losses = None
        else:
            w, losses = fit_linear_regression_gradient_decent(
                X, y, args.lr, args.epochs
            )

        plot_regression(X, y, w)
