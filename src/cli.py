import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils
from linear_regression import (
    fit_linear_regression,
    fit_linear_regression_gradient_decent,
)


def parse_args():
    parser = argparse.ArgumentParser("Linear Regression from Scratch")

    parser.add_argument("--train", type=str, help="Training CSV path")
    parser.add_argument("--predict", type=str, help="Prediction CSV path")
    parser.add_argument("--weights", type=str, help="Path to saved weights")

    parser.add_argument("--method", choices=["normal", "gd"], default="normal")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)

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


def main():
    args = parse_args()
    if args.train:
        X, y = utils.load_dataset(args.train)
        if args.method == "normal":
            w = fit_linear_regression(X, y)
        else:
            X, means, stds = utils.normalise_features(X)
            w, losses = fit_linear_regression_gradient_decent(
                X, y, args.lr, args.epochs
            )
        np.save("weights.npy", w)
        np.save("scaler_means.npy", means)
        np.save("scaler_stds.py", stds)
        y_pred = utils.predict(X, w)

        print("Training complete")
        print("MSE :", utils.mse(y, y_pred))
        print("RMSE:", utils.rmse(y, y_pred))
        print("MAE :", utils.mae(y, y_pred))

        plot_regression(X, y, w)
    if args.predict:
        if not args.weights:
            raise ValueError("Provide --weights for prediction")

        w = np.load(args.weights)
        print(w)
        X, _ = utils.load_dataset(args.predict)
        if args.method == "gd":
            means = np.load("scaler_means.npy")
            stds = np.load("scaler_stds.npy")
            X[:, 1:] = (X[:, 1:] - means) / stds

        y_pred = utils.predict(X, w)
        print("Predictions:")
        print(y_pred[:10])


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
