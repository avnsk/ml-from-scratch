import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils
from linear_regression import (
    fit_linear_regression,
    fit_linear_regression_gradient_decent,
)

from logistic_regression import (
    fit_logistic_regression,
    predict as logistic_predict,
    predict_proba,
)


def parse_args():
    parser = argparse.ArgumentParser("ML from Scratch")

    parser.add_argument("--train", type=str, help="Training CSV path")
    parser.add_argument("--predict", type=str, help="Prediction CSV path")
    parser.add_argument("--weights", type=str, help="Path to saved weights")
    parser.add_argument("--model", choices=["linear", "logistic"], default="linear")
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
        X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.2)
        print(f"\nTraining {args.model.upper()} regression...\n")
        # Linear Regression
        if args.model == "linear":
            if args.method == "normal":
                w = fit_linear_regression(X_train, y_train)
            else:
                X_train, means, stds = utils.normalise_features(X_train)
                X_test[:, 1:] = (X_test[:, 1:] - means) / stds
                w, losses = fit_linear_regression_gradient_decent(
                    X_train, y_train, args.lr, args.epochs
                )
            np.save("weights.npy", w)
            np.save("scaler_means.npy", means)
            np.save("scaler_stds.npy", stds)
            # Predictions
            train_pred = utils.predict(X_train, w)
            test_pred = utils.predict(X_test, w)
            print("\nTraining complete\n")

            print("Train Metrics:")
            print("MSE :", utils.mse(y_train, train_pred))
            print("RMSE:", utils.rmse(y_train, train_pred))
            print("MAE :", utils.mae(y_train, train_pred))

            print("\nTest Metrics:")
            print("MSE :", utils.mse(y_test, test_pred))
            print("RMSE:", utils.rmse(y_test, test_pred))
            print("MAE :", utils.mae(y_test, test_pred))

            plot_regression(X, y, w)
        elif args.model == "logistic":
            w, losses = fit_logistic_regression(
                X_train, y_train.ravel(), lr=args.lr, epochs=args.epochs
            )
            np.save("weights.npy", w)

            y_train_pred = logistic_predict(X_train, w)
            y_test_pred = logistic_predict(X_test, w)

            print("Train Metrics:")
            print("Accuracy :", utils.accuracy(y_train.ravel(), y_train_pred))
            print("Precision:", utils.precision(y_train.ravel(), y_train_pred))
            print("Recall   :", utils.recall(y_train.ravel(), y_train_pred))

            print("\nTest Metrics:")
            print("Accuracy :", utils.accuracy(y_test.ravel(), y_test_pred))
            print("Precision:", utils.precision(y_test.ravel(), y_test_pred))
            print("Recall   :", utils.recall(y_test.ravel(), y_test_pred))
    if args.predict:
        if not args.weights:
            raise ValueError("Provide --weights for prediction")
        w = np.load(args.weights)
        print(w)
        X, Y = utils.load_dataset(args.predict)
        if args.model == "linear":
            if args.method == "gd":
                means = np.load("scaler_means.npy")
                stds = np.load("scaler_stds.npy")
                X[:, 1:] = (X[:, 1:] - means) / stds

            y_pred = utils.predict(X, w)
            print("Predictions:")
            print(y_pred[:10])
        elif args.model == "logistic":
            y_pred = predict_proba(X, w)
            fprs, tprs = utils.roc_curve(Y, y_pred)
            auc = utils.auc_score(fprs, tprs)
            print(f"\nROC AUC Score: {auc:.4f}")
            print("Predictions (first 10):")
            print(y_pred[:10])


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
