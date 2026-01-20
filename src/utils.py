import numpy as np
import pandas as pd

# For reproducibilty purpose.
np.random.seed(42)


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


def mse(y_true, y_pred):
    error = y_true - y_pred
    return np.mean(error**2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def validate_dataset(df):
    if "y" not in df.columns:
        raise ValueError("Dataset must contain a 'y' column")
    if len(df.columns) < 2:
        raise ValueError("Dataset must contain at least one feature column")
    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("All columns must be numeric")


def load_dataset(csv_path):
    """
    Expects CSV with columns: x1, x2, ..., y
    """
    df = pd.read_csv(csv_path)
    validate_dataset(df)
    X = df.drop("y", axis=1).values
    y = df["y"].values.reshape(-1, 1)

    # Add bias column
    X_bias = np.c_[np.ones(len(X)), X]
    return X_bias, y


def train_test_split(X, y, test_size=0.2, shuffle=True):
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def predict(X, w):
    return X @ w


def generate_data(n_samples=50, n_features=3, noise_std=0.5):

    X = np.random.rand(n_samples, n_features)
    true_w = np.array([[5], [2], [-3], [1]])
    X_bias = np.c_[np.ones(n_samples), X]
    noise = np.random.normal(0, noise_std, size=(n_samples, 1))
    y = X_bias @ true_w + noise
    return X, X_bias, y, true_w


def normalise_features(X):
    X_norm = X.copy()
    means = X_norm[:, 1:].mean(axis=0)
    stds = X_norm[:, 1:].std(axis=0)
    X_norm[:, 1:] = (X_norm[:, 1:] - means) / stds
    return X_norm, means, stds


def roc_curve(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    tprs = []
    fprs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

        tpr = tp / (tp + fn + 1e-15)
        fpr = fp / (fp + tn + 1e-15)

        tprs.append(tpr)
        fprs.append(fpr)

    return np.array(fprs), np.array(tprs)


def auc_score(fprs, tprs):
    order = np.argsort(fprs)
    fprs = fprs[order]
    tprs = tprs[order]

    # Trapezoidal rule
    return np.trapezoid(tprs, fprs)
