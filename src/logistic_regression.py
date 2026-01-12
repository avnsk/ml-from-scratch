import math


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def linear_combination(x: float, w: float, b: float) -> float:
    return w * x + b


def predict_proba(X, w: float, b: float):
    """
    Predict probabilities for inputs X
    """
    probs = []
    for x in X:
        z = linear_combination(x, w, b)
        probs.append(sigmoid(z))
    return probs


if __name__ == "__main__":
    X = [0, 1, 2, 3]
    y = [0, 0, 1, 1]

    w = 1.0
    b = -1.5

    probs = predict_proba(X, w, b)

    print("Probabilities:", probs)
