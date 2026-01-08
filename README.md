# ML From Scratch

This repository is a **hands-on Machine Learning project** focused on implementing core ML algorithms **from first principles** using Python and NumPy.

The goal is to deeply understand how Machine Learning works internally by manually implementing:
- model equations
- loss functions
- optimization algorithms
- feature scaling
- evaluation metrics
- command-line interfaces

No high-level ML libraries (e.g. scikit-learn, TensorFlow, PyTorch) are used.

---

## 🚀 Implemented Features

### Linear Regression (From Scratch)
- Normal Equation solution
- Gradient Descent optimization
- Mean Squared Error (MSE) loss
- Training and prediction pipeline

### Feature Normalization
- Z-score normalization (mean = 0, std = 1)
- Bias term excluded from normalization
- Normalization parameters reused during inference
- Stable and faster gradient descent convergence

### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### Command-Line Interface (CLI)
Run training and prediction directly from the terminal.

```bash
python cli.py --train data/train.csv --method gd
python cli.py --predict data/predict.csv --weights weights.npy
