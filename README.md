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

### 🔢 Linear Regression (From Scratch)
- Normal Equation solution
- Gradient Descent optimization
- Mean Squared Error (MSE) loss
- Training and prediction pipeline
- Feature normalization (Z-score)
- Train/Test split for evaluation

---

### 🔐 Logistic Regression (Binary Classification)
- Sigmoid activation
- Binary Cross-Entropy (Log Loss)
- Gradient Descent optimization
- Probability prediction + class prediction
- Train/Test split
- Classification metrics:
  - Accuracy
  - Precision
  - Recall
- 2D Decision Boundary visualization

---

### 📊 Evaluation Metrics

**Regression:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

**Classification:**
- Accuracy
- Precision
- Recall
- Confusion Matrix (manual implementation)

---

### ⚙️ Feature Normalization
- Z-score normalization (mean = 0, std = 1)
- Bias term excluded from normalization
- Normalization parameters reused during inference
- Improves gradient descent stability and convergence

---

### 🖥️ Command-Line Interface (CLI)

Train and evaluate models directly from the terminal.

#### ➤ Train Linear Regression
```
python src/cli.py --train data/train.csv --model linear --method gd --lr 0.1 --epochs 2000
```
#### ➤ Train Logistic Regression
```
python src/cli.py --train data/logistic_train.csv --model logistic --lr 0.1 --epochs 2000
```
### ➤ Run Prediction
```
python src/cli.py --predict data/new_data.csv --weights weights.npy --model logistic
```