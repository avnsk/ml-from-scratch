import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 50)

y = 2*x + 5

noise = np.random.normal(0, 1, size = len(x))

y = y + noise

plt.scatter(x, y)
plt.xlabel("x (input features)")
plt.ylabel("y (target)")
plt.title("Synthetic Linear Regression Data")
plt.show()