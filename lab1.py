import numpy as np


class Adaline:
    def __init__(self, lr=0.01, iters=10):
        self.lr, self.iters = lr, iters

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        for _ in range(self.iters):
            output = np.dot(X, self.w[1:]) + self.w[0]
            error = y - output
            self.w[1:] += self.lr * X.T.dot(error)
            self.w[0] += self.lr * error.sum()

    def predict(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]


X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([1, 1, 0])
model = Adaline()
model.fit(X, y)
print("Weights:", model.w)
print("Prediction:", model.predict(np.array([2, 3])))
