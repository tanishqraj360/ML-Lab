import numpy as np
import matplotlib.pyplot as plt


def lwlr(x0, X, Y, tau):
    X_b = np.c_[np.ones(len(X)), X]
    x0_b = np.r_[1, x0]
    w = np.exp(-np.sum((X_b - x0_b) ** 2, axis=1) / (2 * tau**2))
    W = np.diag(w)
    beta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ Y
    return x0_b @ beta


X = np.linspace(-3, 3, 200)
Y = np.log(np.abs(X**2 - 1) + 0.5) + np.random.normal(0, 0.1, 200)
domain = np.linspace(-3, 3, 300)

for tau in [10, 1, 0.1, 0.01]:
    plt.plot(domain, [lwlr(x, X, Y, tau) for x in domain], label=f"tau={tau}")
plt.scatter(X, Y, alpha=0.3)
plt.legend()
plt.show()
