from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

X = load_iris().data
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
gmm = GaussianMixture(n_components=3, random_state=42).fit(X)
k_labels, g_labels = kmeans.predict(X), gmm.predict(X)
print(
    f"k-Means Silhouette: {silhouette_score(X, k_labels):.2f}, GMM Silhouette: {silhouette_score(X, g_labels):.2f}"
)
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=k_labels)
plt.title("k-Means")
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=g_labels)
plt.title("GMM (EM)")
plt.show()
