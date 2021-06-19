
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans


X, y_true = make_blobs(n_samples=500, centers=3, cluster_std= 0.6, random_state=40)
plt.scatter(X[:, 0], X[:, 1], s=50)

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=1)

x, y = make_moons(500, noise=.05, random_state=42)
labels = KMeans(5, random_state=42).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis')

model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis')

intertia = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    intertia.append(km.inertia_)

plt.plot(K, intertia, marker= "x")
plt.xlabel('k')
plt.xticks(np.arange(20))
plt.ylabel('Intertia')
plt.title('Elbow Method')
plt.show()

