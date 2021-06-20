import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('BrainTumor.csv')
X = dataset.iloc[:, [0,14]].values



def rand_jitter(arr):
    stdev = 0.01
    return arr + np.random.randn(len(arr)) * stdev

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters


# Splitting Zeros into array and Ones into array

Zeros = []
Ones = []
k=0
for i in X:
    if(i[0] == 0):
        Zeros.append(i.tolist())
    k+=1

k = 0
for i in X:
    if(i[0] == 1):
        Ones.append(i.tolist())
    k += 1
Zeros = np.array(Zeros)
Ones = np.array(Ones)

plt.scatter(rand_jitter(Zeros[:, 0]), rand_jitter(Zeros[:, 1]),
            s=1, c='blue', label='Cluster 1')
plt.scatter(rand_jitter(Ones[:, 0]), rand_jitter(Ones[:,
            1]), s=1, c='red', label='Cluster 2')

# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Brain Tumors Clustring')
plt.xlabel('Class')
plt.ylabel('Number of instances')

plt.legend()
plt.show()
