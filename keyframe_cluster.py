import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

source = "multi"
max_n_cluster = 20  # Typically 20 for cn and multi, 10 for us
perplexity = 10  # Smaller perplexity associates with tighter clusters
# Typically 10 for multi, 15 for cn, 20 for us
dir_path = "output/" + source + "/"

data_encode = np.genfromtxt(dir_path + "data.csv", delimiter = ",")
print(data_encode.shape)

tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = 3407)
data_2d = tsne.fit_transform(data_encode)
# plt.scatter(data_2d[:, 0], data_2d[:, 1], marker = ".", alpha = 0.5)
# plt.show()

if source == "multi":
    data_label = np.genfromtxt(dir_path + "data_label.csv", delimiter = ",")
    print(data_label.shape)
    fig = plt.scatter(data_2d[:, 0], data_2d[:, 1], c = -data_label, cmap = "coolwarm", marker = ".", alpha = 0.5)
    plt.legend(fig.legend_elements()[0], ["US", "China"])
    plt.show()

    gmm = GaussianMixture(n_components = 2, max_iter = 1000, random_state = 3407)
    gmm.fit(data_encode)
    pred_class = gmm.predict(data_encode)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c = pred_class, cmap = "tab20", marker = ".", alpha = 0.5)
    plt.show()

    data_gmm = pd.DataFrame({"clust": pred_class, "label": data_label})
    print(data_gmm.groupby("clust").count())
    print(np.where(pred_class == 0))

avg_inter_dist = []
avg_cross_dist = []
pop_array = np.zeros((max_n_cluster - 2, max_n_cluster - 1))
for k in range(2, max_n_cluster):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data_encode)

    inter_dist_mat = kmeans.transform(data_encode)
    avg_inter_dist.append(np.mean(np.min(inter_dist_mat, axis = 1)))

    cross_dist_mat = np.zeros((k, k))
    for i in range(k - 1):
        for j in range(i + 1, k):
            cross_dist_mat[i, j] = np.linalg.norm(kmeans.cluster_centers_[i, :] - kmeans.cluster_centers_[j, :], ord = 2)

    avg_cross_dist.append(np.sum(cross_dist_mat) / k / (k - 1))

    clust_pop = np.unique(kmeans.labels_, return_counts = True)
    for i, population in enumerate(np.sort(-clust_pop[1])):
        pop_array[k - 2, i] = -population

plt.plot(range(2, max_n_cluster), avg_inter_dist, ".-")
# plt.xticks(range(0, 25, 5))
plt.xlabel("Number of Clusters")
plt.ylabel("Inter-Cluster Distance")
plt.show()

plt.plot(range(2, max_n_cluster), avg_cross_dist, ".-")
# plt.xticks(range(0, 25, 5))
plt.xlabel("Number of Clusters")
plt.ylabel("Cross-Cluster Distance")
plt.show()

print(pop_array)
cum_pop = np.cumsum(pop_array, axis = 1)
plt.bar(range(2, max_n_cluster), pop_array[:, 0])
for i in range(max_n_cluster - 2):
    plt.bar(range(2, max_n_cluster), pop_array[:, i + 1], bottom = cum_pop[:, i])

# plt.xticks(range(0, 25, 5))
plt.xlabel("Number of clusters")
plt.ylabel("Population in each cluster")
plt.show()
