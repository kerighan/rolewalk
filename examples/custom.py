import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from rolewalk2 import rolewalk2
import time


G = nx.DiGraph()
G.add_edges_from([
    ("A", "B"),
    ("A", "C"),
    ("E", "F"),
    ("G", "H"),
    ("I", "G"),
    ("D", "J"),
    ("D", "K"),
    ("A", "L"),
    ("A", "M"),
    ("G", "N")
])
start_time = time.time()
X = rolewalk2(G, walk_len=5, embedding_dim=2)
# X = rolewalk(G, walk_len=6, method="entropy")
print(time.time() - start_time)
# X = X[:, :2]
# X = PCA(2).fit_transform(X)

y = KMeans(7).fit_predict(X)

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.subplot(122)
nx.draw_networkx(G, node_color=y)
plt.show()
