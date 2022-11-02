import time

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from rolewalk import RoleWalk
from sklearn.cluster import KMeans

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
X = RoleWalk().fit_transform(G)
print(time.time() - start_time)

y = KMeans(7).fit_predict(X)

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.subplot(122)
pos = graphviz_layout(G, prog="neato")
nx.draw_networkx(G, node_color=y, pos=pos, font_color="#FFFFFF")
plt.show()
