import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
from rolewalk import RoleWalk
from sklearn.cluster import KMeans

G = nx.barbell_graph(10, 5)
X = RoleWalk(walk_len=3).fit_transform(G)
y = RoleWalk(walk_len=3).fit_predict(X)

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.subplot(122)
pos = graphviz_layout(G, prog="neato")
nx.draw(G, node_color=y, node_size=50, pos=pos)
plt.show()
