import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
from rolewalk import RoleWalk
from sklearn.cluster import KMeans

# instantiate balanced tree
# G = nx.balanced_tree(2, 7)
G = nx.barbell_graph(10, 5)

# create embeddings
# X = RoleWalk(walk_len=3).fit_transform(G)
y = RoleWalk(walk_len=3).fit_predict(G)
print(y)
# y = KMeans(5).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# # draw graph
pos = graphviz_layout(G, prog="dot")
nx.draw(G, node_color=y, node_size=50, pos=pos)
plt.show()
