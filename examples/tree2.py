import networkx as nx
from sklearn.cluster import KMeans
from rolewalk2 import rolewalk2
from rolewalk import rolewalk
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np

# instantiate balanced tree
# G = nx.balanced_tree(2, 7)
G = nx.barbell_graph(10, 5)

# create embeddings
X = rolewalk2(G, walk_len=4, embedding_dim=2)
y = KMeans(8).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# draw graph
pos = graphviz_layout(G, prog="dot")
nx.draw(G, node_color=y, node_size=50, pos=pos)
plt.show()
