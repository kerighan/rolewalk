Structural role embedding on directed graphs
============================================

```python
import networkx as nx
from sklearn.cluster import KMeans
from rolewalk import RoleWalk
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

# instantiate balanced tree
G = nx.balanced_tree(2, 6)

# create embeddings
X = RoleWalk(walk_len=5).transform(G)
y = KMeans(7).fit_predict(X)

# draw graph
pos = graphviz_layout(G, prog="dot")
nx.draw(G, node_color=y, node_size=50, pos=pos)
plt.show()
```
