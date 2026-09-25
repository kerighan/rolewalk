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

Choosing theta
--------------

Each walk step compares nodes through the characteristic function of their
transition probabilities, sampled at `n_samples` values of theta. The squared
distance between two embeddings is an MMD under the kernel
`K(d) = mean_theta cos(theta * d)`, so theta sets the kernel's bandwidth.

Since 1.1.0 theta is log-spaced on `[1, 100]` (`theta_scheme="geomspace"`),
which matches the best uniform grid on noise-free graphs and is much less
sensitive to the upper bound under edge noise. The 1.0 behaviour is
available with `theta_scheme="linspace"`. See `benchmarks/theta_study.py`.
