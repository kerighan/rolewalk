from karateclub.dataset import GraphReader
from karateclub.node_embedding.structural import GraphWave
import matplotlib.pyplot as plt
from rolewalk import RoleWalk
import time

reader = GraphReader("twitch")
G = reader.get_graph()
print(G, G.is_directed())

start_time = time.time()
X = RoleWalk(walk_len=3, embedding_dim=2).transform(G)
# mdl = GraphWave(sample_number=20)
# mdl.fit(G)
# X = mdl.get_embedding()
elapsed = time.time() - start_time
print(f"T={elapsed:.2f}s")
print(X.shape)

plt.scatter(X[:, 0], X[:, 1])
plt.show()
