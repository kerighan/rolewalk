from karateclub.dataset import GraphReader
from karateclub.node_embedding.structural import GraphWave
import matplotlib.pyplot as plt
from rolewalk2 import rolewalk2
import time

reader = GraphReader("twitch")
G = reader.get_graph()
print(G, G.is_directed())

start_time = time.time()
X = rolewalk2(G, walk_len=3, embedding_dim=2)
# mdl = GraphWave(sample_number=20)
# mdl.fit(G)
# X = mdl.get_embedding()
elapsed = time.time() - start_time
print(f"T={elapsed:.2f}s")
print(X.shape)

plt.scatter(X[:, 0], X[:, 1])
plt.show()
