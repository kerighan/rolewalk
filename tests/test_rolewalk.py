import numpy as np
import networkx as nx

from rolewalk import RoleWalk


def test_transform_shapes():
    walk_len = 2
    n_samples = 5
    rw = RoleWalk(walk_len=walk_len, n_samples=n_samples, embedding_dim=None)

    # Undirected graph
    G = nx.path_graph(3)
    X = rw.transform(G)
    expected_dim = 2 * n_samples * walk_len
    assert X.shape == (G.number_of_nodes(), expected_dim)

    # Directed graph
    Gd = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    Xd = rw.transform(Gd)
    expected_d_dim = 4 * n_samples * walk_len
    assert Xd.shape == (Gd.number_of_nodes(), expected_d_dim)


def test_fit_predict_returns_valid_role_count():
    G = nx.path_graph(5)
    rw = RoleWalk()
    min_roles, max_roles = 2, 4
    labels = rw.fit_predict(G, min_n_roles=min_roles, max_n_roles=max_roles)
    n_roles = len(np.unique(labels))
    assert min_roles <= n_roles <= max_roles
