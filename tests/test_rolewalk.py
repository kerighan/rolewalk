import numpy as np
import networkx as nx

from rolewalk import RoleWalk, mean_average_precision, pairwise_role_distances


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


def test_pairwise_role_distances_and_map():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    labels = np.array([0, 0, 0, 1])

    dists, ranking = pairwise_role_distances(X)
    assert dists.shape == (4, 4)
    assert ranking.shape == (4, 3)
    # node 0 is closest to node 1
    assert ranking[0, 0] == 1

    m_ap = mean_average_precision(X, labels)
    assert 0.7 < m_ap < 0.8  # expected value is 0.75


def test_fit_predict_does_not_mutate_input():
    X = RoleWalk().transform(nx.path_graph(6))
    X_copy = X.copy()
    RoleWalk().fit_predict(X, max_n_roles=3)
    np.testing.assert_array_equal(X, X_copy)


def test_fit_predict_rejects_unknown_options():
    X = RoleWalk().transform(nx.path_graph(6))
    for kwargs in ({"metric": "nope"}, {"method": "nope"}):
        try:
            RoleWalk().fit_predict(X, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kwargs}")


def test_theta_schemes():
    geo = RoleWalk(n_samples=5)
    assert geo.theta_scheme == "geomspace"
    np.testing.assert_allclose(geo.theta[0], np.geomspace(1, 100, 5), rtol=1e-6)

    lin = RoleWalk(n_samples=5, theta_scheme="linspace")
    np.testing.assert_allclose(lin.theta[0], np.linspace(1e-3, 100, 5), rtol=1e-6)

    custom = RoleWalk(n_samples=3, bounds=(2, 8))
    np.testing.assert_allclose(custom.theta[0], [2, 4, 8], rtol=1e-6)

    try:
        RoleWalk(theta_scheme="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("no ValueError for unknown theta_scheme")
