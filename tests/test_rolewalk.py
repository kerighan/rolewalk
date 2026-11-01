import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize

from rolewalk import (
    RoleWalk, compute_embedding, compute_iter, mean_average_precision,
    node_curvature, pairwise_role_distances, role_summary)


def test_transform_shapes():
    walk_len = 2
    n_samples = 5
    n_shells = walk_len * (walk_len + 3) // 2
    for shells, extra in ((False, 0), (True, n_shells)):
        rw = RoleWalk(walk_len=walk_len, n_samples=n_samples, shells=shells)

        # Undirected graph
        G = nx.path_graph(3)
        X = rw.transform(G)
        assert X.shape == (3, 2 * n_samples * walk_len + extra)

        # Directed graph: out- and in-walks
        Gd = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        Xd = rw.transform(Gd)
        assert Xd.shape == (3, 2 * (2 * n_samples * walk_len + extra))


def test_fit_predict_fixed_number_of_roles():
    rw = RoleWalk()
    labels = rw.fit_predict(nx.barbell_graph(8, 4), n_roles=3)
    assert len(np.unique(labels)) == 3 == rw.n_roles_


def test_fit_predict_finds_roles():
    # barbell: clique cores, clique gates, and two depths along the bar
    G = nx.barbell_graph(8, 4)
    rw = RoleWalk()
    # the gates score 0.955 merged with the cores vs 1.0 apart: within the
    # default tolerance, so ask for the best silhouette
    labels = rw.fit_predict(G, tolerance=0)
    assert rw.n_roles_ == 4
    assert labels[0] == labels[15] != labels[7] == labels[12]
    assert labels[8] == labels[11] != labels[9] == labels[10]
    assert rw.n_roles_ in rw.silhouette_scores_

    rw.fit_predict(G, max_n_roles=3)
    assert rw.n_roles_ <= 3
    # nothing to tell apart: a single role
    assert len(np.unique(rw.fit_predict(nx.cycle_graph(10)))) == 1


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


def test_aggregates():
    # star: the hub reaches every leaf, a leaf reaches only the hub
    G = nx.star_graph(4)
    X = RoleWalk(walk_len=1, n_samples=3, embedding_dim=None).transform(G)
    leaves = X[1:]
    np.testing.assert_allclose(leaves, leaves[[0]].repeat(4, 0), atol=1e-6)
    assert not np.allclose(X[0], X[1])

    # "sum" is the full characteristic function: imaginary part ~ theta * sum(p)
    rw = RoleWalk(walk_len=1, n_samples=3, bounds=(1e-3, 1e-2),
                  embedding_dim=None, shells=False)
    X = rw.transform(nx.path_graph(5))
    np.testing.assert_allclose(X[:, 3:], np.tile(rw.theta, (5, 1)), rtol=1e-3)

    try:
        RoleWalk(aggregate="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("no ValueError for unknown aggregate")


def test_compute_iter_chunks_match():
    G = nx.gnp_random_graph(60, 0.1, seed=0)
    H = normalize(nx.to_scipy_sparse_array(G, format="csr"), norm="l1").tocsr()
    theta = RoleWalk().theta
    for aggregate in ("sum", "mean"):
        np.testing.assert_allclose(
            compute_iter(H, theta, aggregate, chunk_size=5),
            compute_iter(H, theta, aggregate), atol=1e-5)


def test_pruning_matches_exact_walks():
    G = nx.barabasi_albert_graph(300, 2, seed=0)
    H = normalize(nx.to_scipy_sparse_array(G, format="csr"), norm="l1").tocsr()
    theta = RoleWalk().theta
    exact = compute_embedding(H, theta, walk_len=4)
    pruned = compute_embedding(H, theta, walk_len=4, tol=1e-3)
    err = np.linalg.norm(pruned - exact, axis=1) / np.linalg.norm(exact, axis=1)
    assert err.max() < 0.05


def test_role_summary():
    G = nx.star_graph(5)
    summary = role_summary(G, [0, 1, 1, 1, 1, 1])
    assert list(summary.index) == [1, 0]  # largest role first
    assert summary.loc[0, "degree"] == 5
    assert summary.loc[1, "links_to"] == "0 (100%)"
    assert "high degree" in summary.loc[0, "profile"]
    assert summary.loc[1, "share"] == 5 / 6

    D = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    summary = role_summary(D, [0, 1, 1], X=np.eye(3))
    assert {"in_degree", "out_degree"} <= set(summary.columns)
    assert summary.loc[0, "out_degree"] == 2


def test_transform_subset_of_nodes():
    for G in (nx.barabasi_albert_graph(200, 2, seed=0),
              nx.gnp_random_graph(100, 0.05, seed=0, directed=True)):
        G = nx.relabel_nodes(G, {v: f"n{v}" for v in G})
        for rw in (RoleWalk(), RoleWalk(curvature=True)):
            X = rw.transform(G)
            nodes = ["n17", "n3", "n42"]
            np.testing.assert_allclose(
                rw.transform(G, nodes), X[[17, 3, 42]], atol=1e-5)


def test_fit_predict_save_load(tmp_path):
    G = nx.barbell_graph(8, 4)
    rw = RoleWalk(role_names=["a", "b", "c", "d"])
    labels = rw.fit_predict(G)
    np.testing.assert_array_equal(rw.predict(G), labels)
    assert rw.fit(G) is rw

    path = tmp_path / "model.npz"
    rw.save(path)
    loaded = RoleWalk.load(path)
    assert loaded.role_names == ["a", "b", "c", "d"]
    assert loaded.n_roles_ == rw.n_roles_
    np.testing.assert_array_equal(loaded.predict(G), labels)
    # roles carry over to another graph with the same structure
    H = nx.barbell_graph(8, 4)
    H = nx.relabel_nodes(H, {v: 100 + v for v in H})
    np.testing.assert_array_equal(
        loaded.predict(H, nodes=[100, 107, 108]), labels[[0, 7, 8]])

    try:
        RoleWalk().predict(G)
    except ValueError:
        pass
    else:
        raise AssertionError("predict before fit did not raise")


def test_shells():
    # path 0-1-2-3: two steps of the lazy walk from 0
    G = nx.path_graph(4)
    H = normalize(nx.to_scipy_sparse_array(G, format="csr")
                  + np.eye(4), norm="l1")
    rw = RoleWalk(walk_len=2, n_samples=3)
    X = rw.transform(G)
    layers = X[:, -3:] / np.linalg.norm(rw.theta)  # step 2: distance 0, 1, 2
    H2 = np.asarray(H @ H)
    np.testing.assert_allclose(layers[0], [H2[0, 0], H2[0, 1], H2[0, 2]],
                               atol=1e-6)
    np.testing.assert_allclose(layers.sum(axis=1), 1, atol=1e-6)
    # a triangle keeps more mass at distance 1 than a path's end
    T = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    X = rw.transform(T)
    assert X[0, -2] > RoleWalk(walk_len=2, n_samples=3).transform(
        nx.path_graph(3))[0, -2]


def test_models_saved_before_shells_load_without(tmp_path):
    import json
    rw = RoleWalk(shells=False)
    rw.save(tmp_path / "model.npz")
    with np.load(tmp_path / "model.npz") as data:
        params = json.loads(str(data["params"]))
        theta = data["theta"]
    del params["shells"]  # as written by RoleWalk 1.2
    np.savez(tmp_path / "old.npz", params=json.dumps(params), theta=theta)
    assert RoleWalk.load(tmp_path / "old.npz").shells is False


def test_node_curvature():
    A = lambda G: nx.to_scipy_sparse_array(G, format="csr")
    # clique: both lazy walks spread uniformly over the same k nodes
    np.testing.assert_allclose(node_curvature(A(nx.complete_graph(5))), 1)
    # star: hub and leaf walks share only the two edge ends
    star = node_curvature(A(nx.star_graph(4)))
    assert star.max() < 0.5
    # curvature is in [0, 1] and only depends on probabilities
    tree = node_curvature(A(nx.balanced_tree(2, 4)))
    assert 0 <= tree.min() and tree.max() <= 1
    X = RoleWalk(walk_len=2, n_samples=3, curvature=True).transform(
        nx.complete_graph(5))
    rw = RoleWalk(walk_len=2, n_samples=3)
    np.testing.assert_allclose(X[:, -2] / np.linalg.norm(rw.theta), 1,
                               atol=1e-6)  # mean at step 2


def test_node_curvature_matches_overlap():
    G = nx.powerlaw_cluster_graph(200, 3, 0.4, seed=0)
    A = nx.to_scipy_sparse_array(G, format="csr")
    H = normalize(np.eye(len(G)) + A.toarray(), norm="l1")
    brute = np.zeros(len(G))
    for x, y in G.edges():
        overlap = np.minimum(H[x], H[y]).sum()
        brute[x] += overlap
        brute[y] += overlap
    brute /= np.array([G.degree(v) for v in G])
    np.testing.assert_allclose(node_curvature(A), brute, atol=1e-12)
    np.testing.assert_allclose(node_curvature(A, [3, 7])[[3, 7]], brute[[3, 7]], atol=1e-12)


def test_blocks_do_not_change_embeddings():
    G = nx.barabasi_albert_graph(500, 3, seed=0)
    X = RoleWalk(curvature=True).transform(G)
    Y = RoleWalk(curvature=True, block_work=200).transform(G)
    np.testing.assert_array_equal(X, Y)


def test_max_missing_keeps_hub_rows():
    # at step 2 every leaf of a large star is below tol, but half the mass
    # returns to the centre at step 3
    G = nx.star_graph(20000)
    exact = RoleWalk(tol=0).transform(G, nodes=[0])
    pruned = RoleWalk(tol=1e-4).transform(G, nodes=[0])
    kept = RoleWalk(tol=1e-4, max_missing=0.05).transform(G, nodes=[0])
    assert np.linalg.norm(pruned - exact) > 0.1 * np.linalg.norm(exact)
    np.testing.assert_allclose(kept, exact, rtol=1e-4, atol=1e-4)


def test_layers_are_nonnegative_with_pruning():
    G = nx.barabasi_albert_graph(3000, 2, seed=1)
    model = RoleWalk(walk_len=3)
    X = model.transform(G)
    m = model.theta.shape[1]
    start, layers = 0, []
    for w in range(1, 4):
        start += 2 * m
        layers.append(X[:, start:start + w + 1])
        start += w + 1
    assert min(block.min() for block in layers) >= -1e-6


def test_mixed_separates_shared_targets():
    one = nx.DiGraph([(0, 1), (0, 2)])
    k22 = nx.DiGraph([(0, 1), (0, 2), (3, 1), (3, 2)])
    for curvature in (False, True):
        model = RoleWalk(tol=0, curvature=curvature)
        np.testing.assert_allclose(model.transform(one)[0], model.transform(k22)[0], atol=1e-6)
    model = RoleWalk(tol=0, mixed=True)
    X, Y = model.transform(one), model.transform(k22)
    assert X.shape[1] == RoleWalk(tol=0).transform(one).shape[1] + 4 * 10
    assert np.abs(X[0] - Y[0]).max() > 1e-2
    # undirected graphs are unchanged
    G = nx.karate_club_graph()
    np.testing.assert_array_equal(RoleWalk(mixed=True).transform(G), RoleWalk().transform(G))


def test_fused_kernel_matches_sparse_powers():
    import rolewalk
    if rolewalk.numba is None:
        return
    G = nx.powerlaw_cluster_graph(1500, 3, 0.3, seed=2)
    D = nx.gnp_random_graph(300, 0.03, seed=3, directed=True)
    for graph in (G, D):
        for kw in (dict(tol=0), dict(tol=1e-4), dict(tol=1e-3, max_missing=0.05),
                   dict(curvature=True), dict(shells=False)):
            fused = RoleWalk(**kw).transform(graph)
            saved, rolewalk.numba = rolewalk.numba, None
            try:
                sparse = RoleWalk(**kw).transform(graph)
            finally:
                rolewalk.numba = saved
            np.testing.assert_allclose(fused, sparse, rtol=1e-4,
                                       atol=1e-5 * np.abs(sparse).max())
    X = RoleWalk(n_jobs=4).transform(G)
    np.testing.assert_allclose(X, RoleWalk().transform(G), rtol=1e-6, atol=1e-6)


def test_negative_weights_rejected():
    G = nx.Graph()
    G.add_edge(0, 1, weight=-1.0)
    G.add_edge(1, 2, weight=1.0)
    try:
        RoleWalk().transform(G)
    except ValueError:
        return
    raise AssertionError("negative weights must be rejected")


def test_triangle_curvature_linear_on_stars():
    import time
    A = nx.to_scipy_sparse_array(nx.star_graph(200000), format="csr")
    start = time.perf_counter()
    kappa = node_curvature(A)
    assert time.perf_counter() - start < 5
    np.testing.assert_allclose(kappa[1:], 2 / 200001)
