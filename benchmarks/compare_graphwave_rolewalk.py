import argparse
import warnings
from typing import Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from rolewalk import RoleWalk

try:
    from karateclub.node_embedding.structural import GraphWave
except ImportError:  # pragma: no cover
    GraphWave = None  # type: ignore


def generate_barbell_graph(m1: int = 10, m2: int = 5) -> Tuple[nx.Graph, np.ndarray]:
    """Return a barbell graph and role labels.

    Nodes on the path joining the two cliques are assigned label 1 while
    nodes inside the cliques receive label 0.
    """
    G = nx.barbell_graph(m1, m2)
    y = np.zeros(G.number_of_nodes(), dtype=int)
    path_nodes = list(range(m1, m1 + m2))
    y[path_nodes] = 1
    return G, y


def generate_tree_graph(r: int = 2, h: int = 3) -> Tuple[nx.Graph, np.ndarray]:
    """Return a balanced tree and role labels based on node depth."""
    G = nx.balanced_tree(r, h)
    depth = nx.shortest_path_length(G, source=0)
    labels = []
    for v in G.nodes():
        if depth[v] == 0:
            labels.append(0)  # root
        elif G.degree[v] == 1:
            labels.append(2)  # leaves
        else:
            labels.append(1)  # internal nodes
    return G, np.asarray(labels, dtype=int)


def load_wikipedia_voting_graph() -> Tuple[Optional[nx.Graph], Optional[np.ndarray]]:
    """Attempt to load the Wikipedia voting network using karateclub."""
    try:
        from karateclub.dataset import GraphReader
    except Exception as err:  # pragma: no cover
        warnings.warn(f"karateclub is required for the Wikipedia dataset: {err}")
        return None, None

    try:
        reader = GraphReader("wiki-vote")
        G = reader.get_graph()
        return G, None
    except Exception as err:  # pragma: no cover
        warnings.warn(f"Unable to load Wikipedia voting graph: {err}")
        return None, None


def evaluate_classification(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return accuracy and macro F1 for a train/test split.

    The split is stratified unless at least one class has fewer than two
    samples, in which case stratification is skipped and a warning is emitted.
    """

    class_counts = np.bincount(y)
    stratify: Optional[np.ndarray]
    if class_counts.min() < 2:
        warnings.warn(
            "At least one class has fewer than two samples; stratified split "
            "is disabled."
        )
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=stratify, random_state=0
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro", labels=np.unique(y), zero_division=0)
    return acc, f1


def evaluate_clustering(X: np.ndarray, n_clusters: int = 4) -> float:
    labels = KMeans(n_clusters, random_state=0).fit_predict(X)
    return silhouette_score(X, labels)


def main():
    parser = argparse.ArgumentParser(description="Compare RoleWalk and GraphWave")
    parser.add_argument(
        "--output", default="comparison.csv", help="Where to store the CSV summary."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate a comparison bar plot."
    )
    args = parser.parse_args()

    graphs = {
        "barbell": generate_barbell_graph,
        "tree": generate_tree_graph,
        "wiki": load_wikipedia_voting_graph,
    }

    results = []
    for name, loader in graphs.items():
        G, labels = loader()
        if G is None:
            continue

        rw = RoleWalk(walk_len=3, embedding_dim=16)
        X_rw = rw.transform(G)
        if GraphWave is not None:
            gw = GraphWave()
            gw.fit(G)
            X_gw = gw.get_embedding()
        else:  # pragma: no cover
            warnings.warn("GraphWave is unavailable; skipping.")
            X_gw = None

        if labels is not None:
            acc, f1 = evaluate_classification(X_rw, labels)
            results.append(
                {"graph": name, "method": "rolewalk", "accuracy": acc, "macro_f1": f1}
            )
            if X_gw is not None:
                acc, f1 = evaluate_classification(X_gw, labels)
                results.append(
                    {
                        "graph": name,
                        "method": "graphwave",
                        "accuracy": acc,
                        "macro_f1": f1,
                    }
                )
        else:
            score = evaluate_clustering(X_rw)
            results.append({"graph": name, "method": "rolewalk", "silhouette": score})
            if X_gw is not None:
                score = evaluate_clustering(X_gw)
                results.append(
                    {"graph": name, "method": "graphwave", "silhouette": score}
                )

    if not results:
        print("No graphs were evaluated.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(df)

    if args.plot and len(df.columns) > 3:
        import matplotlib.pyplot as plt

        metric_cols = [c for c in df.columns if c not in {"graph", "method"}]
        for metric in metric_cols:
            pivot = df.pivot(index="graph", columns="method", values=metric)
            pivot.plot(kind="bar")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(f"comparison_{metric}.png")
            plt.close()


if __name__ == "__main__":
    main()
