import argparse
import warnings
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from rolewalk import RoleWalk
from datasets import (
    generate_barbell_graph,
    generate_tree_graph,
    generate_ring_of_cliques,
    generate_grid_graph,
    generate_star_graph,
    generate_house_graph,
    load_wikipedia_voting_graph,
)

try:
    from karateclub.node_embedding.structural import GraphWave
except ImportError:  # pragma: no cover
    GraphWave = None  # type: ignore


def evaluate_classification(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return accuracy and macro F1 for a train/test split.

    Stratification is skipped when a class has fewer than two samples to avoid
    noisy warnings from ``train_test_split``.
    """

    class_counts = np.bincount(y)
    stratify: Optional[np.ndarray]
    stratify = y if class_counts.min() >= 2 else None

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
        "ring_of_cliques": generate_ring_of_cliques,
        "grid": generate_grid_graph,
        "star": generate_star_graph,
        "house": generate_house_graph,
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
            m_ap = mean_average_precision(X_rw, labels)
            results.append(
                {
                    "graph": name,
                    "method": "rolewalk",
                    "accuracy": acc,
                    "macro_f1": f1,
                    "map": m_ap,
                }
            )
            if X_gw is not None:
                acc, f1 = evaluate_classification(X_gw, labels)
                m_ap = mean_average_precision(X_gw, labels)
                results.append(
                    {
                        "graph": name,
                        "method": "graphwave",
                        "accuracy": acc,
                        "macro_f1": f1,
                        "map": m_ap,
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
