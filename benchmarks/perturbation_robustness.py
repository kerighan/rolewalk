import argparse
import warnings
from typing import Optional, Tuple, List

import networkx as nx
import numpy as np
import pandas as pd
from rolewalk import RoleWalk
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

try:  # Optional dependency used for comparison
    from karateclub.node_embedding.structural import GraphWave
except Exception:  # pragma: no cover
    GraphWave = None  # type: ignore

# Graph generators shared with the comparison script
from datasets import (
    generate_barbell_graph,
    generate_tree_graph,
    load_wikipedia_voting_graph,
)

# Evaluation helpers duplicated from the comparison script

def evaluate_classification(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return accuracy and macro F1 for a train/test split.

    Stratification is skipped when a class has fewer than two samples to
    avoid noisy warnings from ``train_test_split``.
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
    """Return silhouette score after k-means clustering."""
    labels = KMeans(n_clusters, random_state=0).fit_predict(X)
    return silhouette_score(X, labels)


def perturb_graph(
    G: nx.Graph, add_rate: float, remove_rate: float, rng: np.random.Generator
) -> nx.Graph:
    """Return a perturbed copy of ``G`` with edges added and removed.

    Parameters
    ----------
    G : nx.Graph
        The original graph.
    add_rate : float
        Fraction of existing edges to add as new random edges.
    remove_rate : float
        Fraction of existing edges to remove.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    H = G.copy()
    edges = list(H.edges())
    n_edges = len(edges)

    # Remove edges
    n_remove = int(remove_rate * n_edges)
    if n_remove > 0:
        remove_idx = rng.choice(len(edges), size=n_remove, replace=False)
        for idx in remove_idx:
            H.remove_edge(*edges[idx])

    # Add edges
    n_add = int(add_rate * n_edges)
    if n_add > 0:
        non_edges = list(nx.non_edges(H))
        if n_add > len(non_edges):
            n_add = len(non_edges)
        add_idx = rng.choice(len(non_edges), size=n_add, replace=False)
        for idx in add_idx:
            u, v = non_edges[idx]
            H.add_edge(u, v)

    return H


def evaluate_graph(
    G: nx.Graph,
    labels: Optional[np.ndarray],
    perturb_levels: List[float],
    n_variants: int,
) -> pd.DataFrame:
    """Compute metrics for perturbed versions of ``G`` for each embedding method."""

    rng = np.random.default_rng(0)
    methods = {
        "rolewalk": lambda H: RoleWalk(walk_len=3, embedding_dim=16, random_state=0).transform(H)
    }
    if GraphWave is not None:
        methods["graphwave"] = lambda H: GraphWave().fit(H).get_embedding()
    else:  # pragma: no cover
        warnings.warn("GraphWave is unavailable; skipping GraphWave comparison.")

    rows: List[dict] = []
    for method_name, embed_fn in methods.items():
        X_base = embed_fn(G)
        if labels is not None:
            base_acc, base_f1 = evaluate_classification(X_base, labels)
        else:
            base_sil = evaluate_clustering(X_base)

        for level in perturb_levels:
            if level == 0:
                if labels is not None:
                    rows.append(
                        {
                            "method": method_name,
                            "perturbation": 0.0,
                            "accuracy": base_acc,
                            "macro_f1": base_f1,
                            "degradation_acc": 0.0,
                            "degradation_f1": 0.0,
                        }
                    )
                else:
                    rows.append(
                        {
                            "method": method_name,
                            "perturbation": 0.0,
                            "silhouette": base_sil,
                            "degradation_silhouette": 0.0,
                        }
                    )
                continue

            metrics = []
            for _ in range(n_variants):
                H = perturb_graph(G, level, level, rng)
                X = embed_fn(H)
                if labels is not None:
                    acc, f1 = evaluate_classification(X, labels)
                    metrics.append((acc, f1))
                else:
                    sil = evaluate_clustering(X)
                    metrics.append((sil,))

            if labels is not None:
                accs, f1s = zip(*metrics)
                mean_acc = float(np.mean(accs))
                mean_f1 = float(np.mean(f1s))
                rows.append(
                    {
                        "method": method_name,
                        "perturbation": level,
                        "accuracy": mean_acc,
                        "macro_f1": mean_f1,
                        "degradation_acc": base_acc - mean_acc,
                        "degradation_f1": base_f1 - mean_f1,
                    }
                )
            else:
                sils = [m[0] for m in metrics]
                mean_sil = float(np.mean(sils))
                rows.append(
                    {
                        "method": method_name,
                        "perturbation": level,
                        "silhouette": mean_sil,
                        "degradation_silhouette": base_sil - mean_sil,
                    }
                )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding robustness to edge perturbations"
    )
    parser.add_argument(
        "--output", default="robustness.csv", help="Where to store the CSV summary."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate metric vs. perturbation plots."
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.1, 0.15, 0.2],
        help="Perturbation levels as fractions of edges added/removed",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=5,
        help="Number of random variants to generate per perturbation level",
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
        df_graph = evaluate_graph(G, labels, args.levels, args.variants)
        df_graph.insert(0, "graph", name)
        results.append(df_graph)

    if not results:
        print("No graphs were evaluated.")
        return

    df = pd.concat(results, ignore_index=True)
    df.to_csv(args.output, index=False)
    print(df)

    if args.plot:
        import matplotlib.pyplot as plt

        for (name, method), df_graph in df.groupby(["graph", "method"]):
            metric_cols = [
                c for c in df_graph.columns if c not in {"graph", "method", "perturbation"}
            ]
            metrics = [c for c in metric_cols if not c.startswith("degradation")]
            for metric in metrics:
                plt.figure()
                plt.plot(df_graph["perturbation"], df_graph[metric], marker="o")
                plt.xlabel("Perturbation level")
                plt.ylabel(metric)
                plt.title(f"{name} graph - {method}")
                plt.tight_layout()
                plt.savefig(f"robustness_{name}_{method}_{metric}.png")
                plt.close()


if __name__ == "__main__":
    main()
