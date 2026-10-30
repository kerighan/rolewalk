"""Which clustering turns RoleWalk embeddings into roles best?

Two settings, on the synthetic shapes graphs of ``synthetic_benchmark.py``:

- ``fixed``: the true number of roles is given;
- ``auto``: the number of roles must be found from the data.

Scores are the adjusted Rand index and adjusted mutual information against
the true roles, plus the number of roles found and the time taken.

    python clustering_benchmark.py
    python clustering_benchmark.py --settings auto --seeds 5
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
import warnings
from typing import Callable, Dict

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import (
    HDBSCAN,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

from rolewalk import RoleWalk
from synthetic_benchmark import DATASETS, add_noise

warnings.filterwarnings("ignore")
MAX_K = 40


def gmm(k: int) -> GaussianMixture:
    return GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, random_state=0)


FIXED: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "rolewalk (kmeans)": lambda X, k: RoleWalk().fit_predict(X, n_roles=k),
    "kmeans n_init=3": lambda X, k: KMeans(k, n_init=3, random_state=0).fit_predict(X),
    "minibatch kmeans": lambda X, k: MiniBatchKMeans(k, n_init=3, random_state=0).fit_predict(X),
    "ward": lambda X, k: AgglomerativeClustering(k).fit_predict(X),
    "average linkage": lambda X, k: AgglomerativeClustering(k, linkage="average").fit_predict(X),
    "gmm diag": lambda X, k: gmm(k).fit_predict(X.astype(float)),
    "birch": lambda X, k: Birch(n_clusters=k, threshold=0.05).fit_predict(X),
    "spectral knn": lambda X, k: SpectralClustering(
        k, affinity="nearest_neighbors", n_neighbors=10, random_state=0).fit_predict(X),
}


def kmeans_silhouette(X: np.ndarray) -> np.ndarray:
    best, best_labels = -1.0, None
    for k in range(2, MAX_K + 1):
        labels = KMeans(k, n_init=3, random_state=0).fit_predict(X)
        score = silhouette_score(X, labels, sample_size=1000, random_state=0)
        if score > best:
            best, best_labels = score, labels
    return best_labels


def gmm_bic(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    models = [gmm(k).fit(X) for k in range(1, MAX_K + 1)]
    return min(models, key=lambda m: m.bic(X)).predict(X)


def spectral_eigengap(X: np.ndarray) -> np.ndarray:
    A = kneighbors_graph(X, 10)
    A = 0.5 * (A + A.T)
    eigvals = np.sort(eigsh(laplacian(A, normed=True), k=MAX_K + 1,
                            sigma=-1e-3, which="LM")[0])
    k = int(np.argmax(np.diff(eigvals[1:])) + 2)
    return SpectralClustering(
        k, affinity="precomputed_nearest_neighbors", random_state=0).fit_predict(A)


AUTO: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "rolewalk (ward+silhouette, tolerance=0.05)": lambda X: RoleWalk().fit_predict(X),
    "rolewalk tolerance=0": lambda X: RoleWalk().fit_predict(X, tolerance=0),
    "rolewalk tolerance=0.1": lambda X: RoleWalk().fit_predict(X, tolerance=0.1),
    "kmeans + silhouette": kmeans_silhouette,
    "gmm + bic": gmm_bic,
    "hdbscan": lambda X: HDBSCAN(min_cluster_size=5).fit_predict(X),
    "spectral + eigengap": spectral_eigengap,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", nargs="*", default=["fixed", "auto"],
                        choices=["fixed", "auto"])
    parser.add_argument("--datasets", nargs="*", default=sorted(DATASETS),
                        choices=sorted(DATASETS))
    parser.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default="clustering.csv")
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        for seed in range(args.seeds):
            G, y = DATASETS[dataset](seed)
            n_roles = len(np.unique(y))
            for level in args.noise:
                H = add_noise(G, level, np.random.default_rng(seed))
                X = RoleWalk().transform(H)
                for setting in args.settings:
                    methods = FIXED if setting == "fixed" else AUTO
                    for method, cluster in methods.items():
                        start = time.perf_counter()
                        if setting == "fixed":
                            labels = cluster(X, n_roles)
                        else:
                            labels = cluster(X)
                        rows.append({
                            "setting": setting, "dataset": dataset,
                            "seed": seed, "noise": level, "method": method,
                            "time": time.perf_counter() - start,
                            "ari": adjusted_rand_score(y, labels),
                            "ami": adjusted_mutual_info_score(y, labels),
                            "n_roles": len(np.unique(labels)),
                            "true_n_roles": n_roles,
                        })
                print(f"{dataset} seed={seed} noise={level} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        for setting, part in df.groupby("setting"):
            print(f"\n== {setting} ==")
            print(part.groupby("method")[["ari", "ami", "n_roles", "time"]]
                  .mean().sort_values("ari", ascending=False).round(3))


if __name__ == "__main__":
    main()
