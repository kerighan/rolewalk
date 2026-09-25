"""How should RoleWalk sample the characteristic function?

The squared distance between two RoleWalk embeddings is, per walk step,
``n * MMD^2`` between the nodes' transition-probability distributions under
the kernel ``K(d) = mean_theta cos(theta * d)``. Choosing theta means choosing
that kernel. This script compares sampling schemes and bandwidths:

- ``linspace``: current default, uniform on [~0, T] -> periodic Dirichlet kernel
- ``geomspace``: log-spaced on [T/100, T] -> multi-scale kernel
- ``gauss``: half-normal quantiles with scale T/2 -> Gaussian (RBF) kernel

on shapes-on-a-cycle graphs whose motifs are scaled up (``--scale``) to move
the typical degree, and therefore the p-differences between roles (~1/d^2).

    python theta_study.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import norm

import datasets
from rolewalk import RoleWalk
from synthetic_benchmark import add_noise, score


def thetas(scheme: str, T: float, n: int) -> np.ndarray:
    if scheme == "linspace":
        return np.linspace(1e-3, T, n)
    if scheme == "geomspace":
        return np.geomspace(T / 100, T, n)
    if scheme == "gauss":
        return norm.ppf(0.5 + 0.5 * (np.arange(n) + 0.5) / n) * T / 2
    raise ValueError(scheme)


def embed(G, theta: np.ndarray, walk_len: int = 3) -> np.ndarray:
    rw = RoleWalk(walk_len=walk_len, n_samples=len(theta), embedding_dim=None)
    rw.theta = theta[None, :].astype(np.float32)
    return rw.transform(G)


def scaled_graph(scale: int, seed: int):
    """``varied`` graph with motif sizes multiplied by ``scale``."""
    shapes = {
        "house": datasets._shape_house,
        "star": partial(datasets._shape_star, 5 * scale),
        "clique": partial(datasets._shape_clique, 5 * scale),
        "fan": partial(datasets._shape_fan, 6 * scale),
        "diamond": datasets._shape_diamond,
        "tree": partial(datasets._shape_tree, 2 * scale),
    }
    saved = dict(datasets.SHAPES)
    datasets.SHAPES.update(shapes)
    try:
        return datasets.generate_varied(n_per_shape=5, seed=seed)
    finally:
        datasets.SHAPES.clear()
        datasets.SHAPES.update(saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schemes", nargs="*", default=["linspace", "geomspace", "gauss"])
    parser.add_argument("--T", type=float, nargs="*",
                        default=[5, 10, 25, 50, 100, 200, 400, 800])
    parser.add_argument("--scales", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.02, 0.05])
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="theta_study.csv")
    args = parser.parse_args()

    rows = []
    for scale in args.scales:
        for seed in range(args.seeds):
            G, y = scaled_graph(scale, seed)
            mean_degree = 2 * G.number_of_edges() / G.number_of_nodes()
            for level in args.noise:
                H = add_noise(G, level, np.random.default_rng(seed))
                for scheme in args.schemes:
                    for T in args.T:
                        X = embed(H, thetas(scheme, T, args.n_samples))
                        rows.append({
                            "scale": scale, "n_nodes": G.number_of_nodes(),
                            "mean_degree": mean_degree, "seed": seed,
                            "noise": level, "scheme": scheme, "T": T,
                            **score(X, y, seed),
                        })
        print(f"scale={scale} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", None):
        for metric in ["map", "ari", "accuracy"]:
            print(f"== {metric}")
            print(df.groupby(["scale", "noise", "scheme", "T"])[metric]
                  .mean().unstack("T").round(3))


if __name__ == "__main__":
    main()
