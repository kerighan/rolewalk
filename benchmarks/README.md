# Benchmarks

This directory provides a simple script to compare node role embeddings
produced by [RoleWalk](../rolewalk.py) against embeddings from the
GraphWave library.

## Setup

The script relies on a few scientific Python packages.  The easiest way
to install them is with `pip`:

```bash
pip install -r requirements.txt
```

Required packages:

- [`networkx`](https://networkx.org/)
- [`numpy`](https://numpy.org/)
- [`pandas`](https://pandas.pydata.org/)
- [`scikit-learn`](https://scikit-learn.org/)
- [`matplotlib`](https://matplotlib.org/) (for optional plots)
- [`karateclub`](https://karateclub.readthedocs.io/) (provides the GraphWave implementation)

## Usage

Run the benchmark script to generate embeddings and compare performance
on several graphs:

```bash
python compare_graphwave_rolewalk.py
```

This prints a summary table to the console and writes the results to
`comparison.csv`.  Use the `--plot` flag to create bar plots comparing
the methods:

```bash
python compare_graphwave_rolewalk.py --plot
```

The script currently evaluates the following graphs:

- Barbell graph
- Balanced tree
- Ring of cliques
- Grid graph
- Star graph
- House graph
- Wikipedia voting network (if available via `karateclub`)

Graphs with known structural role labels are evaluated with a
logistic‑regression classifier, reporting accuracy and macro‑F1 scores.
In addition, nodes are ranked by structural similarity to compute
retrieval metrics such as mean average precision (mAP).
Graphs without ground‑truth labels are evaluated using the silhouette
score after K‑means clustering.

## Robustness to Edge Perturbations

The `perturbation_robustness.py` script assesses how both RoleWalk and
GraphWave embeddings degrade when random edges are added or removed. It
generates several perturbed versions of each graph, recomputes embeddings
and metrics, and summarizes the results. Enable the `--plot` flag to
visualize accuracy or silhouette score versus perturbation level for each
method.

```bash
python perturbation_robustness.py --plot
```

Results are written to `robustness.csv` and plots are saved as
`robustness_<graph>_<method>_<metric>.png`.

## Synthetic shapes benchmark

`synthetic_benchmark.py` reproduces the setting of the GraphWave paper
(Donnat et al., KDD 2018): motifs (house, star, clique, fan, diamond, tree)
are attached to a base cycle, and each node's ground-truth role is its
position inside its motif. Edges are then randomly removed and added
(`--noise`, as a fraction of the edge count); every method sees the same
noisy graph, and results are averaged over `--seeds` random graphs.

Metrics: k-means homogeneity, completeness and ARI (as in the paper), plus
logistic-regression accuracy / macro-F1 on a stratified 50/50 split, mAP,
and embedding time.

```bash
python synthetic_benchmark.py                                    # houses (80 nodes), varied (275 nodes)
python synthetic_benchmark.py --datasets houses_large varied_large --seeds 3   # 800 and 2750 nodes
```

Per-seed results are written to `synthetic.csv` (or `--output`) and a
mean ± std summary is printed. The script pins BLAS to one thread: with many
small solves, multithreading was ~80x slower on a 20-core machine.

GraphWave needs a working `karateclub`, which in recent environments means
`gensim>=4.3.3` (scipy 1.13 removed `scipy.linalg.triu`) and `pygsp==0.5.1`
(newer pygsp renamed the heat kernel's `tau` argument).
