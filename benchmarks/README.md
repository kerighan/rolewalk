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

Methods: `rolewalk` (current defaults), `rolewalk_w5` (`walk_len=5`),
`rolewalk_mean` (1.1 defaults, `aggregate="mean"`), `rolewalk_linspace`
(1.0 defaults) and `graphwave`. The GraphWave rows of the committed CSVs come
from an earlier run on the same graphs and seeds.

`mean_average_precision` breaks ties between equal distances arbitrarily, so
when nodes with different labels have identical embeddings (no noise,
`varied`), mAP can move by about 0.02 from float32 rounding alone.

Per-seed results are written to `synthetic.csv` (or `--output`) and a
mean ± std summary is printed. The script pins BLAS to one thread: with many
small solves, multithreading was ~80x slower on a 20-core machine.

GraphWave needs a working `karateclub`, which in recent environments means
`gensim>=4.3.3` (scipy 1.13 removed `scipy.linalg.triu`) and `pygsp==0.5.1`
(newer pygsp renamed the heat kernel's `tau` argument).

## Clustering

`clustering_benchmark.py` compares clustering algorithms on RoleWalk
embeddings of the synthetic graphs, with the true number of roles given
(`fixed`) or to be found (`auto`), and writes `clustering.csv`. See the main
README for the results and the method `RoleWalk.fit_predict` uses.

```bash
python clustering_benchmark.py
```

## Centralities

`centrality_benchmark.py` predicts centralities (degree, PageRank,
betweenness, closeness, harmonic, eigenvector, clustering, k-core) from
RoleWalk embeddings with gradient boosting, on held-out nodes
(`centrality.csv`) or on another graph of the same family (`--transfer`,
`centrality_transfer.csv`). See the main README for the results.

```bash
python centrality_benchmark.py
python centrality_benchmark.py --transfer
```

## Betweenness: learning vs sampling

`betweenness_benchmark.py` trains a regressor on a 2,000-node graph, then
compares its predictions on a new 5,000-node graph with source-sampling
estimates (`k` = 10 to 1,000 sources), both against the exact betweenness
(time and Spearman correlation). Results go to `betweenness.csv`.

```bash
python betweenness_benchmark.py
```

## Plots

`plot_benchmarks.py` draws the README's charts from the CSVs above, in
light and dark versions, into `docs/img/`.

```bash
python plot_benchmarks.py
```
