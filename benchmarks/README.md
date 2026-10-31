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

## Role benchmarks of the literature

`role_benchmark.py` runs the role benchmarks used by struc2vec (Ribeiro et
al., KDD 2017) and the RONE survey (Jiao et al., IEEE TBD 2021), downloaded
once into `data/`:

- classification: air-traffic networks of Brazil (131 airports), Europe
  (399) and the USA (1,190) labelled by activity quartile, and the
  reality-call (6,809), actor (7,758) and film (27,312) networks. Each
  embedding feeds a standardised logistic regression, scored on 50
  stratified 80/20 splits (`roles.csv`);
- retrieval: six Wikipedia talk networks (446 to 7,935 users), where bots
  and admins must be found among each other's 10 nearest neighbours.

```bash
python role_benchmark.py
python role_benchmark.py --datasets brazil europe usa --transfer
```

Baselines: struc2vec (`ge` package, paper parameters; skipped above 3,000
nodes), RESD (2021, the strongest method of RONE) and ReFeX, the input of
RolX and RESD (both re-implemented in `baselines.py` after the RONE code),
GraphWave and Role2Vec (karateclub; GraphWave skipped above 10,000 nodes:
it is dense), RWSE (the return probabilities diag(P^k), k = 1..16, used as
structural encoding by graph transformers such as GPS) and log-degree.
Embeddings are cached in `data/embeddings`.

Accuracy (mean over 50 splits):

| Method | Brazil | Europe | USA | Reality-call | Actor | Film | Time (film) |
|---|---|---|---|---|---|---|---|
| RoleWalk | 0.743 | 0.583 | 0.657 | **0.860** | **0.487** | 0.569 | 7.3 s |
| RoleWalk, `curvature=True` | 0.741 | 0.596 | **0.666** | **0.860** | 0.485 | 0.581 | 7.6 s |
| RoleWalk 1.2 (no onion layers) | 0.749 | 0.584 | 0.628 | **0.860** | 0.482 | 0.549 | 6.4 s |
| struc2vec | 0.681 | **0.598** | 0.590 | – | – | – | (564 s on USA) |
| RESD | 0.728 | 0.544 | 0.653 | 0.857 | 0.476 | 0.563 | 243 s |
| GraphWave | **0.769** | 0.561 | 0.587 | **0.860** | 0.481 | – | (490 s on actor) |
| ReFeX | 0.691 | 0.576 | 0.622 | 0.851 | 0.478 | 0.579 | 0.8 s |
| RWSE | 0.711 | 0.572 | 0.517 | 0.855 | 0.363 | 0.450 | 862 s |
| Role2Vec | 0.476 | 0.366 | 0.562 | 0.844 | 0.399 | **0.743** | 213 s |
| Degree only | 0.756 | 0.554 | 0.567 | 0.858 | 0.476 | 0.478 | 0 s |

Airport and actor labels track activity, so log-degree alone is a strong
baseline there. struc2vec is stochastic: rerunning it moved its Europe score
from 0.56 to 0.60. On Wikipedia, no method dominates (precision@10 averaged
over the networks each method ran on: RoleWalk 0.13 for bots and 0.32 for
admins, RESD 0.11 / 0.25, GraphWave 0.12 / 0.30, struc2vec 0.15 / 0.39 on
the three smallest networks only).

With `--transfer`, the classifier is trained on one airport network and
tested on another (`role_transfer.csv`), which only embeddings with
graph-independent coordinates can attempt. Mean accuracy over the six pairs:
degree 0.542, RoleWalk 0.488 (0.470 with curvature, 0.445 for 1.2), RWSE
0.375, GraphWave 0.249 (chance: its diffusion scales are chosen per graph).
Activity quartiles are almost a function of degree, so this test mostly
rewards degree; the centrality transfer below tests richer structure.

## Directed roles

No public role benchmark is directed. `directed_benchmark.py` hangs directed
motifs (broadcaster and collector stars, cascades in both directions,
feed-forward loops, cycles, mutual stars, bi-fans) from a directed cycle;
several pairs of roles are identical once directions are dropped. Methods
that read directions get the directed graph, the others the symmetrised
one, as does `rolewalk_undirected` (ablation). Accuracy at 0 / 5 / 10% edge
noise (`directed.csv`): RoleWalk with curvature 0.99 / 0.71 / 0.55,
RoleWalk 0.91 / 0.68 / 0.51, RoleWalk without directions 0.77 / 0.56 / 0.42,
best baseline 0.73 (Role2Vec) / 0.47 (degree) / 0.43 (ReFeX).

```bash
python directed_benchmark.py
```

## Structural encoding for GNNs (ZINC)

`pe_benchmark.py` feeds each encoding, batch-normalised and projected, to
the same GINE network (4 layers, ~230k parameters, 300 epochs) on ZINC
(12k molecules) and reports the test MAE at the best validation epoch, over
3 seeds (`pe_zinc_v12.csv` for RoleWalk 1.2, `pe_zinc_v13.csv` for 1.3). Needs
`torch_geometric` and a GPU.

| Encoding | Test MAE |
|---|---|
| none | 0.202 ± 0.010 |
| RWSE (GPS) | 0.107 ± 0.004 |
| RoleWalk | 0.108 ± 0.008 |
| RoleWalk + RWSE | 0.104 ± 0.006 |
| RoleWalk (`curvature=True`) + RWSE | **0.095 ± 0.002** |

This is a relative comparison on a small model: published graph
transformers reach about 0.07 with much longer training.

## Kernel grid

`kernel_grid.py` searches the theta weighting, theta range, walk length and
onion layers that maximise unsupervised roles (k-means ARI) on synthetic
motifs and real networks with equal weight (`kernel_grid.csv`). The current
defaults (theta up to 100, 3 steps) win whichever dataset is left out; what
matters is to scale the onion layers like the characteristic function
(score 0.274 to 0.323), while weighting the frequencies changes little.

```bash
python kernel_grid.py
```

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
