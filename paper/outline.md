# RoleWalk — paper outline

Working title: *RoleWalk: fast, inductive structural roles for directed graphs
from the characteristic function of random walks*

Status: outline with current numbers. Intentions in italics; TODO marks
missing experiments. All numbers come from `benchmarks/` (one CPU core
unless stated).

## Abstract

*One paragraph: roles = structural position, not proximity. Existing methods
are either slow (struc2vec), transductive (word2vec-based), or undirected
(GraphWave). RoleWalk: characteristic function of random-walk transition
probabilities + low-order moments of the explored geometry. Inductive,
directed, weighted, 10-1,000x faster, best or tied on most role benchmarks,
best on directed roles and centrality transfer, improves GNN structural
encodings.*

## 1. Introduction

*Claims, in order:*

1. *One embedding, every desirable property (Table 1).*
2. *State-of-the-art or tied accuracy on literature role benchmarks at a
   fraction of the cost (Tables 2-3, Fig. 1).*
3. *Only method that reads directions: directed roles (Table 4).*
4. *Inductive: what is learned on one graph transfers (Tables 5-6).*
5. *Useful beyond role discovery: GNN structural encoding (Table 7).*
6. *Honest benchmarking: degree baselines, dataset diagnostics (film),
   kernel scaling pitfalls (Section 6).*

## 2. Related work

*Short, grouped:*

- *Feature-based roles: RolX / ReFeX, RIDeRs.*
- *Walk + word2vec: struc2vec, Role2Vec, (node2vec as proximity contrast).*
- *Spectral / diffusion: GraphWave (heat-kernel characteristic function).*
- *Deep: DRNE, GAS, RESD (RONE survey).*
- *GNN structural/positional encodings: RWSE (GPS), LapPE, SignNet, GPSE.*
- *Curvature on graphs: Ollivier-Ricci, Forman, Jost-Liu bound.*
- *Directed structural roles: struc2vec++ (Steenfatt et al. 2018, directed
  weighted struc2vec), SDW2vec (Applied Network Science 2025, weighted
  signed directed). TODO: compare if code available.*
- *Directed proximity: magnetic Laplacian (MagNet), NERD — direction-aware
  but not role embeddings.*

## 3. Method

*Figure 2: one node, its lazy walk distributions at steps 1-3, onion layers,
characteristic function samples.*

- 3.1 *Lazy walk H = normalize(I + A); out-walk and in-walk (A^T) for
  directed graphs; weights handled natively.*
- 3.2 *Characteristic function of each row of H^w, aggregated by sum:
  phi(theta) = sum_j (exp(i theta p_ij) - 1); theta log-spaced in [1, 100].
  Squared distance = MMD between transition-probability distributions.*
- 3.3 *Onion layers: probability mass at distance 0..w after w steps, from
  supports of earlier powers (no extra walk). Loops/triangles keep mass
  inside. Equivalent to grouping by a quaternion-like basis per layer.*
- 3.4 *Curvature (option): overlap sum_z min(m_x(z), m_y(z)) of lazy walks
  at edge ends, a probabilistic upper bound of Ollivier-Ricci; mean and std
  where the walk lands.*
- 3.5 *Scaling: moments x ||theta|| = units of the first-order term of the
  CF (Im phi ~ theta * mass). Without it they carry 0.02% of the variance.*
- 3.6 *Complexity: sparse powers with pruning tol = 1e-4; O(walk_len * nnz)
  ; local computation for a subset of nodes => online prediction.*
- 3.7 *Role discovery: Ward + silhouette (auto k), k-means (fixed k);
  persistence of fitted roles (save/load).*

## 4. Properties (theory)

*Draft in `paper/theory.md`: P1 invariance and locality (why it
transfers), P2 distances are an MMD, P3 RWSE is the distance-0 onion layer,
P4 relation to GraphWave and why GraphWave cannot transfer (1/n
normalisation and per-graph scale), P5 strongly regular graphs are a hard
limit, P6 complexity.*

- *Invariance: automorphic (structurally equivalent) nodes get identical
  embeddings.*
- *Relation to RWSE: RWSE = return probabilities = the distance-0 onion
  layer; RoleWalk strictly generalises it.*
- *Relation to GraphWave: same characteristic-function construction on a
  heat kernel; lazy walk instead => sparse, local, directed, inductive
  (no per-graph scale selection).*
- *Expressiveness vs 1-WL: Table 4c.*
- *Stability to edge perturbations (empirical: robustness benchmark).*

## 5. Experiments

### Table 1 — Properties

| | Inductive (transfer) | Directed | Weighted | Fast (USA, 1,190 nodes) | Online (new node) | Deterministic |
|---|---|---|---|---|---|---|
| **RoleWalk** | ✅ | ✅ | ✅ | ✅ 0.2 s | ✅ | ✅ |
| struc2vec | ❌ | ❌ (struc2vec++, 2018: ✅) | ❌ (struc2vec++: ✅) | ❌ 564 s | ❌ | ❌ |
| GraphWave | ❌ (chance in transfer, see P4) | ~ (extension discussed, not implemented) | ✅ | ~ 18 s | ❌ | ✅ |
| Role2Vec | ~ (claimed by the paper; word2vec spaces per run) | ✅ in the paper (karateclub: undirected) | ❌ | ~ 10 s | ~ | ❌ |
| RESD | ❌ | ❌ | ❌ | ~ 8 s | ❌ | ❌ |
| ReFeX | ~ (feature count varies) | ~ | ~ | ✅ 0.1 s | ~ | ✅ |
| RWSE | ✅ (weaker) | ~ | ✅ | ✅ 1.7 s | ✅ | ✅ |

*Checked against the papers for struc2vec (undirected and unweighted;
struc2vec++ by Steenfatt et al., Complex Networks 2018, extends it to
directed weighted graphs), Role2Vec (Ahmed et al., IJCAI 2018: directed
input and inductive by design) and GraphWave. TODO: RESD (directed? no
mechanism for unseen nodes found), ReFeX, RWSE rows.*

### Table 2 — Role classification (accuracy, 50 stratified 80/20 splits)

| Method | Brazil | Europe | USA | Reality-call | Actor | Film* |
|---|---|---|---|---|---|---|
| RoleWalk | 0.743 | 0.583 | 0.657 | **0.860** | **0.487** | 0.569 |
| RoleWalk + curvature | 0.741 | 0.596 | **0.666** | **0.860** | 0.485 | 0.581 |
| struc2vec (mean of 5 runs) | 0.705 ± 0.022 | 0.561 ± 0.020 | 0.596 ± 0.004 | – (too slow) | – | – |
| RESD | 0.728 | 0.544 | 0.653 | 0.857 | 0.476 | 0.563 |
| GraphWave | **0.769** | 0.561 | 0.587 | **0.860** | 0.481 | – (memory) |
| ReFeX | 0.691 | 0.576 | 0.622 | 0.851 | 0.478 | 0.579 |
| Role2Vec | 0.476 | 0.366 | 0.562 | 0.844 | 0.399 | **0.743** |
| RWSE | 0.711 | 0.572 | 0.517 | 0.855 | 0.363 | 0.450 |
| Degree | 0.756 | 0.554 | 0.567 | 0.858 | 0.476 | 0.478 |

*\*Film is a node-type task, see Section 6.4. struc2vec is not seeded: its
runs range 0.53-0.58 on Europe (`struc2vec_runs.py`); std over splits in
appendix. With struc2vec averaged, RoleWalk leads on Europe too.*

### Fig. 1 — Accuracy vs embedding time (log scale), all methods, all datasets

*Pareto front: RoleWalk alone in the top-left corner.*

### Table 3 — Bots and admins retrieval on Wikipedia talk (precision@10)

| Method | Bots | Admins |
|---|---|---|
| RoleWalk | 0.129 | 0.324 |
| struc2vec (3 smallest networks only) | 0.146 | 0.391 |
| GraphWave | 0.115 | 0.298 |
| RESD | 0.114 | 0.254 |
| ReFeX | 0.101 | 0.299 |
| RWSE | 0.118 | 0.297 |
| Degree | 0.075 | 0.296 |

*Mixed results: report honestly, per network in appendix. Networks were
distributed undirected (TODO: rebuild directed versions from SNAP).*

### Table 4 — Directed roles (synthetic, 3 seeds)

| Method | Acc. 0% | Acc. 5% | Acc. 10% | ARI 5% | mAP 5% |
|---|---|---|---|---|---|
| **RoleWalk + curvature** | **0.989** | **0.712** | **0.547** | **0.381** | **0.625** |
| RoleWalk | 0.908 | 0.680 | 0.510 | 0.366 | 0.574 |
| RoleWalk, directions dropped | 0.767 | 0.560 | 0.420 | 0.351 | 0.479 |
| Role2Vec | 0.730 | 0.451 | 0.368 | 0.148 | 0.295 |
| ReFeX | 0.532 | 0.429 | 0.431 | 0.342 | 0.356 |
| GraphWave | 0.586 | 0.388 | 0.383 | 0.317 | 0.416 |
| RESD | 0.547 | 0.420 | 0.423 | 0.253 | 0.348 |
| struc2vec | 0.510 | 0.366 | 0.277 | 0.141 | 0.260 |
| In/out degree | 0.516 | 0.466 | 0.405 | 0.310 | 0.349 |

### Table 4b — Real directed roles: Wikipedia requests for adminship (SNAP wiki-RfA)

*11,377 users, 188k votes (voter -> candidate). Task A: elected or not among
the 3,494 candidates; task B: voter / failed candidate / admin. Methods that
cannot read directions get the undirected graph.*

| Method | Elected? (candidates) | 3 classes | Time |
|---|---|---|---|
| **RoleWalk + curvature** | **0.823** | **0.943** | 5 s |
| RoleWalk | 0.808 | 0.938 | 4 s |
| RoleWalk 1.2 | 0.793 | 0.934 | 4 s |
| RoleWalk, directions dropped | 0.791 | 0.863 | |
| RoleWalk + curvature, directions dropped | 0.814 | 0.900 | |
| RESD | 0.798 | 0.843 | 68 s |
| Role2Vec | 0.798 | 0.857 | 82 s |
| ReFeX | 0.794 | 0.847 | 1 s |
| In/out degree | 0.755 | 0.922 | 0 s |
| RWSE (out-walk) | 0.690 | 0.826 | 51 s |

*struc2vec and GraphWave do not run at this size. TODO: other real directed
sets (directed Wikipedia talk rebuilt from SNAP).*

### Table 4c — Expressiveness (pairs told apart; all sets 1-WL-indistinguishable)

| | Atlas <= 7 nodes (20 pairs) | Random regular (3,181 pairs) | CSL (10 classes) | Decalin / bicyclopentyl | 2 triangles / hexagon | Shrikhande / rook (SRG) |
|---|---|---|---|---|---|---|
| **RoleWalk (3 steps)** | **20** | **3,181** | 6 | yes | yes | no |
| RoleWalk (5 steps) | **20** | **3,181** | **10** | yes | yes | no |
| RoleWalk 1.2 (no layers) | 16 | 3,180 | 6 | yes | yes | no |
| RWSE (16 steps) | **20** | 3,180 | **10** | yes | yes | no |
| Degree (= 1-WL here) | 0 | 0 | 1 | no | no | no |

BREC (400 pairs, Wang & Zhang 2023):

| | Basic (60) | Regular (100) | Extension (100) | CFI (100) | 4-vertex (20) | Distance-reg. (20) | Total |
|---|---|---|---|---|---|---|---|
| RoleWalk (3 steps) | 60 | 50 | 97 | 0 | 0 | 0 | 207 |
| RoleWalk (5 steps) | 60 | 50 | 100 | 3 | 0 | 0 | 213 |
| RoleWalk 1.2 | 58 | 48 | 97 | 0 | 0 | 0 | 203 |
| RWSE (16 steps) | 60 | 50 | 100 | 4 | 0 | 0 | **214** |

*Beyond 1-WL; on par with RWSE with 5 steps instead of 16; CFI,
4-vertex and distance-regular pairs defeat walk statistics (see P5).
No false difference on BREC's isomorphic relabellings.*

### Table 5 — Centrality transfer (train on one graph, test on another of the same family; Spearman)

| Centrality | RoleWalk | + curvature | RoleWalk 1.2 | ReFeX† | RWSE | GraphWave | Degree |
|---|---|---|---|---|---|---|---|
| Betweenness | 0.901 | **0.905** | 0.888 | 0.844 | 0.708 | 0.723 | 0.634 |
| Closeness | **0.866** | 0.864 | 0.833 | 0.684 | 0.732 | 0.597 | 0.428 |
| Clustering | **0.873** | 0.870 | 0.803 | 0.852 | 0.803 | 0.392 | 0.212 |
| k-core | 0.747 | 0.748 | 0.742 | **0.935** | 0.660 | 0.566 | 0.692 |
| Eigenvector | **0.656** | 0.649 | 0.634 | 0.492 | 0.531 | 0.462 | 0.428 |
| Harmonic | **0.918** | 0.916 | 0.893 | 0.782 | 0.784 | 0.699 | 0.500 |
| PageRank | **0.990** | **0.990** | 0.984 | 0.968 | 0.864 | 0.857 | 0.927 |
| Mean (excl. degree) | **0.850** | 0.849 | 0.825 | 0.794 | 0.726 | 0.614 | 0.546 |

*†ReFeX only where its feature count matches across graphs. Graph families:
Barabási-Albert, powerlaw-cluster, Watts-Strogatz, SBM, random geometric,
motifs. Plus Fig. 3: betweenness learned vs source sampling (time /
Spearman), already in README.*

### Table 6 — Role transfer between airport networks (accuracy, 6 pairs)

| Degree | RoleWalk | + curvature | RoleWalk 1.2 | RWSE | GraphWave |
|---|---|---|---|---|---|
| 0.542 | **0.488** | 0.470 | 0.445 | 0.375 | 0.249 (chance) |

*Labels = activity quartiles ≈ degree: degree is near ground truth here;
best among structural embeddings.*

### Table 7 — Structural encoding for GNNs (ZINC, GINE, 3 seeds, test MAE)

| Encoding | Test MAE |
|---|---|
| none | 0.202 ± 0.010 |
| RWSE | 0.107 ± 0.004 |
| RoleWalk | 0.108 ± 0.008 |
| RoleWalk + RWSE | 0.104 ± 0.006 |
| **RoleWalk + curvature + RWSE** | **0.095 ± 0.002** |

### Table 7b — Full GPS protocol (ZINC, GPS 10 layers, ~470k params, 500 epochs, 2 seeds)

| Encoding | Test MAE | Val MAE |
|---|---|---|
| RWSE (published GPS: 0.070 ± 0.004) | 0.077 ± 0.006 | 0.103 ± 0.000 |
| RoleWalk + curvature (no RWSE) | 0.076 ± 0.001 | 0.092 ± 0.003 |
| **RoleWalk + curvature + RWSE** | **0.073 ± 0.001** | **0.089 ± 0.005** |

*RoleWalk alone (3 steps) matches RWSE (16 steps); combined, best and
4x less seed variance. TODO: more seeds (≥4), Peptides (LRGB).*

### Fig. 4 — Scalability: time vs nodes up to 10^6 (TODO), RoleWalk vs baselines that run

## 6. Analysis and ablations

### Table 8 — How to summarise the walk distribution (same budget)

| Summary | Airports | Transfer | Motifs acc. (varied / houses) | Time 20k |
|---|---|---|---|---|
| **Characteristic function** | **0.656** | 0.445 | **0.744** / 0.733 | 6.5 s |
| Laplace transform | 0.657 | 0.345 | 0.614 / 0.717 | 18 s |
| Dyadic histogram (Haar) | 0.631 | 0.353 | 0.614 / 0.725 | 15 s |
| Power sums (Rényi) | 0.653 | 0.420 | 0.647 / 0.692 | 9.7 s |
| Top-10 probabilities | 0.652 | **0.512** | 0.679 / **0.758** | 3.5 s |

### Table 9 — Geometry: moments vs integrated characteristic functions

| Variant | Airports | Transfer | Motifs acc. (varied / houses) | Directed acc. |
|---|---|---|---|---|
| No geometry (1.2) | 0.656 | 0.445 | 0.744 / 0.733 | 0.678 |
| + onion layers (moments) | 0.661 | **0.488** | 0.744 / **0.758** | 0.682 |
| + layers, one CF per layer | 0.654 | 0.424 | 0.751 / 0.742 | 0.695 |
| + degree marks (joint CF) | 0.65-0.67 | 0.34-0.36 | – | – |
| + curvature, counting bound | 0.667 | 0.388 | 0.754 / 0.767 | 0.721 |
| + curvature, probabilistic (moments) | **0.669** | 0.470 | 0.725 / 0.742 | 0.712 |
| + curvature, exact Ollivier-Ricci | 0.662 | **0.494** | 0.746 / 0.750 | **0.743** |
| + curvature, integrated CF | 0.663 | 0.400 | 0.751 / 0.750 | 0.697 |

*Message: the CF carries the role signal; geometry is best added as
low-order moments in probability units (transfer-safe); anything
graph-size dependent (degree, counting bounds) breaks transfer.*

### 6.3 Kernel scaling (Table 10 / Fig. 5)

*In raw embeddings theta = 100 carries 53% of the variance; unscaled moments
are invisible to k-means. Grid (48 configs, synthetic + real, equal weight):
scaling the layers is what matters (unsupervised score 0.274 -> 0.323); the
frequency weighting barely matters; defaults win leave-one-dataset-out.*

### 6.4 Dataset diagnostic: film

| | Class 3 vs rest (node type) | Within classes 0-2 |
|---|---|---|
| Role2Vec | **0.946** | 0.637 |
| RoleWalk + curvature | 0.740 | 0.634 |
| ReFeX | 0.745 | 0.627 |
| Degree | 0.536 | 0.623 |
| Majority class | 0.593 | 0.623 |

*Class 3 is an independent set in a separate id block (node type of a
bipartite-like graph); classes 0-2 are not predictable by any method. Not
homophily: edge homophily 0.28 < 0.33 expected. Role2Vec wins through walk
co-occurrence = position, which a structural (automorphism-invariant)
embedding cannot and should not capture. Recommendation: do not use film
to compare role methods.*

## 7. Limitations

- *On real networks, roles below the most central tier often form a
  continuum: hard clusters overstate them, and the k-means partition of the
  middle tiers is sensitive to feature scaling. Report per-node stability
  (resampled data, method variants) next to the role.*
- *Degree-like labels (airports) favour trivial baselines.*
- *One real directed labelled benchmark (wiki-RfA); more would help.*
- *GPS/ZINC on 2 seeds only; gains are within published seed noise on test, clearer on validation.*

## Appendix

- *Dataset statistics; hyper-parameters of every baseline; per-network
  Wikipedia results; std tables; RESD re-implementation vs reported
  numbers (0.728 vs ~0.79 on Brazil, different protocol).*
