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
- Wikipedia voting network (if available via `karateclub`)

Graphs with known structural role labels are evaluated with a
logistic‑regression classifier, reporting accuracy and macro‑F1 scores.
Graphs without ground‑truth labels are evaluated using the silhouette
score after K‑means clustering.
