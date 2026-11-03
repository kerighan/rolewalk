"""Illustrations of the human paper: the method and the roles it finds.

- ``fig_method.pdf``: a three-step lazy walk from one node of Zachary's karate
  club, its onion layers, the two laws RoleWalk reads at each step and the
  resulting coordinates;
- ``fig_roles.pdf``: houses attached to a cycle (the GraphWave setting),
  coloured by k-means on RoleWalk embeddings, next to a PCA of the embeddings
  coloured by the true role.

Both are written to ../paper/human/.

    python illustration_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from datasets import generate_shapes_graph  # noqa: E402
from rolewalk import RoleWalk  # noqa: E402

OUT = os.path.join("..", "paper", "human")
INK, MUTED, EDGE, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
# onion layers: one blue ramp, darkest at the root
LAYERS = ["#0d366b", "#2a78d6", "#6da7ec", "#b7d3f6"]
CF = plt.get_cmap("Oranges")
ROLES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": EDGE,
                     "text.color": INK, "axes.labelcolor": MUTED})


def lazy_walk(G):
    A = nx.to_numpy_array(G, nodelist=range(len(G)))
    d = (A > 0).sum(1)
    H = A / A.sum(1, keepdims=True) * (d / (d + 1))[:, None]
    H[np.diag_indices_from(H)] = 1 / (d + 1)
    return H


def radial_layout(G, root, rmax=4):
    """Nodes on circles by distance to ``root``, evenly spread on each circle,
    ordered by the mean angle of their neighbours on the previous circle."""
    dist = nx.single_source_shortest_path_length(G, root)
    angle, pos = {root: 0.0}, {root: np.zeros(2)}
    for k in range(1, rmax + 1):
        layer = [v for v in G if min(dist[v], rmax) == k]
        def key(v):
            prev = [angle[x] for x in G[v] if x in angle]
            return np.angle(np.exp(1j * np.array(prev)).mean()) if prev else 0.0
        if not layer:
            break
        layer.sort(key=key)
        shift = key(layer[0]) if k > 1 else np.pi / 2
        for i, v in enumerate(layer):
            angle[v] = shift + 2 * np.pi * i / len(layer)
            pos[v] = k * np.array([np.cos(angle[v]), np.sin(angle[v])])
    return pos, dist


def method_figure(L=3, root=3):
    G = nx.karate_club_graph()
    H = lazy_walk(G)
    pos, dist = radial_layout(G, root)
    theta = np.geomspace(1, 100, 10)
    norm = np.sqrt((theta ** 2).sum())
    P = np.linalg.matrix_power(H, L)[root]

    fig = plt.figure(figsize=(6.8, 2.75))
    # (a) the graph and its onion layers
    ax = fig.add_axes([0.0, 0.02, 0.33, 0.92])
    for k in range(L, -1, -1):
        ax.add_patch(Circle((0, 0), k + 0.5, color=LAYERS[k], alpha=0.10, lw=0, zorder=0))
    for k in range(1, L + 1):
        ax.add_patch(Circle((0, 0), k + 0.5, fill=False, ec=LAYERS[k], lw=0.6, alpha=0.6, zorder=0))
    # the walk only sees the ball of radius L: draw it alone
    ball = [v for v in G if dist[v] <= L]
    for u, v in G.subgraph(ball).edges():
        # edges that step outwards carry the onion; edges within a layer recede
        outward = dist[u] != dist[v]
        ax.plot(*zip(pos[u], pos[v]), color="#a9a79e" if outward else "#e6e5df",
                lw=0.7 if outward else 0.45, zorder=1.5 if outward else 1)
    for v in ball:
        size = 14 + 380 * np.sqrt(P[v])
        ax.scatter(*pos[v], s=size, color=LAYERS[dist[v]], ec="#ffffff", lw=0.8, zorder=3)
    ax.annotate("$u$", pos[root], xytext=(0, 0), textcoords="offset points", ha="center",
                va="center", color="#ffffff", fontsize=8, zorder=4)
    for k in range(1, L + 1):
        a = np.deg2rad(-118)
        ax.text((k + 0.5) * np.cos(a), (k + 0.5) * np.sin(a), f"$d={k}$", ha="center",
                va="center", fontsize=6.5, color=MUTED,
                bbox=dict(fc="#ffffff", ec="none", pad=0.4, alpha=0.85))
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-3.7, 3.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) 3-ball of $u$, onion layers; size $= p^3_{uj}$", fontsize=8, color=INK)

    # (b) per step: the multiset of probabilities and the law of the distance,
    # (c) the coordinates they give
    cv = fig.add_axes([0.35, 0.02, 0.65, 0.92])
    cv.set_xlim(0, 100)
    cv.set_ylim(0, 40)
    cv.axis("off")
    cv.text(12, 38.5, "(b) probabilities $p^w_{uj}$", ha="center", fontsize=8)
    cv.text(41, 38.5, "(b') law of $d(u, X_w)$", ha="center", fontsize=8)
    cv.text(78, 38.5, "(c) coordinates of step $w$", ha="center", fontsize=8)
    Hw = np.eye(len(G))
    pmax = max(np.linalg.matrix_power(H, w)[root].max() for w in range(1, L + 1))
    for w in range(1, L + 1):
        Hw = Hw @ H
        p = Hw[root]
        y0 = 36 - w * 11.5
        cv.text(-1.5, y0 + 4, f"$w={w}$", ha="right", va="center", fontsize=8)
        # sorted probabilities, coloured by layer
        order = np.argsort(-p)
        order = order[p[order] > 0]
        bw = 22 / 20
        for i, j in enumerate(order[:20]):
            h = 8.5 * p[j] / pmax
            cv.add_patch(Rectangle((1 + i * bw, y0), bw * 0.8, max(h, 0.15),
                                   color=LAYERS[min(dist[j], L)], lw=0))
        cv.plot([1, 23.5], [y0, y0], color=EDGE, lw=0.5)
        # law of the distance as one stacked bar
        s = np.array([p[[j for j in G if dist[j] == k]].sum() for k in range(w + 1)])
        x = 31
        for k in range(w + 1):
            cv.add_patch(Rectangle((x, y0 + 2.5), 20 * s[k], 3.5, color=LAYERS[k], lw=0))
            x += 20 * s[k]
        # coordinates: Re phi, Im phi (orange by magnitude), layers (blue by layer)
        phi = (np.exp(1j * np.outer(theta, p[p > 0])) - 1).sum(1)
        x0, sq = 56.5, 1.55
        for b, vals in enumerate([phi.real, phi.imag]):
            m = np.abs(vals) / max(np.abs(vals).max(), 1e-9)
            for i, v in enumerate(m):
                cv.add_patch(Rectangle((x0 + (b * 10 + i) * sq, y0 + 2.5), sq * 0.85, 3.5,
                                       color=CF(0.15 + 0.75 * v), lw=0))
        xl = x0 + 20 * sq + 1.2
        for k in range(w + 1):  # bar height = mass of layer k
            cv.add_patch(Rectangle((xl + k * sq, y0 + 2.5), sq * 0.85, 3.5, fill=False,
                                   ec=GRID, lw=0.5))
            cv.add_patch(Rectangle((xl + k * sq, y0 + 2.5), sq * 0.85, 3.5 * s[k],
                                   color=LAYERS[k], lw=0))
        if w == 1:
            for xc, lab in [(x0 + 5 * sq, r"Re $\phi^w_u(\theta)$"),
                            (x0 + 15 * sq, r"Im $\phi^w_u(\theta)$"),
                            (xl + 1.5 * sq, r"$\|\Theta\|\,s^w_u$")]:
                cv.text(xc, y0 + 7.3, lab, ha="center", fontsize=6.5, color=MUTED)
        for xa, xb in [(24.5, 29.5), (52, 55.5)]:
            cv.add_patch(FancyArrowPatch((xa, y0 + 4.2), (xb, y0 + 4.2), arrowstyle="-|>",
                                         mutation_scale=7, color=MUTED, lw=0.6))
    cv.text(56.5, 1.0, r"$\theta$ log-spaced in $[1, 100]$; concatenated over $w = 1, 2, 3$: "
            r"$3 \times 20 + 9 = 69$ coordinates", fontsize=6.5, color=MUTED)
    cv.text(56.5, -2.0, "shade = magnitude; layer bars = mass at each distance",
            fontsize=6.5, color=MUTED)
    fig.savefig(os.path.join(OUT, "fig_method.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_method.png"), dpi=200, bbox_inches="tight")


def shapes_layout(G, n_cycle):
    pos = {}
    R = 3.0
    for v in range(n_cycle):
        a = 2 * np.pi * v / n_cycle
        pos[v] = R * np.array([np.cos(a), np.sin(a)])
    for anchor in range(n_cycle):
        motif = [v for v in G[anchor] if v >= n_cycle]
        if not motif:
            continue
        nodes = nx.node_connected_component(G.subgraph(range(n_cycle, len(G))), motif[0])
        sub = G.subgraph(nodes)
        a = 2 * np.pi * anchor / n_cycle
        off = min(sub)
        # nx.house_graph: base 0-1, walls 0-2 and 1-3, roof 4; 0 is attached
        local = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 1.7)}
        rot = a - np.pi / 2  # local +y points away from the centre
        Rm = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        base = (R + 0.6) * np.array([np.cos(a), np.sin(a)])
        for v in sub:
            pos[v] = base + 0.62 * Rm @ np.array(local[v - off])
    return pos


def roles_figure(n_houses=8):
    G, y = generate_shapes_graph(("house",) * n_houses, spacing=3, seed=0)
    n_cycle = 3 * n_houses
    X = RoleWalk().transform(G)
    k = len(set(y))
    clusters = KMeans(k, n_init=20, random_state=0).fit_predict(X)
    # match clusters to roles for colouring
    C = np.zeros((k, k))
    for c, t in zip(clusters, y):
        C[c, t] += 1
    rows, cols = linear_sum_assignment(-C)
    match = dict(zip(rows, cols))
    pred = np.array([match[c] for c in clusters])
    pos = shapes_layout(G, n_cycle)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    for u, v in G.edges():
        ax.plot(*zip(pos[u], pos[v]), color=EDGE, lw=0.6, zorder=1)
    for v in G:
        ax.scatter(*pos[v], s=34, color=ROLES[pred[v]], ec="#ffffff", lw=0.7, zorder=3)
        if pred[v] != y[v]:
            ax.scatter(*pos[v], s=90, facecolor="none", ec=INK, lw=0.8, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    acc = (pred == y).mean()
    ax.set_title(f"(a) k-means on RoleWalk ({acc:.0%} of nodes in their role's cluster)",
                 fontsize=8)

    Z = PCA(2).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-12))
    rng = np.random.default_rng(0)
    Z += rng.normal(scale=0.03 * Z.std(), size=Z.shape)  # copies coincide: jitter
    for t in range(k):
        m = y == t
        bx.scatter(Z[m, 0], Z[m, 1], s=26, color=ROLES[t], ec="#ffffff", lw=0.6,
                   label=f"role {t + 1}", zorder=3)
    bx.set_xticks([])
    bx.set_yticks([])
    for s in ["top", "right"]:
        bx.spines[s].set_visible(False)
    bx.set_xlabel("PC 1")
    bx.set_ylabel("PC 2")
    bx.set_title("(b) PCA of the embeddings, by true role", fontsize=8)
    bx.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5),
              handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_roles.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_roles.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    method_figure()
    roles_figure()
