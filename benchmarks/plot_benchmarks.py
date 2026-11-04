"""Charts of the README, in light and dark versions.

    python plot_benchmarks.py

Writes ``docs/img/<name>-light.png`` and ``<name>-dark.png`` at the
repository root; the README picks one with a ``<picture>`` element.

- ``roles``: houses on a cycle, coloured by k-means on RoleWalk, and a PCA
  of the embeddings (computed here);
- ``speed``: accuracy against embedding time (from roles.csv,
  struc2vec_runs.csv, directed.csv and roles_rfa*.csv, as in the paper);
- ``alignment``, ``transfer``, ``scale``: the paper's Enron alignment,
  centrality transfer and scale tables (values copied from them below).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "img")
os.chdir(HERE)  # the loaders of speed_figure read CSVs from here

import speed_figure  # noqa: E402
from datasets import generate_shapes_graph  # noqa: E402
from illustration_figures import shapes_layout  # noqa: E402
from rolewalk import RoleWalk  # noqa: E402

plt.rcdefaults()  # illustration_figures sets a serif font for the paper

# validated categorical slots (blue, orange, aqua) for GitHub's light and
# dark backgrounds; text never wears a series color
THEMES = {
    "light": dict(surface="#ffffff", text="#0b0b0b", muted="#52514e",
                  grid="#e6e5e1", edge="#c3c2b7", s1="#2a78d6", s2="#eb6834",
                  s3="#1baf7a", neutral="#a9a8a2",
                  roles=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                         "#e87ba4", "#008300", "#6250d6"]),
    "dark": dict(surface="#0d1117", text="#f0efec", muted="#c3c2b7",
                 grid="#2b2f36", edge="#3d434c", s1="#3987e5", s2="#d95926",
                 s3="#199e70", neutral="#6b6f76",
                 roles=["#3987e5", "#d95926", "#199e70", "#c98800",
                        "#d4689a", "#2f9e3a", "#8070e6"]),
}


def style(ax, t, xgrid=False, ygrid=True):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(False)
    if ygrid:
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])


def figure(t, *args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def title(fig, t, text, subtitle):
    fig.text(0.01, 0.98, text, color=t["text"], fontsize=13,
             fontweight="semibold", va="top")
    fig.text(0.01, 0.91, subtitle, color=t["muted"], fontsize=9.5, va="top")


def legend(ax, t, **kwargs):
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(t["text"])


def save(fig, name, mode):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}-{mode}.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def line(ax, x, y, color, label, t, dashed=False):
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", marker="o", markersize=6,
            markeredgecolor=t["surface"], markeredgewidth=1.5, label=label,
            linestyle=(0, (4, 3)) if dashed else "-")


def roles(t, mode, n_houses=8):
    G, y = generate_shapes_graph(("house",) * n_houses, spacing=3, seed=0)
    X = RoleWalk().transform(G)
    k = len(set(y))
    clusters = KMeans(k, n_init=20, random_state=0).fit_predict(X)
    C = np.zeros((k, k))
    for c, r in zip(clusters, y):
        C[c, r] += 1
    rows, cols = linear_sum_assignment(-C)
    pred = np.array([dict(zip(rows, cols))[c] for c in clusters])
    pos = shapes_layout(G, 3 * n_houses)

    fig, (ax, bx) = figure(t, 1, 2, figsize=(9, 4.4),
                           gridspec_kw={"width_ratios": [1.1, 1]})
    for u, v in G.edges():
        ax.plot(*zip(pos[u], pos[v]), color=t["edge"], lw=0.9, zorder=1)
    for v in G:
        ax.scatter(*pos[v], s=46, color=t["roles"][pred[v]],
                   ec=t["surface"], lw=1, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("k-means on the embedding: one colour per role",
                 fontsize=10.5, loc="left", color=t["text"])

    Z = PCA(2).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-12))
    rng = np.random.default_rng(0)
    Z += rng.normal(scale=0.03 * Z.std(), size=Z.shape)  # copies coincide
    for r in range(k):
        m = y == r
        bx.scatter(Z[m, 0], Z[m, 1], s=40, color=t["roles"][r],
                   ec=t["surface"], lw=1, zorder=3)
    style(bx, t, xgrid=True)
    bx.set_xticklabels([])
    bx.set_yticklabels([])
    bx.set_xlabel("PC 1")
    bx.set_ylabel("PC 2")
    bx.set_title("PCA of the embedding, by true role", fontsize=10.5,
                 loc="left", color=t["text"])
    title(fig, t, "Same position, same coordinates",
          f"Eight houses on a cycle: all {len(G)} nodes land in their role's "
          f"cluster ({(pred == y).mean():.0%}); copies of a position coincide")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "roles", mode)


# label offsets in points where the paper's do not fit these sizes
SPEED_OFFSETS = {0: {"rolewalk_curvature": (6, 3, "left"),
                     "rolewalk": (-8, -5, "right")}}
SPEED_NAMES = {**speed_figure.NAMES, "rolewalk_curvature": "RoleWalk + curv.",
               "degree": "degree"}


def speed(t, mode):
    fig, axes = figure(t, 1, 3, figsize=(11, 3.9))
    for k, (ax, (name, load, _)) in enumerate(zip(axes, speed_figure.PANELS)):
        data = load()
        for method, row in data.iterrows():
            ours = method.startswith("rolewalk")
            if method == "degree":  # trivial baseline, not an embedding
                ax.scatter(row.time, row.score, s=42, facecolor=t["surface"],
                           edgecolor=t["neutral"], linewidth=1.5, zorder=3)
            else:
                ax.scatter(row.time, row.score, s=70 if ours else 42,
                           marker="D" if method == "rolewalk_curvature" else "o",
                           color=t["s1"] if ours else t["neutral"], zorder=3,
                           edgecolor=t["surface"], linewidth=1.5)
            dx, dy, ha = SPEED_OFFSETS.get(k, {}).get(
                method, speed_figure.OFFSETS.get(k, {}).get(method, (6, -3, "left")))
            ax.annotate(SPEED_NAMES[method], (row.time, row.score),
                        xytext=(dx * 1.3, dy * 1.3), textcoords="offset points",
                        ha=ha, fontsize=8.5, color=t["text"] if ours else t["muted"],
                        fontweight="semibold" if ours else "normal",
                        fontstyle="italic" if method == "degree" else "normal")
        ax.set_xscale("log")
        style(ax, t, xgrid=True)
        ax.margins(y=0.12)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo / 3, hi * 6)
        ax.set_xlabel("Embedding time (s, log scale)")
        ax.set_title(name, fontsize=10.5, loc="left", color=t["text"])
    axes[0].set_ylabel("Accuracy")
    title(fig, t, "Accurate and fast",
          "Role classification accuracy against embedding time, one CPU core")
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2)
    save(fig, "speed", mode)


# paper, table of network alignment on Enron (top-1 accuracy, mean of 5 noise
# draws; * published results of Digraphwave's paper)
ALIGN_P = [0, 0.01, 0.02, 0.05, 0.075, 0.1]
ALIGN = {
    "RoleWalk": [0.487, 0.406, 0.356, 0.235, 0.172, 0.125],
    "local statistics": [0.427, 0.372, 0.332, 0.249, 0.207, 0.176],
    "Digraphwave (published)": [0.398, 0.331, 0.278, 0.170, 0.116, 0.085],
    "ReFeX (published)": [0.151, 0.138, 0.127, 0.098, 0.080, 0.065],
}


def alignment(t, mode):
    fig, ax = figure(t, figsize=(9, 3.9))
    x = np.array(ALIGN_P) * 100
    colors = {"RoleWalk": t["s1"], "local statistics": t["s3"],
              "Digraphwave (published)": t["s2"], "ReFeX (published)": t["neutral"]}
    nudge = {"Digraphwave (published)": -1, "ReFeX (published)": -8}
    for name, ys in ALIGN.items():
        line(ax, x, ys, colors[name], name, t)
        ax.annotate(name, (x[-1], ys[-1]), xytext=(8, nudge.get(name, 0)),
                    textcoords="offset points",
                    va="center", fontsize=9, color=t["text"])
    style(ax, t)
    ax.set_xlim(-0.3, 13.5)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("Random edges added (% of edges)")
    ax.set_ylabel("Top-1 alignment accuracy")
    title(fig, t, "Recognising nodes in another graph",
          "Enron emails, Digraphwave's own protocol: 22,888-node subgraph "
          "matched into the 91,685-node graph")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "alignment", mode)


# paper, table of centrality transfer to graphs four times larger (Spearman,
# 60 pairs of graphs)
TRANSFER_TARGETS = ["PageRank", "betweenness", "harmonic", "clustering",
                    "closeness", "eigenvector", "k-core"]
TRANSFER = {
    "degree": [0.929, 0.623, 0.460, 0.201, 0.407, 0.463, 0.614],
    "GIN (trained)": [0.982, 0.778, 0.756, 0.387, 0.695, 0.736, 0.604],
    "local statistics": [0.984, 0.852, 0.778, 0.844, 0.716, 0.661, 0.650],
    "RoleWalk": [0.989, 0.864, 0.861, 0.805, 0.797, 0.673, 0.651],
}


def transfer(t, mode):
    fig, ax = figure(t, figsize=(9, 4.4))
    y = np.arange(len(TRANSFER_TARGETS))[::-1]
    values = np.array(list(TRANSFER.values()))
    for i in range(len(y)):
        ax.plot([values[:, i].min(), values[:, i].max()], [y[i]] * 2,
                color=t["grid"], linewidth=2, zorder=1)
    colors = {"degree": t["neutral"], "GIN (trained)": t["s2"],
              "local statistics": t["s3"], "RoleWalk": t["s1"]}
    for name, vs in TRANSFER.items():
        if name == "degree":
            ax.scatter(vs, y, s=55, facecolor=t["surface"], edgecolor=t["neutral"],
                       linewidth=1.5, zorder=2, label=name)
        else:
            ax.scatter(vs, y, s=70 if name == "RoleWalk" else 55, color=colors[name],
                       edgecolors=t["surface"], linewidths=1.5, zorder=3, label=name)
    style(ax, t, xgrid=True, ygrid=False)
    ax.set_yticks(y)
    ax.set_yticklabels(TRANSFER_TARGETS)
    ax.set_xlim(0.15, 1.02)
    ax.set_xlabel("Spearman correlation with the true centrality")
    legend(ax, t, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
           handletextpad=0.2, columnspacing=1.5)
    title(fig, t, "Trained on one graph, applied to a four times larger one",
          "Gradient boosting on the embedding, 6 graph families, 60 pairs "
          "of graphs (2,000 to 8,000 nodes)")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "transfer", mode)


# paper, table of embedding time on one core (defaults)
SCALE_N = [1e4, 3e4, 1e5, 3e5, 1e6]
SCALE = {
    "Barabási-Albert, m=3": [0.5, 1.8, 8.9, 43, 193],
    "Watts-Strogatz, k=6": [0.1, 0.5, 1.8, 5.1, 18],
}


def scale(t, mode):
    fig, ax = figure(t, figsize=(9, 3.6))
    for (name, ys), color in zip(SCALE.items(), (t["s1"], t["s3"])):
        line(ax, SCALE_N, ys, color, name, t)
        ax.annotate(f"{name}: {ys[-1]:.0f} s", (SCALE_N[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=t["text"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, t, xgrid=True)
    ax.set_xlim(7e3, 1e7)
    ax.set_xticks(SCALE_N)
    ax.set_xticklabels(["10k", "30k", "100k", "300k", "1M"])
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1 s", "1 s", "10 s", "100 s"])
    ax.minorticks_off()
    ax.set_xlabel("Nodes")
    title(fig, t, "A million nodes on one core",
          "Embedding time with the defaults, numba installed, one CPU core")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "scale", mode)


def main():
    for mode, t in THEMES.items():
        for chart in (roles, speed, alignment, transfer, scale):
            chart(t, mode)
            print(f"{chart.__name__}-{mode}")


if __name__ == "__main__":
    main()
