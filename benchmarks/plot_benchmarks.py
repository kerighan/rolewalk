"""Plot the benchmark CSVs for the README, in light and dark versions.

    python plot_benchmarks.py

Writes ``docs/img/<name>-light.png`` and ``<name>-dark.png`` at the
repository root; the README picks one with a ``<picture>`` element.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "img")

# validated categorical slots (blue, orange, aqua) for GitHub's light and
# dark backgrounds; text never wears a series color
THEMES = {
    "light": dict(surface="#ffffff", text="#0b0b0b", muted="#52514e",
                  grid="#e6e5e1", s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
                  neutral="#a9a8a2"),
    "dark": dict(surface="#0d1117", text="#f0efec", muted="#c3c2b7",
                 grid="#2b2f36", s1="#3987e5", s2="#d95926", s3="#199e70",
                 neutral="#6b6f76"),
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


def line(ax, x, y, color, label, t):
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", marker="o", markersize=6,
            markeredgecolor=t["surface"], markeredgewidth=1.5, label=label)


def role_recovery(t, mode):
    df = pd.concat([pd.read_csv(os.path.join(HERE, f))
                    for f in ("synthetic.csv", "synthetic_large.csv")])
    df = df[df.method.isin(["rolewalk", "graphwave"])]
    fig, axes = figure(t, 1, 2, figsize=(9, 3.6))
    for ax, metric, name in zip(axes, ["ari", "accuracy"],
                                ["k-means ARI", "Classification accuracy"]):
        mean = df.groupby(["method", "noise"])[metric].mean().unstack("method")
        x = mean.index * 100
        line(ax, x, mean["graphwave"], t["s2"], "GraphWave", t)
        line(ax, x, mean["rolewalk"], t["s1"], "RoleWalk", t)
        style(ax, t)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Edges perturbed (%)")
        ax.set_title(name, fontsize=10.5, loc="left", color=t["text"])
    legend(axes[0], t, loc="lower left")
    title(fig, t, "Role recovery under edge noise",
          "Motifs on a cycle (GraphWave paper), 80-2,750 nodes, mean over "
          "4 graph sizes and 3-5 seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "role_recovery", mode)


def speed(t, mode):
    df = pd.concat([pd.read_csv(os.path.join(HERE, f))
                    for f in ("synthetic.csv", "synthetic_large.csv")])
    df = df[df.method.isin(["rolewalk", "graphwave"])]
    time = df.groupby(["n_nodes", "method"]).time.mean().unstack("method")
    fig, ax = figure(t, figsize=(9, 3.2))
    y = np.arange(len(time))
    for i, (n, row) in enumerate(time.iterrows()):
        ax.plot([row.rolewalk, row.graphwave], [i, i], color=t["grid"],
                linewidth=2, zorder=1)
        ax.text(row.graphwave * 1.3, i, f"{row.graphwave / row.rolewalk:,.0f}x",
                color=t["text"], va="center", fontsize=9)
    ax.scatter(time.graphwave, y, s=60, color=t["s2"], zorder=2,
               edgecolors=t["surface"], linewidths=1.5, label="GraphWave")
    ax.scatter(time.rolewalk, y, s=60, color=t["s1"], zorder=2,
               edgecolors=t["surface"], linewidths=1.5, label="RoleWalk")
    ax.set_xscale("log")
    ax.set_xlim(time.values.min() / 2, time.values.max() * 4)
    ax.set_yticks(y, [f"{n:,} nodes" for n in time.index])
    ax.set_xlabel("Embedding time (s, log scale)")
    style(ax, t, xgrid=True, ygrid=False)
    legend(ax, t, loc="lower right")
    title(fig, t, "Embedding time",
          "Same graphs as above, one CPU core; labels give RoleWalk's speed-up")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "speed", mode)


def centrality(t, mode):
    within = pd.read_csv(os.path.join(HERE, "centrality.csv"))
    transfer = pd.read_csv(os.path.join(HERE, "centrality_transfer.csv"))
    w = within.groupby(["target", "features"]).spearman.mean().unstack()
    tr = transfer.groupby(["target", "features"]).spearman.mean().unstack()
    order = w["rolewalk"].sort_values().index
    names = {"core_number": "k-core", "pagerank": "PageRank"}
    fig, ax = figure(t, figsize=(9, 4))
    y = np.arange(len(order))
    for i, target in enumerate(order):
        lo = w.loc[target, "degree"]
        hi = max(w.loc[target, "rolewalk"], tr.loc[target, "rolewalk"])
        ax.plot([lo, hi], [i, i], color=t["grid"], linewidth=2, zorder=1)
    for values, color, label, offset in [
            (w.loc[order, "degree"], t["s2"], "log-degree only", 0),
            (w.loc[order, "rolewalk"], t["s1"], "RoleWalk, unseen nodes", 0.12),
            (tr.loc[order, "rolewalk"], t["s3"], "RoleWalk, another graph", -0.12)]:
        ax.scatter(values, y + offset, s=55, color=color, zorder=2,
                   edgecolors=t["surface"], linewidths=1.5, label=label)
    ax.set_yticks(y, [names.get(n, n) for n in order])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Spearman correlation with the true centrality")
    style(ax, t, xgrid=True, ygrid=False)
    legend(ax, t, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    title(fig, t, "Centralities predicted from the embedding",
          "Gradient boosting, 6 graph families of 1,000-2,000 nodes")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "centrality", mode)


def betweenness(t, mode):
    df = pd.read_csv(os.path.join(HERE, "betweenness.csv"))
    families = {"barabasi_albert": "Barabási-Albert",
                "powerlaw_cluster": "powerlaw cluster",
                "watts_strogatz": "Watts-Strogatz"}
    fig, axes = figure(t, 1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (family, name) in zip(axes, families.items()):
        part = df[df.family == family]
        sampling = part[part.method == "sampling"].sort_values("k")
        line(ax, sampling.time, sampling.spearman, t["s2"],
             "Source sampling", t)
        first, last = sampling.iloc[0], sampling.iloc[-1]
        ax.annotate(f"k={int(first.k)}", (first.time, first.spearman),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", color=t["muted"], fontsize=8)
        ax.annotate(f"k={int(last.k)}", (last.time, last.spearman),
                    textcoords="offset points", xytext=(8, -3),
                    ha="left", color=t["muted"], fontsize=8)
        rw = part[part.method == "rolewalk"].iloc[0]
        ax.scatter([rw.time], [rw.spearman], s=90, color=t["s1"], zorder=3,
                   edgecolors=t["surface"], linewidths=1.5, label="RoleWalk")
        ax.annotate(f"{rw.spearman:.2f}", (rw.time, rw.spearman),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    color=t["text"], fontsize=9)
        exact = part[part.method == "exact"].iloc[0]
        ax.scatter([exact.time], [1], s=45, color=t["neutral"], zorder=2,
                   label="Exact")
        ax.annotate(f"exact, {exact.time:.0f} s", (exact.time, 1),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    color=t["muted"], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(0.07, 300)
        ax.set_ylim(0.75, 1.05)
        ax.set_title(name, fontsize=10.5, loc="left", color=t["text"])
        ax.set_xlabel("Time (s, log scale)")
        style(ax, t, xgrid=True)
    axes[0].set_ylabel("Spearman with exact")
    legend(axes[0], t, loc="lower right")
    title(fig, t, "Betweenness on new 5,000-node graphs: learning vs sampling",
          "RoleWalk is trained once on a 2,000-node graph of the same "
          "family (11-17 s), then predicts from the embedding")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "betweenness", mode)


def clustering(t, mode):
    df = pd.read_csv(os.path.join(HERE, "clustering.csv"))
    auto = df[df.setting == "auto"].groupby("method").ari.mean().sort_values()
    names = {"rolewalk (ward+silhouette, tolerance=0.05)":
             "RoleWalk (Ward + silhouette, tolerance 0.05)",
             "rolewalk tolerance=0": "RoleWalk, tolerance 0",
             "rolewalk tolerance=0.1": "RoleWalk, tolerance 0.1"}
    fixed = df[(df.setting == "fixed") & (df.method == "rolewalk (kmeans)")].ari.mean()
    ours = "rolewalk (ward+silhouette, tolerance=0.05)"
    fig, ax = figure(t, figsize=(9, 3.6))
    y = np.arange(len(auto))
    colors = [t["s1"] if m == ours else t["neutral"] for m in auto.index]
    ax.barh(y, auto.values, height=0.55, color=colors)
    for i, v in enumerate(auto.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=t["text"], fontsize=9)
    ax.axvline(fixed, color=t["muted"], linewidth=1)
    ax.text(fixed - 0.008, 1, "k-means given the true\nnumber of roles",
            color=t["muted"], fontsize=8, va="center", ha="right")
    ax.set_yticks(y, [names.get(m, m) for m in auto.index])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("ARI against true roles")
    style(ax, t, xgrid=True, ygrid=False)
    title(fig, t, "Finding the number of roles",
          "Synthetic graphs, 80-2,750 nodes, edge noise 0-5%")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save(fig, "clustering", mode)


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    for mode, t in THEMES.items():
        for plot in (role_recovery, speed, centrality, betweenness, clustering):
            plot(t, mode)
    print(f"written to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
