"""Score against embedding time, one panel per benchmark (paper Fig. 1).

Reads roles.csv, struc2vec_runs.csv, directed.csv and roles_rfa.csv and
writes ../paper/fig_speed.svg.

    python speed_figure.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ACCENT = "#2a78d6"
OTHER = "#8a8985"
INK = "#0b0b0b"
MUTED = "#52514e"

NAMES = {
    "rolewalk": "RoleWalk", "rolewalk_curvature": "RoleWalk + curv.",
    "struc2vec": "struc2vec", "graphwave": "GraphWave", "role2vec": "Role2Vec",
    "resd": "RESD", "refex": "ReFeX", "rwse": "RWSE", "degree": "Degree (baseline)",
}


def airports():
    df = pd.read_csv("roles.csv")
    df = df[df.dataset.isin(["brazil", "europe", "usa"]) & df.method.isin(NAMES)]
    out = df.groupby("method")[["time", "accuracy_mean"]].mean()
    runs = pd.read_csv("struc2vec_runs.csv")
    out.loc["struc2vec", "accuracy_mean"] = runs.groupby("dataset").accuracy.mean().mean()
    return out.rename(columns={"accuracy_mean": "score"})


def directed():
    df = pd.read_csv("directed.csv")
    df = df[(df.noise == 0.05) & df.method.isin(NAMES)]
    return df.groupby("method")[["time", "accuracy"]].mean().rename(columns={"accuracy": "score"})


def rfa():
    df = pd.read_csv("roles_rfa.csv")
    df = df[(df.dataset == "rfa-candidates") & df.method.isin(NAMES)]
    return df.set_index("method")[["time", "accuracy_mean"]].rename(columns={"accuracy_mean": "score"})


PANELS = [
    ("Airports (3 networks, ~1k nodes)", airports, "accuracy"),
    ("Directed motifs (5% noise)", directed, "accuracy"),
    ("wiki-RfA: elected? (11k nodes)", rfa, "accuracy"),
]

# label offsets in points, to avoid collisions (method -> (dx, dy, ha))
OFFSETS = {
    0: {"degree": (-4, -12, "left"), "rwse": (6, -3, "left"), "rolewalk": (-7, -8, "right"),
        "rolewalk_curvature": (-7, 1, "right"), "refex": (6, -3, "left"),
        "resd": (0, 6, "center"), "graphwave": (6, -3, "left")},
    1: {"refex": (6, -9, "left"), "rolewalk": (6, -9, "left"), "rolewalk_curvature": (6, 3, "left"),
        "degree": (6, 4, "left"), "role2vec": (6, 4, "left"), "resd": (6, -4, "left"),
        "graphwave": (-7, -3, "right")},
    2: {"rolewalk": (-7, -3, "right"), "rolewalk_curvature": (-7, -3, "right"),
        "resd": (0, 6, "center"), "role2vec": (0, -12, "center"), "refex": (6, -3, "left")},
}


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for k, (ax, (title, load, ylabel)) in enumerate(zip(axes, PANELS)):
        data = load()
        for method, row in data.iterrows():
            ours = method.startswith("rolewalk")
            if method == "degree":  # trivial baseline, not an embedding
                ax.scatter(row.time, row.score, s=22, marker="o", facecolor="white",
                           edgecolor=OTHER, linewidth=1.2, zorder=3)
            else:
                ax.scatter(row.time, row.score, s=36 if ours else 22,
                           marker="D" if method == "rolewalk_curvature" else "o",
                           color=ACCENT if ours else OTHER, zorder=3,
                           edgecolor="white", linewidth=1)
            dx, dy, ha = OFFSETS.get(k, {}).get(method, (6, -3, "left"))
            ax.annotate(NAMES[method], (row.time, row.score), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, fontsize=7,
                        color=INK if ours else MUTED,
                        fontweight="bold" if ours else "normal",
                        fontstyle="italic" if method == "degree" else "normal")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("embedding time (s, log)")
        if k == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, color="#e6e5e1", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo / 2, hi * 4)
    fig.tight_layout(w_pad=1.2)
    fig.savefig("../paper/fig_speed.svg")
    fig.savefig("../paper/fig_speed.png", dpi=150)


if __name__ == "__main__":
    main()
