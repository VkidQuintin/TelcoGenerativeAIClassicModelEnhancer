"""All figures used by the project. Every figure is saved as a PNG."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

COLOURS = {"naive": "#d62728", "generative": "#1f77b4", "marginal": "#ff7f0e",
           "real_replay": "#2ca02c", "joint": "#000000"}
LABELS = {"naive": "Naive (new data only)", "generative": "Generative replay (CVAE)",
          "marginal": "Marginal sampler (control)", "real_replay": "Real buffer replay",
          "joint": "Joint / full retrain (upper bound)"}


def _interactive():
    return matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template")


def _finish(fig, path, show):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show and _interactive():
        plt.show(block=False)
        plt.pause(0.2)
    else:
        plt.close(fig)


def keep_open(show):
    """Keeps the figure windows open until they are closed."""
    if not show:
        return
    if _interactive():
        print("\nFigure windows are open, close them to finish. The figures are also saved as PNG files.")
        plt.show()
    else:
        print("\nNo window backend found, open the PNG files in the figures folder instead.")


def _draw_table(ax, snap):
    ax.axis("off")
    rows, cols = snap["shape"]
    extra = f", {snap['note']}" if snap["note"] else ""
    ax.set_title(f"{snap['title']}   [{rows:,} rows x {cols} columns{extra}]",
                 loc="left", fontsize=9, fontweight="bold")
    table = snap["table"]
    cells = table.astype(str).to_numpy().tolist()
    tbl = ax.table(cellText=cells, colLabels=[str(c) for c in table.columns], loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(list(range(len(table.columns))))
    tbl.scale(1, 1.35)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dbe8f7")
            cell.set_text_props(fontweight="bold")


def plot_snapshots(snapshots, path, show=False):
    fig, axes = plt.subplots(len(snapshots), 1, figsize=(15, 1.75 * len(snapshots)))
    for ax, snap in zip(np.atleast_1d(axes), snapshots):
        _draw_table(ax, snap)
    fig.suptitle("Dataset snapshots at every stage of dataset.py", fontsize=12, fontweight="bold")
    _finish(fig, path, show)


def save_snapshot_images(snapshots, folder):
    for i, snap in enumerate(snapshots, start=1):
        fig, ax = plt.subplots(figsize=(15, 2.2))
        _draw_table(ax, snap)
        fig.savefig(folder / f"snapshot_{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        snap["table"].to_csv(folder / f"snapshot_{i}.csv", index=False)


def plot_dataset_overview(proc, tasks, path, show=False):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    counts = np.bincount(proc.y_all, minlength=2)
    ax[0].bar(["Retained (0)", "Churned (1)"], counts, color=["#4c78a8", "#e45756"])
    for i, c in enumerate(counts):
        ax[0].text(i, c, f"{c:,}", ha="center", va="bottom")
    ax[0].set_title("Checkpoint 1: target balance")

    ax[1].hist(proc.tenure_all, bins=50, color="#636EFA")
    for b in proc.boundaries:
        ax[1].axvline(b, color="red", ls="--")
    ax[1].set_title("Checkpoint 2: task boundaries on tenure")
    ax[1].set_xlabel(proc.split_col)

    rates = [np.concatenate([t.y_train, t.y_test]).mean() for t in tasks]
    ax[2].bar([t.name for t in tasks], rates, color="#72b7b2")
    for i, r in enumerate(rates):
        ax[2].text(i, r, f"{r:.1%}", ha="center", va="bottom")
    ax[2].set_title("Checkpoint 3: churn rate per task")
    _finish(fig, path, show)


def plot_scaling_check(proc, path, show=False):
    if proc.scaling_demo is None:
        return
    name, raw, scaled = proc.scaling_demo
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].boxplot(raw)
    ax[0].set_title(f"{name}: raw values")
    ax[1].boxplot(scaled)
    ax[1].set_title(f"{name}: clipped and scaled (0 to 1)")
    _finish(fig, path, show)


def plot_forgetting_curves(curves, epochs_per_task, path, eval_task=0, show=False):
    df = curves[curves["eval_task"] == eval_task]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (metric, title) in zip(axes, (("auc", "AUC"), ("f1", "F1 (churn class)"))):
        for strategy, g in df.groupby("strategy"):
            agg = g.groupby("global_step")[metric].agg(["mean", "std"]).fillna(0.0)
            ls = "--" if strategy == "joint" else "-"
            ax.plot(agg.index, agg["mean"], ls, label=LABELS.get(strategy, strategy), color=COLOURS.get(strategy))
            ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            alpha=0.15, color=COLOURS.get(strategy))
        for b in range(1, int((df["global_step"].max() - 1) // epochs_per_task) + 1):
            ax.axvline(b * epochs_per_task + 0.5, color="grey", ls=":")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("training epoch (dotted lines: a new task starts)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} on Task {eval_task + 1} test set")
    axes[0].legend(fontsize=8)
    _finish(fig, path, show)


def plot_forgetting_measure(forget, epochs_per_task, path, show=False):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, (metric, title) in zip(axes, (("acc", "Accuracy"), ("f1", "F1 (churn class)"), ("auc", "AUC"))):
        col = f"forget_{metric}"
        for strategy, g in forget.groupby("strategy"):
            agg = g.groupby("global_step")[col].agg(["mean", "std"]).fillna(0.0)
            ls = "--" if strategy == "joint" else "-"
            ax.plot(agg.index, agg["mean"], ls, marker="o", ms=3, label=LABELS.get(strategy, strategy),
                    color=COLOURS.get(strategy))
            ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            alpha=0.15, color=COLOURS.get(strategy))
        for b in range(1, int((forget["global_step"].max() - 1) // epochs_per_task) + 1):
            ax.axvline(b * epochs_per_task + 0.5, color="grey", ls=":")
        ax.axhline(0, color="black", lw=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("training epoch (dotted lines: a new task starts)")
        ax.set_ylabel(f"forgetting in {title}")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Forgetting per epoch = score right after learning a task minus score now "
                 "(higher means more forgotten, 0 means nothing forgotten)", fontsize=10)
    fig.text(0.5, -0.02, "A negative value means the score went up. Accuracy can rise while F1 falls "
             "when the model predicts churn less often.", ha="center", fontsize=8, style="italic")
    _finish(fig, path, show)


def plot_summary(agg, path, show=False):
    strategies = list(agg.index)
    x = np.arange(len(strategies))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metrics, title in ((axes[0], ("avg_auc", "avg_f1"), "Average over all tasks after the last task"),
                               (axes[1], ("bwt_auc", "bwt_f1"), "Backward transfer (0 = nothing forgotten)")):
        for k, m in enumerate(metrics):
            ax.bar(x + (k - 0.5) * 0.38, [agg.loc[s, (m, "mean")] for s in strategies], 0.38,
                   yerr=[agg.loc[s, (m, "std")] for s in strategies], capsize=3,
                   label=m.split("_")[1].upper())
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.legend()
    _finish(fig, path, show)


def plot_generator_training(gen_history, path, show=False):
    df = gen_history[(gen_history["strategy"] == "generative") & (gen_history["seed"] == gen_history["seed"].min())]
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, title in zip(axes.ravel(), ("loss", "recon_num", "recon_bin", "kl"),
                              ("total loss", "reconstruction (continuous)", "reconstruction (binary)", "KL divergence")):
        for task, g in df.groupby("task"):
            ax.plot(g["epoch"], g[col], label=f"trained after Task {task + 1}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=8)
    _finish(fig, path, show)


def plot_real_vs_synthetic(X_real, X_syn, feature_names, numeric_idx, path, k=6, show=False):
    order = np.argsort(X_real[:, numeric_idx].var(axis=0))[::-1][:k]
    cols = [numeric_idx[i] for i in order]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(X_real[:, c], bins=30, range=(0, 1), density=True, alpha=0.55, label="real")
        ax.hist(X_syn[:, c], bins=30, range=(0, 1), density=True, alpha=0.55, label="synthetic")
        ax.set_title(feature_names[c])
    axes[0, 0].legend()
    fig.suptitle("Real vs synthetic customers (scaled features)")
    _finish(fig, path, show)
