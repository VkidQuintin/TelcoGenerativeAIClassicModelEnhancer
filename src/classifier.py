"""
classifier.py - the discriminative model (the "solver") that predicts churn.

A small multi-layer perceptron from scikit-learn. We train it one epoch at a time with
partial_fit() so that (a) we can measure performance after EVERY epoch (forgetting curves)
and (b) training on a second task continues from the weights learned on the first,
which is exactly the situation in which catastrophic forgetting appears.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

METRICS = ("acc", "precision", "recall", "f1", "auc")


def compute_metrics(y_true, proba, threshold=0.5):
    """Accuracy, precision, recall, F1 (churn = positive class) and AUC."""
    pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "acc": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, proba) if both_classes else float("nan"),
    }


class ChurnClassifier:
    def __init__(self, hidden_layers=(64, 32), lr=0.01, batch_size=256, seed=42, alpha=1e-4, threshold=0.5):
        # alpha = L2 weight penalty (regularisation); threshold = probability above which we predict "churn"
        self.threshold = threshold
        self.net = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers), learning_rate_init=lr,
                                 batch_size=batch_size, random_state=seed, alpha=alpha)

    def fit(self, X, y, epochs, on_epoch=None):
        """Train for `epochs` passes over (X, y); optionally call on_epoch(epoch) after each pass."""
        for epoch in range(1, epochs + 1):
            self.net.partial_fit(X, y, classes=[0, 1])
            if on_epoch is not None:
                on_epoch(epoch)

    def predict_proba(self, X):
        return self.net.predict_proba(X)[:, 1]      # probability of churn

    def evaluate(self, X, y):
        return compute_metrics(y, self.predict_proba(X), self.threshold)


# ---------------------------------------------------------------------- catastrophic-forgetting measure
def forgetting_measure(curves, metrics=("acc", "f1", "auc")):
    """Forgetting after EVERY epoch, for every strategy and seed.

    For an old task j, the baseline is the score the classifier reached on task j's test set at the end of
    training on task j. While it trains on a later task, forgetting(j) = baseline - current score on task j.
    The value at an epoch is the average over all old tasks. Positive = knowledge lost, 0 = nothing lost,
    negative = the old task even improved. During the very first task nothing can be forgotten yet, so it is 0.

    `curves` has one row per (strategy, seed, epoch, evaluated task); ReplayEngine and run_forgetting_demo produce it.
    """
    keys = ["strategy", "seed"]
    last_epoch = curves.groupby(keys + ["train_task"])["epoch"].transform("max")
    ends = curves[(curves["train_task"] == curves["eval_task"]) & (curves["epoch"] == last_epoch)]
    baseline = ends[keys + ["eval_task"] + list(metrics)].rename(columns={m: f"base_{m}" for m in metrics})

    later = curves[curves["train_task"] > curves["eval_task"]].merge(baseline, on=keys + ["eval_task"])
    for m in metrics:
        later[f"forget_{m}"] = later[f"base_{m}"] - later[m]
    per_epoch = later.groupby(keys + ["global_step"])[[f"forget_{m}" for m in metrics]].mean().reset_index()

    first = curves[curves["train_task"] == 0][keys + ["global_step"]].drop_duplicates()
    for m in metrics:
        first[f"forget_{m}"] = 0.0
    return pd.concat([first, per_epoch]).sort_values(keys + ["global_step"]).reset_index(drop=True)


# ---------------------------------------------------------------------- stand-alone forgetting test
def run_forgetting_demo(data_path="data/telecomm c2c.csv", n_tasks=3, keep_tenure=False, epochs=6,
                        hidden_layers=(64, 32), lr=0.01, alpha=1e-4, threshold=None, seed=42,
                        out_dir="outputs", show=True):
    """The classic catastrophic-forgetting test, with NO replay.

    One classifier learns Task 1, then Task 2, ... After every epoch we score it on all tasks seen so far and
    plot how much it has forgotten. This is the problem that generative replay is meant to solve.
    """
    from . import visuals                      # imported here to avoid circular imports at start-up
    from .dataset import DataProcessing

    proc = DataProcessing(data_path, n_tasks=n_tasks, drop_split_feature=not keep_tenure, seed=seed)
    tasks = proc.build_tasks()
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    visuals.plot_snapshots(proc.snapshots, fig_dir / "dataset_snapshots.png", show)

    thr = threshold if threshold is not None else float(np.mean(tasks[0].y_train))
    clf = ChurnClassifier(hidden_layers, lr, seed=seed, alpha=alpha, threshold=thr)
    rows = []
    print(f"\nNAIVE SEQUENTIAL TRAINING  (decision threshold {thr:.3f}, {epochs} epochs per task)")

    for t, task in enumerate(tasks):
        print(f"\n--- learning {task.name} ---")

        def on_epoch(epoch, t=t):
            step, parts = t * epochs + epoch, []
            for j in range(t + 1):                       # score on every task seen so far
                m = clf.evaluate(tasks[j].X_test, tasks[j].y_test)
                rows.append({"strategy": "naive", "seed": seed, "train_task": t, "epoch": epoch,
                             "global_step": step, "eval_task": j, **m})
                parts.append(f"T{j + 1}: acc={m['acc']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}")
            print(f"  epoch {epoch}/{epochs} | " + " | ".join(parts))

        clf.fit(task.X_train, task.y_train, epochs, on_epoch)

    curves = pd.DataFrame(rows)
    forget = forgetting_measure(curves)
    forget.to_csv(Path(out_dir) / "forgetting_demo.csv", index=False)
    print("\nFORGETTING after each task (average over earlier tasks; positive = forgotten):")
    print(forget[forget["global_step"] % epochs == 0].drop(columns=["strategy", "seed"]).round(4).to_string(index=False))

    visuals.plot_forgetting_measure(forget, epochs, fig_dir / "forgetting_demo.png", show)
    visuals.plot_forgetting_curves(curves, epochs, fig_dir / "forgetting_demo_task1_score.png", 0, show)
    print(f"\nFigures saved in {fig_dir}/")
    visuals.keep_open(show)
    return forget


if __name__ == "__main__":          # python -m src.classifier
    run_forgetting_demo()
