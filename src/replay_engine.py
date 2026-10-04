"""
replay_engine.py - runs the continual-learning experiment.

The data arrive as a sequence of tasks (Task 1 = newest customers ... Task N = oldest).
For each *strategy* we train one classifier through the sequence and, after every task,
measure how well it performs on the test sets of ALL tasks seen so far.

Strategies compared
-------------------
naive        fine-tune on the newest task only            -> lower bound (forgets most)
generative   new data + synthetic old data from the CVAE  -> OUR METHOD (generative replay)
marginal     new data + synthetic old data sampled feature-by-feature (no correlations) -> control
real_replay  new data + a small buffer of real old rows   -> memory-limited realistic baseline
joint        new data + ALL real old data (full retraining) -> upper bound (the expensive way)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .classifier import ChurnClassifier

STRATEGIES = ("naive", "generative", "marginal", "real_replay", "joint")
GENERATIVE = ("generative", "marginal")      # strategies that carry a generator between tasks


@dataclass
class Config:
    """All hyper-parameters in one place (filled from the command line by main.py)."""
    hidden_layers: tuple = (64, 32)   # classifier architecture
    clf_lr: float = 0.01
    clf_epochs: int = 6               # epochs per task (more than ~6 overfits on this data)
    clf_alpha: float = 1e-4           # L2 regularisation of the classifier
    threshold: float = None           # decision threshold; None = churn base rate of Task 1 (see ReplayEngine)
    vae_hidden: int = 128
    latent_dim: int = 16
    vae_lr: float = 1e-3
    vae_epochs: int = 60
    beta: float = 1.0                 # weight of the KL term
    num_weight: float = 5.0           # weight of continuous-column reconstruction error
    warmup_epochs: int = 10           # KL warm-up length
    vae_batch_size: int = 256
    replay_ratio: float = 1.0         # synthetic samples per real old sample (1.0 = same amount as old data)
    replay_labels: str = "classifier" # who labels synthetic customers: "classifier" (previous model) or "generator" (CVAE condition)
    buffer_size: int = 500            # rows kept by the 'real_replay' baseline
    verbose: bool = False             # print every epoch


def make_generator(kind, input_dim, numeric_mask, cfg, seed):
    """Factory: build the generator that belongs to a strategy name."""
    if kind == "generative":
        from .vae import ConditionalVAEGenerator   # imported here so PyTorch is only needed for this strategy
        return ConditionalVAEGenerator(input_dim, numeric_mask, hidden_dim=cfg.vae_hidden,
                                       latent_dim=cfg.latent_dim, lr=cfg.vae_lr, beta=cfg.beta,
                                       num_weight=cfg.num_weight, warmup_epochs=cfg.warmup_epochs, batch_size=cfg.vae_batch_size, seed=seed)
    if kind == "marginal":
        from .baselines import MarginalGenerator
        return MarginalGenerator(input_dim, numeric_mask, seed=seed)
    raise ValueError(f"unknown generator kind: {kind}")


class ReplayEngine:
    def __init__(self, tasks, numeric_mask, cfg, generator_factory=make_generator, log=print):
        self.tasks = tasks
        self.numeric_mask = numeric_mask
        self.cfg = cfg
        self.input_dim = tasks[0].X_train.shape[1]
        self.make_generator = generator_factory
        self.log = log
        # With ~29% churners a 0.5 cut-off predicts almost nobody as churn, which makes F1 meaningless.
        # We therefore use ONE fixed threshold for every strategy: the churn base rate seen in Task 1.
        self.threshold = cfg.threshold if cfg.threshold is not None else float(np.mean(tasks[0].y_train))
        self.generators = {}          # last generator of the first seed, per strategy (for saving / plots)

    def new_classifier(self, seed):
        c = self.cfg
        return ChurnClassifier(c.hidden_layers, c.clf_lr, seed=seed, alpha=c.clf_alpha, threshold=self.threshold)

    # ------------------------------------------------------------------ one run
    def run_strategy(self, strategy, seed):
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        clf = self.new_classifier(seed)
        gen = None
        if strategy in GENERATIVE:
            gen = self.make_generator(strategy, self.input_dim, self.numeric_mask, cfg, seed)

        old_X, old_y, n_old = [], [], 0            # real data from earlier tasks
        final_rows, curve_rows, gen_rows = [], [], []

        for t, task in enumerate(self.tasks):
            # ---- 1. assemble the training set for this task ------------------------------
            X_fit, y_fit = task.X_train, task.y_train
            n_extra, extra_kind = 0, ""
            if t > 0:
                if strategy == "joint":
                    extra_X, extra_y, extra_kind = np.vstack(old_X), np.concatenate(old_y), "real"
                elif strategy == "real_replay":
                    pool_X, pool_y = np.vstack(old_X), np.concatenate(old_y)
                    pick = rng.choice(len(pool_X), size=min(cfg.buffer_size, len(pool_X)), replace=False)
                    extra_X, extra_y, extra_kind = pool_X[pick], pool_y[pick], "buffered real"
                elif gen is not None:
                    extra_X, extra_y = gen.sample(int(cfg.replay_ratio * n_old))
                    if cfg.replay_labels == "classifier":
                        # Generative replay as in Shin et al. (2017): the PREVIOUS classifier labels the synthetic
                        # customers. `clf` still holds its old weights here (training on the new task has not started),
                        # so this hands the old decision rule to the new model. Same threshold as in the evaluation.
                        extra_y = (clf.predict_proba(extra_X) >= self.threshold).astype(np.int64)
                    extra_kind = "synthetic"
                if extra_kind:
                    X_fit = np.vstack([X_fit, extra_X])
                    y_fit = np.concatenate([y_fit, extra_y])
                    n_extra = len(extra_y)

            # ---- 2. train the classifier, evaluating after every epoch -------------------
            def on_epoch(epoch, t=t):
                row_log = []
                for j in range(t + 1):             # test sets of all tasks seen so far
                    m = clf.evaluate(self.tasks[j].X_test, self.tasks[j].y_test)
                    curve_rows.append({"strategy": strategy, "seed": seed, "train_task": t, "epoch": epoch,
                                       "global_step": t * cfg.clf_epochs + epoch, "eval_task": j, **m})
                    row_log.append(f"T{j + 1} auc={m['auc']:.3f}")
                if cfg.verbose:
                    self.log(f"      epoch {epoch:>2}/{cfg.clf_epochs} | " + " | ".join(row_log))

            clf.fit(X_fit, y_fit, cfg.clf_epochs, on_epoch)

            # ---- 3. record the end-of-task evaluation ------------------------------------
            parts = []
            for j in range(t + 1):
                m = clf.evaluate(self.tasks[j].X_test, self.tasks[j].y_test)
                final_rows.append({"strategy": strategy, "seed": seed, "after_task": t, "eval_task": j, **m})
                parts.append(f"T{j + 1}: F1={m['f1']:.3f} AUC={m['auc']:.3f}")
            used = f"{len(task.y_train):,} real" + (f" + {n_extra:,} {extra_kind}" if n_extra else "")
            self.log(f"  [{strategy:<11}| seed {seed}] after Task {t + 1} (trained on {used}) -> " + " | ".join(parts))

            # ---- 4. update the generator on exactly what the classifier just saw ---------
            # (new real data + replayed old data), so the generator itself does not forget either.
            if gen is not None:
                for h in gen.fit(X_fit, y_fit, cfg.vae_epochs):
                    gen_rows.append({"strategy": strategy, "seed": seed, "task": t, **h})

            old_X.append(task.X_train)
            old_y.append(task.y_train)
            n_old += len(task.y_train)

        if gen is not None and strategy not in self.generators:
            self.generators[strategy] = gen
        return final_rows, curve_rows, gen_rows

    # ------------------------------------------------------------------ many runs
    def run_all(self, strategies, seeds):
        finals, curves, gens = [], [], []
        for seed in seeds:
            for s in strategies:
                f, c, g = self.run_strategy(s, seed)
                finals += f
                curves += c
                gens += g
        return {"final": pd.DataFrame(finals), "curves": pd.DataFrame(curves),
                "gen_history": pd.DataFrame(gens)}

    # ------------------------------------------------------------------ generator quality
    def evaluate_generator(self, gen, seed=0):
        """How good is the synthetic data? (a) feature means, (b) correlations, (c) TSTR.

        TSTR = Train on Synthetic, Test on Real: train a fresh classifier ONLY on synthetic rows
        and score it on the real test set of each task. TRTR (train on real) is the reference.
        """
        cfg = self.cfg
        X_real = np.vstack([t.X_train for t in self.tasks])
        y_real = np.concatenate([t.y_train for t in self.tasks])
        X_syn, y_syn = gen.sample(len(X_real))

        def corr(X):
            with np.errstate(all="ignore"):
                return np.nan_to_num(np.corrcoef(X, rowvar=False))

        out = {"mean_abs_feature_gap": float(np.mean(np.abs(X_real.mean(0) - X_syn.mean(0)))),
               "correlation_mae": float(np.mean(np.abs(corr(X_real) - corr(X_syn))))}
        clf_syn = self.new_classifier(seed)
        clf_syn.fit(X_syn, y_syn, cfg.clf_epochs)
        clf_real = self.new_classifier(seed)
        clf_real.fit(X_real, y_real, cfg.clf_epochs)
        for j, task in enumerate(self.tasks):
            out[f"tstr_auc_task{j + 1}"] = clf_syn.evaluate(task.X_test, task.y_test)["auc"]
            out[f"trtr_auc_task{j + 1}"] = clf_real.evaluate(task.X_test, task.y_test)["auc"]
        return out

    def storage_report(self, gen):
        """The practical argument for generative replay: memory footprint."""
        row = 4 * self.input_dim                     # bytes per customer row (float32)
        return {"generator_kb": gen.size_bytes() / 1024,
                "real_buffer_kb": self.cfg.buffer_size * row / 1024,
                "full_history_kb": sum(len(t.y_train) for t in self.tasks[:-1]) * row / 1024}


# ---------------------------------------------------------------------- summary metrics
def summarise(final_df, n_tasks):
    """One row per (strategy, seed) with the standard continual-learning metrics.

    avg_*  : mean performance over ALL tasks after the last task has been learned
    bwt_*  : backward transfer = mean over old tasks of (final score - score right after learning it).
             0 = no forgetting, negative = forgetting (Lopez-Paz & Ranzato, 2017)
    t1_*   : final performance on the FIRST task (how much of the oldest knowledge survives)
    """
    last = n_tasks - 1
    rows = []
    for (strategy, seed), g in final_df.groupby(["strategy", "seed"]):
        row = {"strategy": strategy, "seed": seed}
        for m in ("acc", "f1", "auc"):
            M = g.pivot(index="after_task", columns="eval_task", values=m)   # M[i][j]: after task i, on task j
            row[f"avg_{m}"] = M.loc[last].mean()
            row[f"bwt_{m}"] = float(np.mean([M.loc[last, j] - M.loc[j, j] for j in range(last)]))
            row[f"t1_{m}"] = M.loc[last, 0]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(per_seed, order=STRATEGIES):
    """mean and std over seeds, ordered from lower bound to upper bound."""
    cols = [c for c in per_seed.columns if c not in ("strategy", "seed")]
    agg = per_seed.groupby("strategy")[cols].agg(["mean", "std"]).fillna(0.0)
    agg = agg.reindex([s for s in order if s in agg.index])
    return agg


def format_table(agg, cols=("avg_f1", "avg_auc", "bwt_f1", "bwt_auc", "t1_f1", "t1_auc")):
    """'mean +- std' strings for printing in the terminal."""
    out = pd.DataFrame(index=agg.index)
    for c in cols:
        out[c] = [f"{agg.loc[s, (c, 'mean')]:+.3f} +- {agg.loc[s, (c, 'std')]:.3f}" if c.startswith("bwt")
                  else f"{agg.loc[s, (c, 'mean')]:.3f} +- {agg.loc[s, (c, 'std')]:.3f}" for s in agg.index]
    return out
