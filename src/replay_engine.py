"""Runs the continual learning experiment: one classifier per strategy, trained task after task.

naive        trains on the newest task only (lower bound)
generative   new data plus synthetic old customers from the CVAE (proposed method)
marginal     new data plus synthetic customers sampled feature by feature (control)
real_replay  new data plus a small buffer of real old customers
joint        new data plus all old data (upper bound, needs the full history)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .classifier import METRICS, ChurnClassifier

STRATEGIES = ("naive", "generative", "marginal", "real_replay", "joint")
GENERATIVE = ("generative", "marginal")
KEEPS_OLD_DATA = ("real_replay", "joint")


@dataclass
class Config:
    hidden_layers: tuple = (64, 32)
    clf_lr: float = 0.01
    clf_epochs: int = 6
    clf_alpha: float = 1e-4
    threshold: float = None          # None: use the churn rate of Task 1
    vae_hidden: int = 128
    latent_dim: int = 16
    vae_lr: float = 1e-3
    vae_epochs: int = 60
    beta: float = 1.0
    num_weight: float = 5.0
    warmup_epochs: int = 10
    vae_batch_size: int = 256
    replay_ratio: float = 1.0        # synthetic customers per old real customer
    replay_labels: str = "classifier"  # "classifier": previous model labels them, "generator": the CVAE labels them
    buffer_size: int = 500
    verbose: bool = False


def make_generator(kind, input_dim, numeric_mask, cfg, seed):
    if kind == "generative":
        from .vae import ConditionalVAEGenerator
        return ConditionalVAEGenerator(input_dim, numeric_mask, hidden_dim=cfg.vae_hidden,
                                       latent_dim=cfg.latent_dim, lr=cfg.vae_lr, beta=cfg.beta,
                                       num_weight=cfg.num_weight, warmup_epochs=cfg.warmup_epochs,
                                       batch_size=cfg.vae_batch_size, seed=seed)
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
        # about 29% of customers churn, so a 0.5 cut-off predicts almost no churn.
        # every strategy uses the same threshold: the churn rate of Task 1.
        self.threshold = cfg.threshold if cfg.threshold is not None else float(np.mean(tasks[0].y_train))
        self.generators = {}

    def new_classifier(self, seed):
        c = self.cfg
        return ChurnClassifier(c.hidden_layers, c.clf_lr, seed=seed, alpha=c.clf_alpha, threshold=self.threshold)

    def run_strategy(self, strategy, seed):
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        clf = self.new_classifier(seed)
        gen = None
        if strategy in GENERATIVE:
            gen = self.make_generator(strategy, self.input_dim, self.numeric_mask, cfg, seed)

        # old real data is only kept for the strategies that need it
        old_X, old_y, n_old = [], [], 0
        final_rows, curve_rows, gen_rows = [], [], []

        for t, task in enumerate(self.tasks):
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
                        # the classifier still holds the weights from the previous task here,
                        # so it labels the synthetic customers the way the old model would
                        extra_y = (clf.predict_proba(extra_X) >= self.threshold).astype(np.int64)
                    extra_kind = "synthetic"
                if extra_kind:
                    X_fit = np.vstack([X_fit, extra_X])
                    y_fit = np.concatenate([y_fit, extra_y])
                    n_extra = len(extra_y)

            def on_epoch(epoch, t=t):
                row_log = []
                for j in range(t + 1):
                    m = clf.evaluate(self.tasks[j].X_test, self.tasks[j].y_test)
                    curve_rows.append({"strategy": strategy, "seed": seed, "train_task": t, "epoch": epoch,
                                       "global_step": t * cfg.clf_epochs + epoch, "eval_task": j, **m})
                    row_log.append(f"T{j + 1} auc={m['auc']:.3f}")
                if cfg.verbose:
                    self.log(f"      epoch {epoch:>2}/{cfg.clf_epochs} | " + " | ".join(row_log))

            clf.fit(X_fit, y_fit, cfg.clf_epochs, on_epoch)

            parts = []
            for j in range(t + 1):
                m = clf.evaluate(self.tasks[j].X_test, self.tasks[j].y_test)
                final_rows.append({"strategy": strategy, "seed": seed, "after_task": t, "eval_task": j, **m})
                parts.append(f"T{j + 1}: F1={m['f1']:.3f} AUC={m['auc']:.3f}")
            used = f"{len(task.y_train):,} real" + (f" + {n_extra:,} {extra_kind}" if n_extra else "")
            self.log(f"  [{strategy:<11}| seed {seed}] after Task {t + 1} (trained on {used}) -> " + " | ".join(parts))

            # the generator learns from the same mix as the classifier (new real + replayed customers),
            # so it never needs the old real data
            if gen is not None:
                for h in gen.fit(X_fit, y_fit, cfg.vae_epochs):
                    gen_rows.append({"strategy": strategy, "seed": seed, "task": t, **h})

            if strategy in KEEPS_OLD_DATA:
                old_X.append(task.X_train)
                old_y.append(task.y_train)
            n_old += len(task.y_train)

        if gen is not None and strategy not in self.generators:
            self.generators[strategy] = gen
        return final_rows, curve_rows, gen_rows

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

    def evaluate_generator(self, gen, seed=0, max_rows=20000):
        """Compares synthetic customers with the real ones.

        mean_abs_feature_gap   average difference between the feature means
        correlation_mae        average difference between the feature correlations
        real_vs_synthetic_auc  how well a classifier tells real from synthetic (0.5 = it cannot)
        tstr / trtr            AUC on real test data of a classifier trained on synthetic / real data
        """
        cfg = self.cfg
        X_real = np.vstack([t.X_train for t in self.tasks])
        y_real = np.concatenate([t.y_train for t in self.tasks])
        X_syn, y_syn = gen.sample(len(X_real))

        def correlations(X):
            with np.errstate(all="ignore"):
                return np.nan_to_num(np.corrcoef(X, rowvar=False))

        out = {"mean_abs_feature_gap": float(np.mean(np.abs(X_real.mean(0) - X_syn.mean(0)))),
               "correlation_mae": float(np.mean(np.abs(correlations(X_real) - correlations(X_syn))))}

        rng = np.random.default_rng(seed)
        n = min(max_rows, len(X_real))
        X_mix = np.vstack([X_real[rng.choice(len(X_real), n, replace=False)],
                           X_syn[rng.choice(len(X_syn), n, replace=False)]])
        is_synthetic = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        X_tr, X_te, s_tr, s_te = train_test_split(X_mix, is_synthetic, test_size=0.3,
                                                  random_state=seed, stratify=is_synthetic)
        judge = self.new_classifier(seed)
        judge.fit(X_tr, s_tr, cfg.clf_epochs)
        out["real_vs_synthetic_auc"] = float(roc_auc_score(s_te, judge.predict_proba(X_te)))

        clf_syn = self.new_classifier(seed)
        clf_syn.fit(X_syn, y_syn, cfg.clf_epochs)
        clf_real = self.new_classifier(seed)
        clf_real.fit(X_real, y_real, cfg.clf_epochs)
        for j, task in enumerate(self.tasks):
            out[f"tstr_auc_task{j + 1}"] = clf_syn.evaluate(task.X_test, task.y_test)["auc"]
            out[f"trtr_auc_task{j + 1}"] = clf_real.evaluate(task.X_test, task.y_test)["auc"]
        return out

    def storage_report(self, gen):
        row = 4 * self.input_dim
        return {"generator_kb": gen.size_bytes() / 1024,
                "real_buffer_kb": self.cfg.buffer_size * row / 1024,
                "full_history_kb": sum(len(t.y_train) for t in self.tasks[:-1]) * row / 1024}


def summarise(final_df, n_tasks):
    """One row per strategy and seed.

    avg_  mean score over all tasks after the last task was learned
    bwt_  backward transfer: final score minus the score right after learning, averaged over old tasks
    t1_   final score on the first task that was learned
    """
    last = n_tasks - 1
    rows = []
    for (strategy, seed), g in final_df.groupby(["strategy", "seed"]):
        row = {"strategy": strategy, "seed": seed}
        for m in METRICS:
            M = g.pivot(index="after_task", columns="eval_task", values=m)
            row[f"avg_{m}"] = M.loc[last].mean()
            row[f"bwt_{m}"] = float(np.mean([M.loc[last, j] - M.loc[j, j] for j in range(last)]))
            row[f"t1_{m}"] = M.loc[last, 0]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(per_seed, order=STRATEGIES):
    cols = [c for c in per_seed.columns if c not in ("strategy", "seed")]
    agg = per_seed.groupby("strategy")[cols].agg(["mean", "std"]).fillna(0.0)
    return agg.reindex([s for s in order if s in agg.index])


def format_table(agg, cols):
    out = pd.DataFrame(index=agg.index)
    for c in cols:
        sign = "+" if c.startswith("bwt") else ""
        out[c] = [f"{agg.loc[s, (c, 'mean')]:{sign}.3f} +- {agg.loc[s, (c, 'std')]:.3f}" for s in agg.index]
    return out
