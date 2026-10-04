"""
dataset.py - data preparation for the continual-learning experiment.

Pipeline
--------
load CSV -> clean -> one-hot encode -> cut the customers into sequential "tasks"
by tenure (months in service) -> per-task train/test split -> MinMax scale.

Why tenure?  New customers (low tenure) and old customers (high tenure) behave
differently, so training on them one after the other simulates *concept drift*:
the classifier first sees "Task 1" (newest customers), then "Task 2", and so on.
"""
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CHURN_MAP = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}


@dataclass
class Task:
    """One slice of the data stream (a group of customers with similar tenure)."""
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class DataProcessing:
    def __init__(self, filepath, n_tasks=2, split_col="monthsinservice", test_size=0.2,
                 max_cardinality=20, clip_quantile=0.99, drop_split_feature=False,
                 seed=42, verbose=True):
        self.filepath = filepath
        self.n_tasks = n_tasks
        self.split_col = split_col
        self.test_size = test_size
        self.max_cardinality = max_cardinality   # categorical columns with more levels are dropped
        self.clip_quantile = clip_quantile       # winsorise numeric outliers at this quantile
        self.drop_split_feature = drop_split_feature
        self.seed = seed
        self.verbose = verbose
        self.scaler = MinMaxScaler()

        # Filled in by build_tasks(); other modules (VAE, plots) read these.
        self.feature_names = None
        self.numeric_idx = None      # column positions of continuous features
        self.numeric_mask = None     # same thing as a True/False vector
        self.boundaries = None       # tenure values where one task ends and the next begins
        self.tenure_all = None
        self.y_all = None
        self.scaling_demo = None
        self.snapshots = []          # data previews taken at every stage (see _snapshot)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------ snapshots
    def _snapshot(self, title, df, extra="", n_rows=5, max_cols=9):
        """Record (and print) a small preview of a DataFrame at one stage of the pipeline.

        Wide tables are cut to `max_cols` columns, always keeping churn and the split column first.
        """
        priority = [c for p in ("churn", self.split_col) for c in df.columns if str(c).lower() == p]
        cols = (priority + [c for c in df.columns if c not in priority])[:max_cols]
        table = df[cols].head(n_rows).copy()
        for c in table.select_dtypes(include="number").columns:
            table[c] = table[c].round(3)
        note = ", ".join(x for x in (f"showing {len(cols)} of {df.shape[1]} columns" if df.shape[1] > len(cols) else "", extra) if x)
        self.snapshots.append({"title": title, "shape": df.shape, "table": table, "note": note})
        if self.verbose:
            print(f"\n--- {title}  [{df.shape[0]:,} rows x {df.shape[1]} columns" + (f"; {note}" if note else "") + "] ---")
            print(table.to_string(index=False))

    # ------------------------------------------------------------------ cleaning
    def load_and_clean(self):
        df = pd.read_csv(self.filepath)
        self._snapshot("1. RAW CSV (exactly as loaded)", df, extra=f"{int(df.isna().sum().sum()):,} missing cells")
        df.columns = df.columns.str.strip().str.lower()
        df = df.drop(columns=[c for c in ("customerid",) if c in df.columns])

        # Target: map Yes/No (or 1/0) to 1/0 and drop rows where it is missing.
        df = df.dropna(subset=["churn"])
        df["churn"] = df["churn"].astype(str).str.strip().str.lower().map(CHURN_MAP)
        df = df.dropna(subset=["churn"])
        df["churn"] = df["churn"].astype(int)

        if self.split_col not in df.columns:
            raise ValueError(f"split column '{self.split_col}' not found in the CSV")
        df = df.dropna(subset=[self.split_col])

        # Missing values: median for numbers, "unknown" for text (the old code used 0 for everything).
        num_cols = [c for c in df.select_dtypes(include="number").columns if c != "churn"]
        cat_cols = list(df.select_dtypes(exclude="number").columns)
        df[num_cols] = df[num_cols].fillna(df[num_cols].median()).fillna(0)
        df[cat_cols] = df[cat_cols].fillna("unknown")

        # One-hot encoding a column with hundreds of levels (e.g. service area) would create
        # hundreds of mostly-empty columns and swamp the generator, so we drop those.
        too_many = [c for c in cat_cols if df[c].nunique() > self.max_cardinality]
        constant = [c for c in df.columns if c != "churn" and df[c].nunique() <= 1]
        df = df.drop(columns=too_many + constant)
        if too_many:
            self._log(f"[data] dropped high-cardinality columns: {too_many}")
        if constant:
            self._log(f"[data] dropped constant columns: {constant}")
        self._snapshot("2. CLEANED (churn -> 0/1, gaps filled, long categoricals dropped)", df,
                       extra=f"{int(df.isna().sum().sum())} missing cells left")
        return df

    # ------------------------------------------------------------------ tasks
    def build_tasks(self):
        df = self.load_and_clean()
        y = df["churn"].to_numpy(dtype=np.int64)
        tenure = df[self.split_col].to_numpy(dtype=float)

        # Encode ONCE on the full table. (The old code encoded each task separately with
        # drop_first=True, which can drop a *different* category in each task.)
        X = pd.get_dummies(df.drop(columns="churn"), drop_first=True, dtype=np.float32)
        if self.drop_split_feature:
            X = X.drop(columns=self.split_col)
        self.feature_names = list(X.columns)
        self._snapshot("3. ENCODED feature matrix (one-hot, before clipping and scaling)", X)
        X_np = X.to_numpy(dtype=np.float32)

        # A column is "binary" if every value is 0/1 (one-hot dummies, flags); the rest are continuous.
        is_binary = np.isin(X_np, [0, 1]).all(axis=0)
        self.numeric_mask = ~is_binary
        self.numeric_idx = np.where(~is_binary)[0]

        # Winsorise: clip each continuous column to its [1-q, q] quantiles. MinMax scaling is
        # very sensitive to outliers (one huge value squashes everyone else towards 0).
        num = self.numeric_idx
        raw_demo_col = self._pick_demo_column()
        raw_demo = X_np[:, raw_demo_col].copy() if raw_demo_col is not None else None
        lo = np.quantile(X_np[:, num], 1 - self.clip_quantile, axis=0)
        hi = np.quantile(X_np[:, num], self.clip_quantile, axis=0)
        X_np[:, num] = np.clip(X_np[:, num], lo, hi)

        # Cut into tasks of (roughly) equal size using tenure quantiles.
        task_id = pd.qcut(tenure, q=self.n_tasks, labels=False, duplicates="drop").astype(int)
        n_real = int(task_id.max()) + 1
        self.boundaries = np.quantile(tenure, np.linspace(0, 1, n_real + 1)[1:-1])
        self.tenure_all, self.y_all = tenure, y

        # Per-task stratified train/test split (keeps the churn rate equal in train and test).
        raw = []
        for t in range(n_real):
            idx = np.where(task_id == t)[0]
            Xtr, Xte, ytr, yte = train_test_split(
                X_np[idx], y[idx], test_size=self.test_size, random_state=self.seed, stratify=y[idx])
            raw.append((Xtr, Xte, ytr, yte))

        # Scale to [0, 1]. The scaler is fitted on the training rows of all tasks: we assume the
        # feature ranges (e.g. 0..max minutes) are fixed business constants. Fitting only on
        # Task 1 would push later tasks above 1 and the generator's sigmoid output cannot reach that.
        self.scaler.fit(np.vstack([r[0][:, num] for r in raw]))
        tasks = []
        for t, (Xtr, Xte, ytr, yte) in enumerate(raw):
            for A in (Xtr, Xte):
                A[:, num] = np.clip(self.scaler.transform(A[:, num]), 0.0, 1.0)
            tasks.append(Task(f"Task {t + 1}", Xtr.astype(np.float32), Xte.astype(np.float32), ytr, yte))

        summary = pd.DataFrame({
            "task": [t.name for t in tasks],
            "tenure range (months)": [f"{tenure[task_id == i].min():.0f} - {tenure[task_id == i].max():.0f}" for i in range(n_real)],
            "customers": [len(t.y_train) + len(t.y_test) for t in tasks],
            "train rows": [len(t.y_train) for t in tasks],
            "test rows": [len(t.y_test) for t in tasks],
            "churn rate": [round(float(np.concatenate([t.y_train, t.y_test]).mean()), 3) for t in tasks]})
        self._snapshot("4. TASKS (customers grouped by tenure, in the order the classifier will see them)", summary)
        self._snapshot("5. SCALED Task 1 training matrix (exactly what the classifier receives)",
                       pd.DataFrame(tasks[0].X_train, columns=self.feature_names),
                       extra="all continuous values now between 0 and 1")

        if raw_demo is not None:
            scaled = np.concatenate([t.X_train[:, raw_demo_col] for t in tasks])
            self.scaling_demo = (self.feature_names[raw_demo_col], raw_demo, scaled)

        self._log(f"[data] {len(df):,} customers | {len(self.feature_names)} features "
                  f"({len(num)} continuous, {int(is_binary.sum())} binary) | overall churn rate {y.mean():.1%}")
        for t, task in enumerate(tasks):
            n = len(task.y_train) + len(task.y_test)
            rate = np.concatenate([task.y_train, task.y_test]).mean()
            self._log(f"[data] {task.name}: {n:,} customers, churn rate {rate:.1%}")
        return tasks

    def _pick_demo_column(self):
        """Column used for the 'effect of scaling on outliers' figure."""
        for i in self.numeric_idx:
            if self.feature_names[i] == "monthlyrevenue":
                return int(i)
        return int(self.numeric_idx[0]) if len(self.numeric_idx) else None

    # ------------------------------------------------------------------ decoding
    def artifacts(self):
        """Everything needed to turn generated numbers back into readable customer records."""
        return {"feature_names": self.feature_names, "numeric_idx": self.numeric_idx,
                "numeric_mask": self.numeric_mask, "scaler": self.scaler}

    def save_artifacts(self, path):
        joblib.dump(self.artifacts(), path)

    def decode(self, X, y=None):
        return decode_features(X, y, self.artifacts())


def decode_features(X, y, artifacts):
    """Scaled array -> DataFrame in original units (dollars, minutes, ...)."""
    X = np.array(X, dtype=np.float64, copy=True)          # float64 so rounding gives 85.7, not 85.699997
    idx = artifacts["numeric_idx"]
    X[:, idx] = artifacts["scaler"].inverse_transform(X[:, idx])
    df = pd.DataFrame(X, columns=artifacts["feature_names"])
    binary_cols = [c for c, is_num in zip(artifacts["feature_names"], artifacts["numeric_mask"]) if not is_num]
    df[binary_cols] = df[binary_cols].round().astype(int)
    df = df.round(2)
    if y is not None:
        df.insert(0, "churn", np.asarray(y, dtype=int))
    return df
