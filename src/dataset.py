"""Loads the churn CSV, cleans it and splits the customers into tenure based tasks."""
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CHURN_MAP = {"yes": 1, "no": 0}


@dataclass
class Task:
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
        self.max_cardinality = max_cardinality
        self.clip_quantile = clip_quantile
        self.drop_split_feature = drop_split_feature
        self.seed = seed
        self.verbose = verbose
        self.scaler = MinMaxScaler()

        self.feature_names = None
        self.numeric_idx = None
        self.numeric_mask = None
        self.boundaries = None
        self.tenure_all = None
        self.y_all = None
        self.scaling_demo = None
        self.snapshots = []

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _snapshot(self, title, df, extra="", n_rows=5, max_cols=9):
        """Keeps (and prints) the first rows of a table, cut to max_cols columns."""
        priority = [c for p in ("churn", self.split_col) for c in df.columns if str(c).lower() == p]
        cols = (priority + [c for c in df.columns if c not in priority])[:max_cols]
        table = df[cols].head(n_rows).copy()
        for c in table.select_dtypes(include="number").columns:
            table[c] = table[c].round(3)
        cut = f"showing {len(cols)} of {df.shape[1]} columns" if df.shape[1] > len(cols) else ""
        note = ", ".join(x for x in (cut, extra) if x)
        self.snapshots.append({"title": title, "shape": df.shape, "table": table, "note": note})
        if self.verbose:
            print(f"\n--- {title}  [{df.shape[0]:,} rows x {df.shape[1]} columns" + (f"; {note}" if note else "") + "] ---")
            print(table.to_string(index=False))

    def load_and_clean(self):
        df = pd.read_csv(self.filepath)
        self._snapshot("1. RAW CSV (as loaded)", df, extra=f"{int(df.isna().sum().sum()):,} missing cells")
        df.columns = df.columns.str.strip().str.lower()
        df = df.drop(columns=[c for c in ("customerid",) if c in df.columns])

        # only yes/no are accepted, anything else stops the run instead of being dropped quietly
        df = df.dropna(subset=["churn"])
        labels = df["churn"].astype(str).str.strip().str.lower()
        unexpected = set(labels.unique()) - set(CHURN_MAP)
        if unexpected:
            raise ValueError(f"Unexpected values in the churn column: {unexpected}")
        df["churn"] = labels.map(CHURN_MAP).astype(int)

        if self.split_col not in df.columns:
            raise ValueError(f"split column '{self.split_col}' not found in the CSV")
        df = df.dropna(subset=[self.split_col])

        # median for numbers, "unknown" for text
        num_cols = [c for c in df.select_dtypes(include="number").columns if c != "churn"]
        cat_cols = list(df.select_dtypes(exclude="number").columns)
        df[num_cols] = df[num_cols].fillna(df[num_cols].median()).fillna(0)
        df[cat_cols] = df[cat_cols].fillna("unknown")

        # a column with hundreds of categories would add hundreds of one-hot columns
        too_many = [c for c in cat_cols if df[c].nunique() > self.max_cardinality]
        constant = [c for c in df.columns if c != "churn" and df[c].nunique() <= 1]
        df = df.drop(columns=too_many + constant)
        if too_many:
            self._log(f"[data] dropped high-cardinality columns: {too_many}")
        if constant:
            self._log(f"[data] dropped constant columns: {constant}")
        self._snapshot("2. CLEANED (churn as 0/1, gaps filled, long categoricals dropped)", df,
                       extra=f"{int(df.isna().sum().sum())} missing cells left")
        return df

    def build_tasks(self):
        df = self.load_and_clean()
        y = df["churn"].to_numpy(dtype=np.int64)
        tenure = df[self.split_col].to_numpy(dtype=float)

        # encode before splitting so every task gets the same columns
        X = pd.get_dummies(df.drop(columns="churn"), drop_first=True, dtype=np.float32)
        if self.drop_split_feature:
            X = X.drop(columns=self.split_col)
        self.feature_names = list(X.columns)
        self._snapshot("3. ENCODED feature matrix (one-hot, before clipping and scaling)", X)
        X_np = X.to_numpy(dtype=np.float32)

        # columns that only hold 0 and 1 are binary, the rest are continuous
        is_binary = np.isin(X_np, [0, 1]).all(axis=0)
        self.numeric_mask = ~is_binary
        self.numeric_idx = np.where(~is_binary)[0]

        # clip outliers so min-max scaling is not squashed by a few huge values
        num = self.numeric_idx
        raw_demo_col = self._pick_demo_column()
        raw_demo = X_np[:, raw_demo_col].copy() if raw_demo_col is not None else None
        lo = np.quantile(X_np[:, num], 1 - self.clip_quantile, axis=0)
        hi = np.quantile(X_np[:, num], self.clip_quantile, axis=0)
        X_np[:, num] = np.clip(X_np[:, num], lo, hi)

        # tasks are slices of tenure with about the same number of customers
        task_id = pd.qcut(tenure, q=self.n_tasks, labels=False, duplicates="drop").astype(int)
        n_real = int(task_id.max()) + 1
        self.boundaries = np.quantile(tenure, np.linspace(0, 1, n_real + 1)[1:-1])
        self.tenure_all, self.y_all = tenure, y

        # train/test split inside every task, keeping the churn rate the same in both
        raw = []
        for t in range(n_real):
            idx = np.where(task_id == t)[0]
            Xtr, Xte, ytr, yte = train_test_split(
                X_np[idx], y[idx], test_size=self.test_size, random_state=self.seed, stratify=y[idx])
            raw.append((Xtr, Xte, ytr, yte))

        # the scaler is fitted on the training rows of all tasks so every value stays between 0 and 1
        # (the generator uses a sigmoid output and cannot produce values above 1)
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
        self._snapshot("4. TASKS (customers grouped by tenure, in the order they are learned)", summary)
        self._snapshot("5. SCALED Task 1 training matrix (what the classifier receives)",
                       pd.DataFrame(tasks[0].X_train, columns=self.feature_names),
                       extra="continuous values between 0 and 1")

        if raw_demo is not None:
            scaled = np.concatenate([t.X_train[:, raw_demo_col] for t in tasks])
            self.scaling_demo = (self.feature_names[raw_demo_col], raw_demo, scaled)

        self._log(f"[data] {len(df):,} customers | {len(self.feature_names)} features "
                  f"({len(num)} continuous, {int(is_binary.sum())} binary) | overall churn rate {y.mean():.1%}")
        for task in tasks:
            n = len(task.y_train) + len(task.y_test)
            rate = np.concatenate([task.y_train, task.y_test]).mean()
            self._log(f"[data] {task.name}: {n:,} customers, churn rate {rate:.1%}")
        return tasks

    def _pick_demo_column(self):
        for i in self.numeric_idx:
            if self.feature_names[i] == "monthlyrevenue":
                return int(i)
        return int(self.numeric_idx[0]) if len(self.numeric_idx) else None

    def artifacts(self):
        return {"feature_names": self.feature_names, "numeric_idx": self.numeric_idx,
                "numeric_mask": self.numeric_mask, "scaler": self.scaler}

    def save_artifacts(self, path):
        joblib.dump(self.artifacts(), path)

    def decode(self, X, y=None):
        return decode_features(X, y, self.artifacts())


def decode_features(X, y, artifacts):
    """Turns scaled values back into the original units (dollars, minutes, ...)."""
    X = np.array(X, dtype=np.float64, copy=True)
    idx = artifacts["numeric_idx"]
    X[:, idx] = artifacts["scaler"].inverse_transform(X[:, idx])
    df = pd.DataFrame(X, columns=artifacts["feature_names"])
    binary_cols = [c for c, is_num in zip(artifacts["feature_names"], artifacts["numeric_mask"]) if not is_num]
    df[binary_cols] = df[binary_cols].round().astype(int)
    df = df.round(2)
    if y is not None:
        df.insert(0, "churn", np.asarray(y, dtype=int))
    return df
