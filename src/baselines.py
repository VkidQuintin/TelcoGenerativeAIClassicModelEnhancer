"""
baselines.py - a deliberately simple generator used as a CONTROL experiment.

MarginalGenerator samples every feature independently, given the class (churn / no churn).
It knows each feature's mean/spread but nothing about how features relate to each other.
If the VAE beats it, we have evidence the VAE really learned customer structure
(and is not just reproducing the average customer).
"""
import numpy as np


class MarginalGenerator:
    name = "marginal"

    def __init__(self, input_dim, numeric_mask, seed=0, **_ignored):
        self.input_dim = input_dim
        self.numeric_mask = np.asarray(numeric_mask, dtype=bool)
        self.rng = np.random.default_rng(seed)
        self.prior = 0.5
        self.mean = np.zeros((2, input_dim), dtype=np.float32)
        self.std = np.zeros((2, input_dim), dtype=np.float32)

    def fit(self, X, y, epochs=None):
        self.prior = float(np.mean(y))
        for c in (0, 1):
            Xc = X[y == c] if np.any(y == c) else X
            self.mean[c] = Xc.mean(axis=0)
            self.std[c] = Xc.std(axis=0)
        return []                                  # nothing to log (no training loop)

    def sample(self, n, y=None):
        if y is None:
            y = (self.rng.random(n) < self.prior).astype(np.int64)
        y = np.asarray(y, dtype=np.int64)
        mu, sd = self.mean[y], self.std[y]
        continuous = np.clip(mu + sd * self.rng.standard_normal(mu.shape), 0.0, 1.0)
        binary = (self.rng.random(mu.shape) < mu).astype(np.float32)   # mean of a 0/1 column = P(1)
        X = np.where(self.numeric_mask, continuous, binary).astype(np.float32)
        return X, y

    def n_params(self):
        return int(self.mean.size + self.std.size)

    def size_bytes(self):
        return 4 * self.n_params()
