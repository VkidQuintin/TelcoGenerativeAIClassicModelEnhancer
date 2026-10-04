"""Control generator that samples every feature on its own, given the class."""
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
        return []

    def sample(self, n, y=None):
        if y is None:
            y = (self.rng.random(n) < self.prior).astype(np.int64)
        y = np.asarray(y, dtype=np.int64)
        mu, sd = self.mean[y], self.std[y]
        continuous = np.clip(mu + sd * self.rng.standard_normal(mu.shape), 0.0, 1.0)
        # the mean of a 0/1 column is the probability of a 1
        binary = (self.rng.random(mu.shape) < mu).astype(np.float32)
        X = np.where(self.numeric_mask, continuous, binary).astype(np.float32)
        return X, y

    def n_params(self):
        return int(self.mean.size + self.std.size)

    def size_bytes(self):
        return 4 * self.n_params()
