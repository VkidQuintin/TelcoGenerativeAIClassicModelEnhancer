"""
vae.py - Conditional Variational Auto-Encoder (CVAE): the generative memory.

Idea
----
Instead of storing old customers, we store a small neural network that has learned
"what old customers look like". When the classifier needs to be updated on new data, we ask
the network to dream up synthetic old customers and mix them with the new real ones.

How it works (Kingma & Welling, 2014; conditional version: Sohn et al., 2015)
-----------------------------------------------------------------------------
* Encoder:  (customer features x, label y)  ->  mean mu and log-variance logvar of a small
            Gaussian "code" z (the latent vector).
* Decoder:  (z, label y)  ->  reconstructed features.
* Training minimises  reconstruction error + beta * KL divergence.  The KL term pulls the codes
  towards N(0, I), which is what lets us SAMPLE new customers later: draw z ~ N(0, I), pick a label,
  and run the decoder.
* The label y is fed to both encoder and decoder ("conditional"), so we can ask for churners or
  non-churners explicitly and the synthetic data comes with a trustworthy label.

Tabular-data tweaks
-------------------
* Continuous columns (scaled to 0..1): sigmoid output + squared error.
* Binary columns (one-hot / flags): sigmoid output + binary cross-entropy; at sampling time we draw
  Bernoulli(p) instead of rounding, so rare categories do not vanish.
* KL warm-up: beta grows from 0 to its target over the first epochs, which stops the model from
  ignoring the latent code ("posterior collapse", Bowman et al., 2016).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, input_dim, numeric_mask, n_classes=2, hidden_dim=128, latent_dim=16):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        # A buffer is a tensor that is saved with the model but is not trained.
        self.register_buffer("numeric_mask", torch.as_tensor(np.asarray(numeric_mask, dtype=bool)))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + n_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))             # raw scores ("logits"), sigmoid applied later

    def encode(self, x, y_onehot):
        h = self.encoder(torch.cat([x, y_onehot], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu, logvar):
        """z = mu + sigma * eps  with eps ~ N(0, I). Written this way so gradients can flow through."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y_onehot):
        return self.decoder(torch.cat([z, y_onehot], dim=1))

    def forward(self, x, y_onehot):
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterise(mu, logvar)
        return self.decode(z, y_onehot), mu, logvar

    def loss(self, logits, x, mu, logvar, beta, num_weight):
        """Per-sample loss averaged over the batch. Returns (total, recon_numeric, recon_binary, kl)."""
        num = self.numeric_mask
        recon_num = F.mse_loss(torch.sigmoid(logits[:, num]), x[:, num], reduction="none").sum(dim=1)
        recon_bin = F.binary_cross_entropy_with_logits(logits[:, ~num], x[:, ~num], reduction="none").sum(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        total = num_weight * recon_num + recon_bin + beta * kl
        return total.mean(), recon_num.mean(), recon_bin.mean(), kl.mean()


class ConditionalVAEGenerator:
    """Training / sampling wrapper around CVAE. Same interface as baselines.MarginalGenerator."""
    name = "vae"

    def __init__(self, input_dim, numeric_mask, hidden_dim=128, latent_dim=16, lr=1e-3, beta=1.0,
                 num_weight=5.0, warmup_epochs=10, batch_size=256, seed=0, n_classes=2):
        torch.manual_seed(seed)
        # Kept so that save()/load() can rebuild an identical model.
        self.init_args = dict(input_dim=int(input_dim), numeric_mask=[bool(v) for v in numeric_mask],
                              hidden_dim=hidden_dim, latent_dim=latent_dim, lr=lr, beta=beta,
                              num_weight=num_weight, warmup_epochs=warmup_epochs,
                              batch_size=batch_size, seed=seed, n_classes=n_classes)
        self.model = CVAE(input_dim, numeric_mask, n_classes, hidden_dim, latent_dim)
        self.lr, self.beta, self.num_weight = lr, beta, num_weight
        self.warmup_epochs, self.batch_size, self.n_classes = warmup_epochs, batch_size, n_classes
        self.prior = 0.5            # fraction of churners in the data it was last trained on

    def fit(self, X, y, epochs):
        """(Continue to) train on (X, y). Returns one dict of loss values per epoch (telemetry)."""
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.prior = float(y_t.float().mean())

        self.model.train()
        history = []
        for epoch in range(1, epochs + 1):
            beta = self.beta * min(1.0, epoch / max(1, self.warmup_epochs))     # KL warm-up
            sums, seen = np.zeros(4), 0
            order = torch.randperm(len(X_t))                       # new random order every epoch
            for start in range(0, len(X_t), self.batch_size):      # slicing tensors is much faster than a DataLoader
                idx = order[start:start + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                y_onehot = F.one_hot(yb, self.n_classes).float()
                logits, mu, logvar = self.model(xb, y_onehot)
                loss, r_num, r_bin, kl = self.model.loss(logits, xb, mu, logvar, beta, self.num_weight)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                n = xb.size(0)
                sums += np.array([loss.item(), r_num.item(), r_bin.item(), kl.item()]) * n
                seen += n
            mean = sums / seen
            history.append({"epoch": epoch, "beta": beta, "loss": mean[0],
                            "recon_num": mean[1], "recon_bin": mean[2], "kl": mean[3]})
        return history

    @torch.no_grad()
    def sample(self, n, y=None):
        """Generate n synthetic customers. If y is None, labels are drawn with the training churn rate."""
        self.model.eval()
        if y is None:
            y_t = (torch.rand(n) < self.prior).long()
        else:
            y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
        z = torch.randn(len(y_t), self.model.latent_dim)
        probs = torch.sigmoid(self.model.decode(z, F.one_hot(y_t, self.n_classes).float()))
        # continuous columns keep the probability value; binary columns are sampled as 0/1
        X = torch.where(self.model.numeric_mask, probs, torch.bernoulli(probs))
        return X.numpy().astype(np.float32), y_t.numpy().astype(np.int64)

    def n_params(self):
        return int(sum(p.numel() for p in self.model.parameters()))

    def size_bytes(self):
        return 4 * self.n_params()                  # float32 = 4 bytes

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(), "init_args": self.init_args,
                    "prior": self.prior}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        gen = cls(**ckpt["init_args"])
        gen.model.load_state_dict(ckpt["state_dict"])
        gen.prior = ckpt["prior"]
        return gen
