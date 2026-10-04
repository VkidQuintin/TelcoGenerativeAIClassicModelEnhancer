"""
test_runs.py - smoke tests. Run them BEFORE the long experiment:

    python main.py test            (or)    python -m unittest src.test_runs -v

They build a small fake telecom CSV (so they work without the real data) and check that every
stage of the pipeline runs and behaves sensibly.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import MarginalGenerator
from .classifier import ChurnClassifier, forgetting_measure, run_forgetting_demo
from . import visuals
from .dataset import DataProcessing
from .replay_engine import Config, ReplayEngine, aggregate, summarise

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def make_demo_csv(path, n=4000, seed=0):
    """Fake Cell2Cell-style table with drift: new and old customers churn for different reasons."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 60, n)
    df = pd.DataFrame({
        "CustomerID": np.arange(n),
        "MonthlyRevenue": rng.gamma(2, 20, n),
        "MonthlyMinutes": rng.gamma(3, 150, n),
        "OverageMinutes": rng.exponential(10, n) * (rng.random(n) < 0.3),
        "DroppedCalls": rng.poisson(5, n),
        "MonthsInService": tenure,
        "CreditRating": rng.choice(["1-Highest", "2-High", "3-Good", "4-Medium"], n),
        "ServiceArea": rng.choice([f"AREA{i}" for i in range(80)], n),     # high cardinality -> dropped
        "HandsetWebCapable": rng.choice(["Yes", "No"], n),
    })
    old = tenure > 24
    logit = np.where(old, 0.06 * df.OverageMinutes - 0.002 * df.MonthlyMinutes + 0.2,
                     0.15 * df.DroppedCalls - 0.02 * df.MonthlyRevenue - 0.2)
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    df.loc[rng.random(n) < 0.02, "MonthlyRevenue"] = np.nan                  # some missing values
    df.to_csv(path, index=False)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.csv = Path(cls.tmp.name) / "demo.csv"
        make_demo_csv(cls.csv)
        cls.proc = DataProcessing(cls.csv, n_tasks=2, verbose=False)
        cls.tasks = cls.proc.build_tasks()
        cls.cfg = Config(clf_epochs=3, vae_epochs=3, buffer_size=100)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- data
    def test_tasks_are_consistent(self):
        self.assertEqual(len(self.tasks), 2)
        dims = {t.X_train.shape[1] for t in self.tasks} | {t.X_test.shape[1] for t in self.tasks}
        self.assertEqual(len(dims), 1, "all tasks must share the same feature columns")
        for t in self.tasks:
            self.assertFalse(np.isnan(t.X_train).any())
            self.assertTrue(set(np.unique(t.y_train)) <= {0, 1})
            self.assertGreaterEqual(t.X_train.min(), 0.0)
            self.assertLessEqual(t.X_train.max(), 1.0)

    def test_high_cardinality_column_dropped(self):
        self.assertFalse(any(n.startswith("servicearea") for n in self.proc.feature_names))

    def test_decode_returns_original_units(self):
        X = self.tasks[0].X_train[:5]
        df = self.proc.decode(X, self.tasks[0].y_train[:5])
        self.assertEqual(len(df), 5)
        self.assertIn("churn", df.columns)
        self.assertGreater(df["monthlyrevenue"].max(), 1.0)       # dollars, no longer 0..1

    def test_snapshots_recorded_for_every_stage(self):
        titles = [snap["title"][0] for snap in self.proc.snapshots]
        self.assertEqual(titles, ["1", "2", "3", "4", "5"])
        for snap in self.proc.snapshots:
            self.assertGreater(len(snap["table"]), 0)
            self.assertLessEqual(snap["table"].shape[1], 9)
        out = Path(self.tmp.name) / "snaps"
        out.mkdir(exist_ok=True)
        visuals.plot_snapshots(self.proc.snapshots, out / "all.png", show=False)
        visuals.save_snapshot_images(self.proc.snapshots, out)
        self.assertTrue((out / "all.png").exists() and (out / "snapshot_5.png").exists())

    # ---- generators / classifier
    def test_marginal_generator(self):
        X, y = self.tasks[0].X_train, self.tasks[0].y_train
        gen = MarginalGenerator(X.shape[1], self.proc.numeric_mask, seed=1)
        gen.fit(X, y)
        Xs, ys = gen.sample(200)
        self.assertEqual(Xs.shape, (200, X.shape[1]))
        binary = Xs[:, ~self.proc.numeric_mask]
        self.assertTrue(np.isin(binary, [0, 1]).all())
        self.assertTrue(((Xs >= 0) & (Xs <= 1)).all())

    def test_classifier_beats_chance(self):
        t = self.tasks[0]
        clf = ChurnClassifier(seed=0)
        clf.fit(t.X_train, t.y_train, epochs=10)
        self.assertGreater(clf.evaluate(t.X_test, t.y_test)["auc"], 0.55)

    # ---- engine
    def _engine(self, strategies, seeds=(0,)):
        eng = ReplayEngine(self.tasks, self.proc.numeric_mask, self.cfg, log=lambda *_: None)
        return eng, eng.run_all(strategies, seeds)

    def test_engine_baselines_and_summary(self):
        _, res = self._engine(["naive", "marginal", "real_replay", "joint"])
        summary = summarise(res["final"], len(self.tasks))
        self.assertEqual(set(summary["strategy"]), {"naive", "marginal", "real_replay", "joint"})
        self.assertTrue(np.isfinite(summary[["avg_auc", "bwt_auc", "t1_f1"]].to_numpy()).all())
        # evaluation matrix is lower-triangular: after task i we evaluate tasks 0..i
        self.assertTrue((res["final"]["eval_task"] <= res["final"]["after_task"]).all())
        self.assertEqual(aggregate(summary).shape[0], 4)

    def test_forgetting_measure_and_plot(self):
        _, res = self._engine(["naive", "joint"])
        forget = forgetting_measure(res["curves"])
        epochs = self.cfg.clf_epochs
        first_task = forget[forget["global_step"] <= epochs]
        self.assertTrue((first_task[["forget_acc", "forget_f1", "forget_auc"]] == 0).all().all())
        self.assertEqual(forget["global_step"].max(), epochs * len(self.tasks))
        path = Path(self.tmp.name) / "forget.png"
        visuals.plot_forgetting_measure(forget, epochs, path, show=False)
        self.assertTrue(path.exists())

    def test_forgetting_demo_runs(self):
        out = Path(self.tmp.name) / "demo_out"
        out.mkdir(exist_ok=True)
        forget = run_forgetting_demo(self.csv, n_tasks=2, epochs=2, out_dir=out, show=False)
        self.assertTrue((out / "figures" / "forgetting_demo.png").exists())
        self.assertFalse(forget.empty)

    def test_replay_labels_option(self):
        """Both labelling modes must run; they differ only in who labels the synthetic customers."""
        out = {}
        for mode in ("generator", "classifier"):
            eng = ReplayEngine(self.tasks, self.proc.numeric_mask,
                               Config(clf_epochs=3, vae_epochs=3, replay_labels=mode), log=lambda *_: None)
            out[mode] = eng.run_all(["marginal"], [0])["final"]
            self.assertEqual(len(out[mode]), 3)
        # same data before Task 2, so the Task-1 row is identical; after replay the results must differ
        self.assertFalse(out["generator"].equals(out["classifier"]))

    def test_same_seed_same_result(self):
        _, a = self._engine(["marginal"])
        _, b = self._engine(["marginal"])
        pd.testing.assert_frame_equal(a["final"], b["final"])

    # ---- VAE (skipped automatically when PyTorch is not installed)
    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_vae_fit_sample_save_load(self):
        from .vae import ConditionalVAEGenerator
        X, y = self.tasks[0].X_train, self.tasks[0].y_train
        gen = ConditionalVAEGenerator(X.shape[1], self.proc.numeric_mask, hidden_dim=32, latent_dim=4,
                                      warmup_epochs=2, seed=0)
        hist = gen.fit(X, y, epochs=6)
        self.assertLess(hist[-1]["recon_num"], hist[0]["recon_num"] * 1.05)
        Xs, ys = gen.sample(100)
        self.assertEqual(Xs.shape, (100, X.shape[1]))
        self.assertTrue(((Xs >= 0) & (Xs <= 1)).all())
        self.assertTrue(np.isin(Xs[:, ~self.proc.numeric_mask], [0, 1]).all())
        path = Path(self.tmp.name) / "gen.pt"
        gen.save(path)
        again = ConditionalVAEGenerator.load(path)
        self.assertAlmostEqual(again.prior, gen.prior)
        self.assertEqual(again.n_params(), gen.n_params())

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_engine_with_generative_replay(self):
        eng, res = self._engine(["generative"])
        self.assertEqual(len(res["final"]), 3)                     # 2 tasks -> 1 + 2 evaluations
        self.assertFalse(res["gen_history"].empty)
        fid = eng.evaluate_generator(eng.generators["generative"])
        self.assertIn("tstr_auc_task1", fid)


def run_all():
    unittest.main(module=__name__, argv=["tests", "-v"], exit=False)


if __name__ == "__main__":
    run_all()
