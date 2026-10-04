"""
main.py - command-line interface.

    python main.py test                     # smoke tests (run these first)
    python main.py eda                      # data snapshots + dataset figures only (fast)
    python main.py forgetting               # stand-alone catastrophic-forgetting test (no replay) + per-epoch graph
    python main.py run                      # full experiment: train + compare all strategies
    python main.py generate --n 20          # generation workflow: sample customers from the saved CVAE

Add --help after any command to see its hyper-parameters. Figures pop up on screen and are always saved to
outputs/figures/ ; add --no-show if you only want the files.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from src import visuals
from src.classifier import forgetting_measure, run_forgetting_demo
from src.dataset import DataProcessing, decode_features
from src.replay_engine import (STRATEGIES, Config, ReplayEngine, aggregate, format_table, summarise)


def add_data_args(p):
    p.add_argument("--data", default="data/telecomm c2c.csv", help="path to the churn CSV")
    p.add_argument("--n-tasks", type=int, default=3, help="how many tenure-based tasks to cut the data into")
    p.add_argument("--keep-tenure", action="store_true",
                   help="keep tenure as an input feature (default: remove it, so the classifier cannot "
                        "tell which task a customer belongs to)")
    p.add_argument("--clip-quantile", type=float, default=0.99, help="winsorise continuous features at this quantile")
    p.add_argument("--max-cardinality", type=int, default=20, help="drop categorical columns with more levels")
    p.add_argument("--seed", type=int, default=42, help="seed for the data split (and first model seed)")
    p.add_argument("--out-dir", default="outputs", help="where results and figures are written")
    p.add_argument("--no-show", action="store_true", help="do not pop up figure windows (PNGs are always saved)")


def add_classifier_args(p):
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 32], help="classifier hidden layer sizes")
    p.add_argument("--clf-lr", type=float, default=0.01)
    p.add_argument("--clf-epochs", type=int, default=6, help="classifier epochs per task")
    p.add_argument("--clf-alpha", type=float, default=1e-4, help="classifier L2 regularisation")
    p.add_argument("--threshold", type=float, default=None,
                   help="decision threshold for churn (default: churn base rate of Task 1)")


def add_model_args(p):
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    p.add_argument("--n-seeds", type=int, default=3, help="repeat every strategy with this many model seeds")
    add_classifier_args(p)
    p.add_argument("--vae-hidden", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--vae-lr", type=float, default=1e-3)
    p.add_argument("--vae-epochs", type=int, default=60, help="CVAE epochs per task")
    p.add_argument("--beta", type=float, default=1.0, help="weight of the KL term")
    p.add_argument("--num-weight", type=float, default=5.0, help="weight of continuous-feature reconstruction")
    p.add_argument("--warmup-epochs", type=int, default=10, help="KL warm-up length")
    p.add_argument("--vae-batch-size", type=int, default=256)
    p.add_argument("--replay-labels", choices=("classifier", "generator"), default="classifier",
                   help="who labels the synthetic customers: the previous classifier (default) or the CVAE itself")
    p.add_argument("--replay-ratio", type=float, default=1.0, help="synthetic samples per real old sample")
    p.add_argument("--buffer-size", type=int, default=500, help="rows kept by the real_replay baseline")
    p.add_argument("--verbose", action="store_true", help="print metrics after every epoch")


def build_parser():
    parser = argparse.ArgumentParser(description="Continual generative replay for telecom churn classification")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test", help="run the smoke tests")
    add_data_args(sub.add_parser("eda", help="data snapshots and dataset figures"))
    forget = sub.add_parser("forgetting", help="catastrophic-forgetting test without replay")
    add_data_args(forget)
    add_classifier_args(forget)
    run = sub.add_parser("run", help="train and compare all strategies")
    add_data_args(run)
    add_model_args(run)
    gen = sub.add_parser("generate", help="sample synthetic customers from a saved generator")
    gen.add_argument("--checkpoint", default="outputs/generator.pt")
    gen.add_argument("--preprocessor", default="outputs/preprocessor.joblib")
    gen.add_argument("--n", type=int, default=20, help="number of customers to generate")
    gen.add_argument("--churn-rate", type=float, default=None,
                     help="force this share of churners (default: the rate the generator was trained on)")
    gen.add_argument("--out", default="outputs/generated_customers.csv")
    return parser


def make_processor(args):
    return DataProcessing(args.data, n_tasks=args.n_tasks, clip_quantile=args.clip_quantile,
                          max_cardinality=args.max_cardinality, drop_split_feature=not args.keep_tenure,
                          seed=args.seed)


def data_stage(args, show):
    """Build the tasks, then show + save the dataset snapshots and dataset figures."""
    fig_dir = Path(args.out_dir) / "figures"
    snap_dir = fig_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    proc = make_processor(args)
    tasks = proc.build_tasks()
    visuals.save_snapshot_images(proc.snapshots, snap_dir)                      # one PNG + CSV per stage
    visuals.plot_snapshots(proc.snapshots, fig_dir / "dataset_snapshots.png", show)   # all stages in one window
    visuals.plot_dataset_overview(proc, tasks, fig_dir / "dataset_overview.png", show)
    visuals.plot_scaling_check(proc, fig_dir / "scaling_check.png", show)
    return proc, tasks, fig_dir


# ---------------------------------------------------------------------- commands
def cmd_test(_args):
    from src import test_runs
    test_runs.run_all()


def cmd_eda(args):
    show = not args.no_show
    _, _, fig_dir = data_stage(args, show)
    print(f"\nFigures and snapshots written to {fig_dir}/")
    visuals.keep_open(show)


def cmd_forgetting(args):
    run_forgetting_demo(args.data, n_tasks=args.n_tasks, keep_tenure=args.keep_tenure, epochs=args.clf_epochs,
                        hidden_layers=tuple(args.hidden_layers), lr=args.clf_lr, alpha=args.clf_alpha,
                        threshold=args.threshold, seed=args.seed, out_dir=args.out_dir, show=not args.no_show)


def cmd_run(args):
    show = not args.no_show
    out = Path(args.out_dir)
    started = time.time()

    print("=" * 78 + "\n 1. DATA\n" + "=" * 78)
    proc, tasks, fig_dir = data_stage(args, show)
    if len(tasks) < 2:
        raise SystemExit("Need at least 2 tasks - use a larger --n-tasks or a different split column.")

    cfg = Config(hidden_layers=tuple(args.hidden_layers), clf_lr=args.clf_lr, clf_epochs=args.clf_epochs,
                 clf_alpha=args.clf_alpha, threshold=args.threshold,
                 vae_hidden=args.vae_hidden, latent_dim=args.latent_dim, vae_lr=args.vae_lr,
                 vae_epochs=args.vae_epochs, beta=args.beta, num_weight=args.num_weight,
                 warmup_epochs=args.warmup_epochs, vae_batch_size=args.vae_batch_size,
                 replay_ratio=args.replay_ratio, replay_labels=args.replay_labels,
                 buffer_size=args.buffer_size, verbose=args.verbose)
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    (out / "config.json").write_text(json.dumps({"config": cfg.__dict__, "args": vars(args), "seeds": seeds}, indent=2))

    print("\n" + "=" * 78 + f"\n 2. TRAINING  strategies={args.strategies}  seeds={seeds}\n" + "=" * 78)
    engine = ReplayEngine(tasks, proc.numeric_mask, cfg)
    print(f" decision threshold for churn = {engine.threshold:.3f} (same for every strategy)")
    try:
        res = engine.run_all(args.strategies, seeds)
    except ImportError as err:
        raise SystemExit(f"{err}\nThe 'generative' strategy needs PyTorch:  pip install torch") from err

    print("\n" + "=" * 78 + "\n 3. RESULTS (mean +- std over seeds)\n" + "=" * 78)
    per_seed = summarise(res["final"], len(tasks))
    agg = aggregate(per_seed)
    print(format_table(agg).to_string())
    print("\n avg_* = mean over all tasks after the last task | bwt_* = backward transfer (0 = no forgetting,"
          "\n negative = forgetting) | t1_* = final score on the OLDEST task")
    forget = forgetting_measure(res["curves"])
    res["final"].to_csv(out / "results_matrix.csv", index=False)
    res["curves"].to_csv(out / "curves.csv", index=False)
    forget.to_csv(out / "forgetting_per_epoch.csv", index=False)
    per_seed.to_csv(out / "summary_per_seed.csv", index=False)
    agg.to_csv(out / "summary.csv")
    visuals.plot_forgetting_measure(forget, cfg.clf_epochs, fig_dir / "forgetting_measure.png", show)
    visuals.plot_forgetting_curves(res["curves"], cfg.clf_epochs, fig_dir / "forgetting_curves.png", 0, show)
    visuals.plot_summary(agg, fig_dir / "summary_bars.png", show)

    gen = engine.generators.get("generative")
    if gen is not None:
        print("\n" + "=" * 78 + "\n 4. GENERATOR QUALITY & OUTPUTS\n" + "=" * 78)
        fidelity = engine.evaluate_generator(gen, seeds[0])
        storage = engine.storage_report(gen)
        (out / "generator_fidelity.json").write_text(json.dumps({**fidelity, **storage}, indent=2))
        for k, v in {**fidelity, **storage}.items():
            print(f" {k:<24} {v:.4f}")
        print(" (tstr = train on synthetic, test on real; trtr = train on real, test on real)")

        gen.save(out / "generator.pt")
        proc.save_artifacts(out / "preprocessor.joblib")
        X_syn, y_syn = gen.sample(5000)
        X_real = np.vstack([t.X_train for t in tasks])
        generated = proc.decode(X_syn[:30], y_syn[:30])
        generated.to_csv(out / "generated_customers.csv", index=False)
        print("\n First generated customers (original units):")
        print(generated.iloc[:5, :8].to_string(index=False))
        visuals.plot_real_vs_synthetic(X_real, X_syn, proc.feature_names, proc.numeric_idx,
                                       fig_dir / "real_vs_synthetic.png", show=show)
        visuals.plot_generator_training(res["gen_history"], fig_dir / "generator_training.png", show)

    print(f"\nDone in {time.time() - started:.0f}s. Everything is in {out}/")
    visuals.keep_open(show)


def cmd_generate(args):
    from src.vae import ConditionalVAEGenerator
    gen = ConditionalVAEGenerator.load(args.checkpoint)
    artifacts = joblib.load(args.preprocessor)
    y = None
    if args.churn_rate is not None:
        y = (np.random.default_rng().random(args.n) < args.churn_rate).astype(np.int64)
    X, y = gen.sample(args.n, y)
    df = decode_features(X, y, artifacts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Generated {args.n} customers ({y.mean():.0%} churners) -> {args.out}")
    print(df.iloc[:, :8].head(10).to_string(index=False))


def main():
    args = build_parser().parse_args()
    {"test": cmd_test, "eda": cmd_eda, "forgetting": cmd_forgetting, "run": cmd_run,
     "generate": cmd_generate}[args.command](args)


if __name__ == "__main__":
    main()
