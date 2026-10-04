```
# Continual Generative Replay for Telecom Customer Churn

This repository contains the prototype for predicting telecom customer churn in a continual learning setting. When the multi-layer perceptron (MLP) churn classifier is updated on newer customers, synthetic historical customers generated via a Conditional Variational Autoencoder (CVAE) are mixed into the training batch. This approach mitigates **catastrophic forgetting** without retaining raw historical data.

## Setup & Installation

Ensure Python 3.8+ is installed. Clone the repository and place the raw dataset at `data/telecomm c2c.csv`.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

## Execution Pipeline

Scripts should be executed via the CLI in the following order. Run `python main.py --help` for full argument lists.

| # | Command | Purpose |
| --- | --- | --- |
| 1 | `python main.py test` | Unit / smoke tests (uses a generated mock CSV) |
| 2 | `python main.py eda` | Generates dataset snapshots and distribution figures |
| 3 | `python main.py forgetting` | Baseline catastrophic-forgetting test (no replay) + per-epoch graph |
| 4 | `python main.py run --n-seeds 1 --vae-epochs 30` | Full pipeline execution (single seed) |
| 5 | `python main.py run --n-seeds 5` | **Final evaluation** (all strategies, 5 seeds) |
| 6 | `python main.py run --n-seeds 5 --replay-labels generator --strategies naive generative joint --out-dir outputs_generator_labels` | Ablation study: CVAE labels its own samples |
| 7 | `python main.py generate --n 20` | Generates 20 synthetic customers decoded into original features |

*Note: Matplotlib figures will render during execution and block script termination until closed. Pass `--no-show` to bypass GUI rendering and save PNGs directly to `outputs/figures/`.*

## Hyperparameter Configurations

* **3 Sequential Tasks**: Tenure is removed from the inputs (`--n-tasks 3`). Without tenure, the classifier cannot identify task origins, enforcing visible concept drift. `--keep-tenure` re-enables it.
* **Epochs**: Capped at 6 epochs per task to prevent MLP overfitting (Task-1 AUC empirically peaks around epoch 5).
* **Probability Threshold**: Fixed decision threshold locked to the Task 1 churn base rate (~0.28). Standard 0.5 thresholds on ~29% minority class datasets result in null churn predictions and meaningless F1 scores.

## Experiment Methodology

Customers are chronologically sorted into tasks by tenure (Task 1 = newest). A single classifier is trained sequentially and evaluated on all previously seen tasks after each task completion.

| Strategy | Training Data at Task 2+ | Role |
| --- | --- | --- |
| `naive` | New data only | Lower bound (demonstrates maximum forgetting) |
| `generative` | New data + CVAE samples | **Proposed Method** |
| `marginal` | New data + feature-by-feature samples | Control baseline (verifies VAE structural learning) |
| `real_replay` | New data + small real buffer | Memory-limited practical baseline |
| `joint` | New data + all historical data | Upper bound (full retraining) |

**Labeling Strategy:**
By default, the *previous classifier* labels synthetic data (`--replay-labels classifier`, aligning with Shin et al., 2017). Because it retains old weights when a new task initiates, its predictions pass historical decision boundaries to the new model. `--replay-labels generator` uses the CVAE's internal conditioning label. On this dataset, the CVAE's standalone label signal is weaker (train-on-synthetic AUC 0.57 vs 0.63 on real data), making classifier distillation the optimal approach.

**Catastrophic-Forgetting Measure:**
Calculated post-epoch for all preceding tasks: `forgetting = (score immediately after task completion) - (current score)`, averaged across historical tasks. Accuracy metrics are volatile due to class imbalance; F1 and Recall serve as the primary evaluation signals.

## Outputs (`outputs/`)

* **Figures:** `dataset_snapshots.png`, `dataset_overview.png`, `scaling_check.png`, `forgetting_measure.png`, `real_vs_synthetic.png`, `generator_training.png`.
* **Metrics & Data:** `summary.csv`, `results_matrix.csv`, `curves.csv`, `generated_customers.csv`, `config.json`, `generator_fidelity.json`, `generator.pt`.

## Repository Structure

```text
main.py                CLI router (test | eda | forgetting | run | generate)
src/dataset.py         Data ingestion, cleaning, one-hot encoding, chronological splitting, scaling
src/classifier.py      MLP classifier, metric calculations, forgetting evaluation
src/vae.py             Conditional VAE generator (PyTorch architecture)
src/baselines.py       Marginal-sampler control generator
src/replay_engine.py   Strategy execution, evaluation matrices, backward transfer logic
src/visuals.py         Matplotlib figure generation
src/test_runs.py       Unit tests and mock data generation

```

## Reproducibility

Every random step is seeded (data split `--seed`, models `--seed ... --seed+n-1`). A `config.json` file is automatically generated in the `outputs/` directory during execution to store the exact hyperparameter settings of each run, ensuring all results can be independently verified.

## Known Limitations

* Feature scaling ranges are fitted across all training rows simultaneously, assuming fixed min/max boundaries. High-cardinality categorical variables are dropped to maintain VAE efficiency.
* Features within a one-hot group are sampled independently via the VAE's Bernoulli distribution, meaning a synthetic row could theoretically activate mutually exclusive categories simultaneously.
* The classifier processes more total rows per epoch under `generative`, `real_replay`, and `joint` strategies than under `naive` training.

```

```
