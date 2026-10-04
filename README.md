# Continual Generative Replay for Telecom Customer Churn

When the churn classifier is updated on newer customers, synthetic "old" customers from the VAE are
mixed in, which reduces **catastrophic forgetting** without storing the old data.

## Quick start in Visual Studio Code
1. **File > Open Folder...** and choose `telecomm_ai`. Put the CSV at `data/telecomm c2c.csv`.
2. Open a terminal: **Ctrl + `** (View > Terminal).
3. One-time setup:
```
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell.  (cmd: .venv\Scripts\activate.bat | macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
   If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
   Then **Ctrl+Shift+P > "Python: Select Interpreter"** and pick the `.venv` one.
4. Run the steps below **in this order**. You can also press **F5** and pick the same step from the dropdown (`.vscode/launch.json`).

## Run order
| # | command | what you get |
|---|---|---|
| 1 | `python main.py test` | smoke tests (about 15 s, uses a fake CSV) |
| 2 | `python main.py eda` | dataset snapshots at every stage + dataset figures |
| 3 | `python main.py forgetting` | catastrophic-forgetting test (no replay) + per-epoch forgetting graph |
| 4 | `python main.py run --n-seeds 1 --vae-epochs 30` | quick full run to confirm everything works |
| 5 | `python main.py run --n-seeds 5` | **FINAL results** (all strategies, 5 seeds) |
| 5b | `python main.py run --n-seeds 5 --replay-labels generator --strategies naive generative joint --out-dir outputs_generator_labels` | ablation: the CVAE labels its own samples |
| 6 | `python main.py generate --n 20` | generation workflow: 20 synthetic customers in real units |

Figures open in windows while the program runs and **stay open at the end: close them to finish**.
They are always saved too (`outputs/figures/`). Add `--no-show` to skip the windows. `--help` lists all hyper-parameters.

## Defaults and why
* **3 tasks, tenure removed from the inputs** (`--n-tasks 3`; `--keep-tenure` brings it back). Without tenure the classifier cannot
  tell which task a customer belongs to, so forgetting is visible.
* **6 epochs per task**: with more, the classifier overfits (Task-1 AUC peaks around epoch 5).
* **One fixed decision threshold** = churn base rate of Task 1 (~0.28), the same for every strategy; at 0.5 a model trained on ~29% churners
  predicts almost no churn and F1 is meaningless (`--threshold` overrides).

## What the experiment does
Customers are sorted into tasks by tenure (Task 1 = newest). One classifier is trained through the tasks in order, then tested on every task seen so far.

| strategy | training data at Task 2+ | role |
|---|---|---|
| `naive` | new data only | lower bound (shows forgetting) |
| `generative` | new data + CVAE samples | **proposed method** |
| `marginal` | new data + feature-by-feature samples | control: does the VAE learn real structure? |
| `real_replay` | new data + small real buffer | memory-limited realistic baseline |
| `joint` | new data + all old data | upper bound (full retraining) |

**Who labels the synthetic customers?** By default the *previous classifier* does (`--replay-labels classifier`, as in Shin et al., 2017):
it holds the old weights when the new task starts, so its predictions hand the old decision rule to the new model. `--replay-labels generator`
uses the CVAE's own conditioning label instead. On this dataset the CVAE's label signal is weak (train-on-synthetic AUC 0.57 vs 0.63 on real data),
so generator labels dilute training, while classifier labels do not.

**Catastrophic-forgetting measure** (`classifier.forgetting_measure`): after every epoch, for each earlier task,
`forgetting = score right after learning that task - score now`, averaged over earlier tasks. 0 = nothing lost, higher = more forgotten.
Accuracy can *rise* while F1 falls (the model predicts "churn" less often), so judge forgetting mainly by F1 and recall.
Other metrics: Accuracy, Precision, Recall, F1, AUC, backward transfer, and generator quality (TSTR/TRTR AUC, correlation error, memory footprint).

## Outputs (`outputs/`)
* `figures/dataset_snapshots.png` (all stages), `figures/snapshots/snapshot_1..5.png/.csv`, `dataset_overview.png`, `scaling_check.png`
* `figures/forgetting_measure.png` (the per-epoch forgetting graph), `forgetting_curves.png`, `summary_bars.png`
* `figures/real_vs_synthetic.png`, `generator_training.png`, `generated_customers.csv`
* `summary.csv`, `results_matrix.csv`, `curves.csv`, `forgetting_per_epoch.csv`, `config.json`, `generator_fidelity.json`, `generator.pt`

## Layout
```
main.py                 CLI (test | eda | forgetting | run | generate)
src/dataset.py          load, clean, encode, split into tasks, scale, snapshots
src/classifier.py       MLP classifier, metrics, forgetting measure, stand-alone forgetting test
src/vae.py              conditional VAE generator (PyTorch)
src/baselines.py        marginal-sampler control generator
src/replay_engine.py    strategies, evaluation matrix, backward transfer, generator fidelity
src/visuals.py          all figures
src/test_runs.py        smoke tests
.vscode/launch.json     F5 run configurations
```

## Reproducibility and source control
Every random step is seeded (data split `--seed`, models `--seed ... --seed+n-1`); `config.json` stores the exact settings of each run.
```
git init
git add .
git commit -m "Pipeline, CVAE, replay engine, CLI"
git branch -M main
git remote add origin https://github.com/VkidQuintin/TelcoGenerativeAIClassicModelEnhancer
git push -u origin main
```

## Known limitations
* Scaling ranges are fitted on all training rows (assumes fixed feature ranges); high-cardinality categoricals are dropped.
* Features inside one one-hot group are generated independently, so a synthetic row can switch on two categories at once.
* The classifier sees more rows per epoch under replay/joint than under naive training.
