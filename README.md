# Consensus Classifier

XGBoost-based ensemble classifier with bagging over samples and features, supporting binary and multiclass classification.

---

## Installation

```bash
poetry install
```

---

## Experiment Framework

A modular evaluation framework lives in `experiments/`. It benchmarks `ConsensusClassifier` against standard baselines on configurable synthetic datasets.

### Package layout

```
experiments/
  config.py              # ExperimentConfig dataclasses + YAML loader
  run.py                 # CLI entrypoint
  data/
    synthetic.py         # five dataset generators
    preprocessing.py     # imputation / optional scaling (Preprocessor)
  models/
    registry.py          # ModelWrapper interface + wrappers for all models
  eval/
    metrics.py           # compute_metrics() — full metric suite
    calibration.py       # CalibrationWrapper (none / sigmoid / isotonic)
    cv.py                # run_experiment() — repeated CV or holdout loop
  diagnostics/
    plots.py             # confusion matrix, ROC, PR, calibration, bar charts
  results/
    io.py                # save / load / aggregate fold artifacts
  configs/
    binary_benchmark.yaml
    multiclass_benchmark.yaml
    imbalanced_benchmark.yaml
```

### Run a benchmark

```bash
# from the repo root
python -m experiments.run --config experiments/configs/binary_benchmark.yaml
```

Results land in `results/binary_benchmark/`:

```
results/binary_benchmark/
  config.json
  fold_metrics.csv          ← per-fold metrics for every model
  aggregate_metrics.csv     ← mean ± std per model
  compare_roc_auc.png       ← side-by-side bar charts
  compare_f1_macro.png
  ...
  consensus/fold_000/
    predictions.csv
    probabilities.csv
    confusion_matrix.png
    roc_curve.png
    pr_curve.png
    calibration.png
    class_distribution.png
  logistic_regression/fold_000/
    ...
```

### Available datasets

| name | description |
|------|-------------|
| `balanced_binary` | Balanced 2-class benchmark |
| `imbalanced_binary` | 2-class with configurable imbalance (default 90/10) |
| `balanced_multiclass` | Balanced n-class |
| `imbalanced_multiclass` | Imbalanced n-class (geometric weights) |
| `biomedical_highdim` | High-dimensional sparse-signal (p >> n) |

### Available models

| name | class |
|------|-------|
| `consensus` | `ConsensusClassifier` |
| `logistic_regression` | `sklearn.linear_model.LogisticRegression` |
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` |
| `xgboost` | `xgboost.XGBClassifier` |

All model params are passed directly through the YAML config.

### Calibration

Set `eval.calibration` to `sigmoid` (Platt) or `isotonic` to enable post-hoc
calibration. When enabled, each fold splits training data into a fit set (80%)
and a calibration set (20%), fits the calibrator on the calibration probabilities,
then re-fits the model on full training data for final test-set predictions.

### Evaluation strategies

| strategy | description |
|----------|-------------|
| `repeated_cv` | `RepeatedStratifiedKFold(n_splits, n_repeats)` |
| `holdout` | Random stratified holdout repeated `n_repeats` times |

### Design targets

The configs are designed for the **biomedical p >> n** regime:
- **Samples**: 300–1000
- **Features**: 10 000–20 000
- **Signal**: sparse (~0.5–2% of features informative)
- **ConsensusClassifier** uses `n_features_per_model="auto"` which computes the minimum features per sub-model needed for 95% ensemble coverage via the formula `k = D · (1 − (1 − 0.95)^(1/M))`. With 15 000 features and 100 models this gives ≈ 870 features per model.

### Example: custom config

```yaml
name: rare_subtype_classifier
output_dir: results
seed: 0

data:
  name: imbalanced_multiclass
  n_samples: 400
  n_features: 20000
  n_informative: 150   # 0.75% signal
  n_classes: 4
  class_weights: [0.6, 0.2, 0.1, 0.1]
  label_noise: 0.01
  seed: 0

models:
  - name: consensus
    params:
      n_models: 100
      n_samples_per_model: 0.8
  - name: logistic_regression
    params:
      C: 0.05
      solver: saga
      penalty: l1

eval:
  strategy: repeated_cv
  n_splits: 5
  n_repeats: 3
  calibration: isotonic
  cal_size: 0.2
```

### Extending the framework

- **New dataset**: add a generator function to `experiments/data/synthetic.py` and register it in `GENERATORS`.
- **New model**: add a wrapper to `experiments/models/registry.py` and handle its name in `build_model()`.
- **New calibration method**: subclass `CalibrationWrapper` and override `_fit_one()` / `_transform_one()`.
- **New metrics**: extend `compute_metrics()` in `experiments/eval/metrics.py`.
