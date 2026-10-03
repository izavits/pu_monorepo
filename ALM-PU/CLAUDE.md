# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This repository runs ALM-PU on 14 tabular benchmark datasets, using a small MLP. ALM-PU is a Positive-Unlabeled (PU) learning method trained with an Augmented Lagrangian Method. The runs use the same setting as `main_LBE.py` in the sibling `../iFPNU` project, so the results can be compared with the methods evaluated there.

The original image-data code was removed after commit `ffe157e`: the CIFAR, STL-10, FashionMNIST and Alzheimer pipelines, the ResNet, CNN and LeNet models, `main.py`, `train.py`, and the credit-card scripts. Recover them from git history if needed.

There is no build step, linter, or test suite.

## Environment and commands

Dependencies (torch, numpy, pandas, scikit-learn, jenkspy) are pinned in `requirements.txt` to the versions installed in `.venv` (Python 3.14). Update the pins when adding a dependency. To recreate the environment:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The code runs on CPU, at a few seconds per run. It uses CUDA automatically if available.

```bash
.venv/bin/python pu_benchmark.py                   # full grid: 14 datasets x SCAR/SAR x 3 ratios x 10 runs, in parallel
.venv/bin/python pu_benchmark.py --summary-only    # re-print the paired comparison against LBE
.venv/bin/python pu_benchmark.py --data dataset/scaled_yeast.csv --mechanisms SAR --pu-ratios 0.75 --out /tmp/x   # small subset
.venv/bin/python tabular.py --data dataset/scaled_cardio.csv -v   # standalone run with per-epoch logging
```

`pu_benchmark.py` skips settings that already have a `results.csv`. Pass `--overwrite` to rerun them, for example after changing hyperparameters. Hyperparameter flags (`--lr`, `--warming-epochs`, `--ft-epochs`, `--eval-step`, `--three-sigma`, …) are shared by both scripts through `tabular.add_method_args`.

## Architecture

The import chain is: `pu_benchmark.py` → `pu_setting.py` (splits, PU labels, metrics) and `tabular.py` (training) → `model/mlp.py` (`CardMLP`), `model/ema.py` (`ModelEMA`), `utils/misc.py` (cosine LR schedule, `three_sigma`).

**Label conventions differ between modules (important):**
- `pu_setting.py` follows `iFPNU`: **positive = 1**, and `y_` = 1 marks a labeled positive.
- `tabular.py` follows the original ALM-PU code: **positive = 0, negative = 1**. Every unlabeled sample is trained with the observed label 1.
- `pu_benchmark.run_setting` converts between the two. Model scores passed to the metrics are P(positive) = `softmax[:, 0]`.

**`pu_setting.py` is a verbatim port** of `iFPNU`'s `CSVDataset.get_splits`, `create_pu_data`, `prepare_pu_data_plain` and `validate_sarem`. It was verified to be byte-identical on all 840 splits and on the metrics. Keep it in sync with the originals; any change breaks comparability. Details of the setting:
- 70/30 stratified split, with `random_state` equal to the run index.
- `create_pu_data` hides `pu_ratio` of the training positives. It always uses `random_state=123`.
- SAR weights each positive by its mean distance to the *first* √n negatives in row order, not the nearest ones.
- `f1` is the best F1 over the PR curve, so its threshold is chosen on the test set.

**ALM-PU training (`tabular.py`):**
1. `train_phase1` trains on cross-entropy for the labeled positives (label smoothing `--rho`) plus a cross-entropy `Ln` that treats all unlabeled samples as negative. An augmented-Lagrangian (PHR) penalty is added on `Ln`. The multiplier `lam` and penalty weight `beta` are updated once per epoch (`--false_alarm_cutoff`, `--penalty_mult`, `--alpha`). After each epoch, the EMA model's P(positive) on U is recorded. Each sample's trend across epochs is split in two with Jenks natural breaks (`jenkspy`), which gives the pseudo-labels.
2. `train` builds a **fresh** model trained on labeled cross-entropy plus cross-entropy on U against a target. That target moves from the observed label toward the pseudo-label as `(epoch / ft_epochs) ** 0.8`. The original FixMatch consistency term is dropped, since tabular data has no augmentations. Test scores come from the final epoch.

Other details:
- Pseudo-labels are looked up by sample index (the unlabeled `TensorDataset` carries it), so the unlabeled loader is free to shuffle.
- The `three_sigma` re-split from the original `train.py` is off by default. Its cutoff was tuned for image data and breaks the pseudo-labels on tabular data such as mammography.
- `CardMLP(in_features)` has dropout that is active only in training mode.

**Data:** `dataset/scaled_*.csv` files are headerless and already standardized. The last column is the label (0/1), and 1 is the minority, positive class. Where a file name differs from `iFPNU`'s experiment name, the mapping is in `pu_benchmark.EXPERIMENT_NAMES` (e.g. `scaled_piechart2` → `SCALED_PIECHART`).

**Outputs:**
- `output_ALMPU/{EXPERIMENT}_{MECH}_{RATIO}_default_ALMPU/` holds `results.csv` and `runs_detail.csv`.
  - `results.csv` has the columns `run,dataset,pu_ratio,labeling_mechanism,roc_auc,ap,f1`, the same format as `iFPNU`'s `output_*` folders, plus `f1_pos,f1_neg,f1_macro` and `f1_pos_prior,f1_neg_prior,f1_macro_prior`.
  - `f1` is the best F1 over the PR curve and must keep that name, because `iFPNU/gather_results.py` reads it.
  - `f1_pos,f1_neg,f1_macro` are F1 at the 0.5 threshold (`round()`, so a score of exactly 0.5 counts as negative).
  - The `*_prior` columns use `pu_setting.f1_at_prior_threshold`, a port of `iFPNU`'s function of the same name (as used in `main_2step_ad.py`). It predicts the top `prior` fraction of test scores as positive, where `prior` is the true positive fraction of the training split (`loss_prior`).
  - None of the extra F1 columns are part of the LBE comparison.
  - `runs_detail.csv` adds the other metrics, `prior`, `prior_threshold` and pseudo-label diagnostics.
- `output_ALMPU/comparison_LBE.csv` compares against `--baseline-dir` (default `../iFPNU/output_LBE`), paired by run index.
- A copy of `output_ALMPU/` lives in `../iFPNU/output_ALMPU/`, where `gather_results.py` reads it as method `ALMPU` for the Friedman and Nemenyi tests. **After rerunning, copy it there again.**
- `tabular.py` run on its own uses a different setting (80/20 split, `--label-frac` labeled positives, F1 at a 0.5 threshold). Its numbers are not comparable to the benchmark.
