# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for Positive-Unlabeled (PU) learning with tree-based classifiers, centered on a
Hellinger-distance splitting criterion modified to account for the PU label prior. `pu_tree_simplified/`
is a vendored, hand-modified fork of scikit-learn 1.0.2's decision tree / random forest internals; the
rest of the repo is baseline PU methods and the experiment harness that compares them.

## Environment and build

Python 3.8 in the checked-in `./venv` (sklearn 1.0.2, numpy 1.24.4, xgboost 2.1.3, torch 2.2.2,
imbalanced-learn, Cython 3.0.11). The vendored tree code is pinned to sklearn 1.0.2 internals — do not
upgrade sklearn.

```bash
source venv/bin/activate
```

The Cython extensions must be compiled in place before anything under `pu_tree_simplified/` imports:

```bash
python setup.py build_ext --inplace
```

Recompile after **any** edit to `_pu_criterion.pyx`, `_pu_splitter.pyx`, `_pu_tree.pyx`, `_utils.pyx`,
or their `.pxd` files. Editing only the `.py` files (`_pu_classes.py`, `_pu_randomforest.py`) needs no
rebuild. Generated `.c`, `.so`, and Windows `.pyd` artifacts are committed to git.

There is no test suite and no linter config. Verification means running an experiment script and
checking the printed PR-AUC / the emitted CSV.

## Architecture

### `pu_tree_simplified/` — the PU tree

Three modifications distinguish this from stock sklearn:

1. **PU-weighted Hellinger criterion** (`_pu_criterion.pyx`, `HellingerDistanceCriterion`). In
   `children_impurity` the labeled-positive counts in each child are rescaled by
   `p_y / ((N-U)/N)` before the Hellinger distance is computed, so the split is evaluated against
   *estimated* true positives rather than observed labels. `p_y` is the true positive prior, `U` the
   number of unlabeled samples (`y == 0`), `N` the total.

2. **The `p_y` / `U` / `N` plumbing.** These flow
   `DecisionTreeClassifier.fit(X, y, p_y=...)` → `builder.build(tree, X, y, sample_weight, p_y, U, N)`
   → `splitter.init(...)` → `criterion.init(...)`. `U` and `N` are derived inside `fit`
   ([_pu_classes.py:149](pu_tree_simplified/_pu_classes.py:149)), not passed by the caller. Changing this
   signature means editing the `.pyx`/`.pxd` chain and recompiling — the Python and Cython layers must
   stay in sync.

3. **PU-biased bootstrap** (`_parallel_build_trees` in
   [_pu_randomforest.py:147](pu_tree_simplified/_pu_randomforest.py:147)). When
   `pu_biased_bootstrap=True`, each tree bootstraps only the *unlabeled* indices and appends **all**
   positives unresampled. Callers pair this with `max_samples=sum(y == 0)`.

`predict_proba` returns raw leaf proportions; `predict_pu_proba` applies an Elkan-Noto style correction
using `p_y` and the root's labeled/unlabeled counts. The experiment scripts use `predict_proba`.

`DecisionTreeClassifier` here defaults to `criterion="hellinger"` (stock sklearn defaults to gini);
`"gini"` and `"entropy"` are still available.

### `pos_noisyneg/` — baselines and data synthesis

`utils.make_noisy_negatives(y, X, flip_ratio, label_noise, random_state)` is how PU data is
manufactured: it flips a fraction of true positives to label 0. `label_noise='uniform'` gives SCAR
(selected completely at random); `label_noise='knn'` gives SAR (biased selection via nearest
neighbours) — this is the distinction behind the `_sar` suffix in output filenames.

The rest are competing PU methods with sklearn-style APIs: `SpyEM`, `Spy`, `RankPruning`,
`WeightedElkanotoPuClassifier`, `BaggingPuClassifier`, `PNN`, `WeightedIsoForest`,
`EnsembleWeightedIsoForest`.

### `ecsmodels/methodologies/cs_boost.py` — CSBoost

XGBoost wrapper dispatching on `obj`: `'nnpu'`, `'imbnnpu'`, `'puwce'` (uPU), `'pucano'`,
`'weightedce'`, `'ce'`, `'aec'`. Each installs a custom gradient/hessian and eval function. Returns the
raw booster from `fit`, so callers score with `expit(model.inplace_predict(X))`.

## Experiment scripts

The four `_clean` scripts each sweep **all 14 datasets × both noise mechanisms** — 28 CSVs per
script, written straight into that script's own output directory:

| Script | Models | Output |
| --- | --- | --- |
| `expsetup_clean.py` | PU Hellinger tree + PU Hellinger stratified forest (the proposed methods) | `output/` |
| `nnPU_clean.py` | CSBoost nnPU | `output_nnpu/` |
| `uPU_clean.py` | CSBoost uPU (`obj='puwce'`) | `output_upu/` |
| `imbalancedNNPU_clean.py` | CSBoost imbalanced nnPU | `output_imbnnpu/` |

`expsetup.py` (the full comparison including all `pos_noisyneg` baselines and plain sklearn RF) and
`evaluation_experiments*.py` (post-hoc ranking) are the older single-dataset scripts and were left
alone.

```bash
python expsetup_clean.py        # ~1 hour; the three baselines are ~3-5 min each
```

Each script's configuration is a four-line block under the imports — `OUT_DIR`, `MODEL_SUFFIX`,
`SKIP_EXISTING`, and `COLUMNS`. Everything else about the sweep comes from
[`experiment_datasets.py`](experiment_datasets.py), the single source of truth for the dataset table:
`DATASETS` (14 `(filename, short name)` pairs, where the short name is both the `dataset` column value
and the output filename stem), `LABEL_NOISES`, `FLIP_RATIOS`, and the `load_dataset` / `output_path` /
`write_unit` / `report_sweep` helpers. `python experiment_datasets.py` prints the table with shapes and
positive counts as a self-check. Adding or renaming a dataset is a one-line edit there, not four
coordinated edits per script.

`load_dataset` uses `np.array(df, dtype=np.float64)` deliberately: for 9 of the 14 files (those that
land in a single pandas block) `np.asarray(df)` returns a *view*, and the loop's `np.random.shuffle`
would then mutate the cached frame in place.

Per (dataset, noise) unit the loop is: 10 random partitions × `flip_ratio` in {0.25, 0.50, 0.75}; a
`StratifiedShuffleSplit` stratified on the three-way `negs / noisy_negs / pos` label so contamination
is balanced across folds; an inner train/val split used only to sweep 100 thresholds for the F1-optimal
cutoff; then a refit on the full training half and scoring on test. `r` stays outside `flip_ratio`, so
rows come out `(0, 0.25), (0, 0.50), (0, 0.75), (1, 0.25), …` — matching the older result files.

Do **not** move the noise mechanism into the `ParameterGrid`. It sorts keys alphabetically and varies
the last fastest, so `{'flip_ratio': [...], 'label_noise': ['uniform', 'knn']}` interleaves the two
mechanisms inside each flip ratio, which would scatter one CSV's rows and break the row order above.

### Resume and partial results

`SKIP_EXISTING = True` (the default) skips any unit whose output CSV already exists, so an interrupted
sweep resumes where it stopped. Set it to `False` to force a clean regeneration — and note that
**because all four output directories ship fully populated, the default means a first run does
nothing**. Writes go through a `.tmp` file plus `os.replace`, so an interrupted write cannot leave a
truncated CSV that a later run mistakes for complete.

Each (partition, flip_ratio) iteration runs inside one `try` catching `(ValueError, IndexError)`. On
failure the whole row is discarded — in `expsetup_clean.py` both models share one row, so a failure
between them can never emit a half-filled row — and the sweep continues; a unit still gets a CSV
containing the rows that succeeded. The end-of-run summary lists incomplete files and rolls failures up
by (dataset, noise, flip_ratio).

This is not hypothetical. `piechart` has 16 positives and `poker` 25, so at `flip_ratio` 0.50 and 0.75
the inner 50/50 split can leave **zero** observed positives in the tuning half; the tree then fits a
single class, `predict_proba` returns one column, and `[:, 1]` raises `IndexError`. Measured rate:
piechart uniform 0.75 → 3/10 partitions, poker knn 0.75 → 3/10, a few others 1/10, everything else
0/10. Those truncated `piechart`/`poker` files predate the `try` and are what an uncaught `IndexError`
left behind. The three CSBoost baselines score via `expit(inplace_predict(X))` and never index
`[:, 1]`, so they do not hit this.

## Result CSVs — naming gotcha

**Column names do not identify the model.** The nnPU, uPU, and imbnnPU scripts write their metrics into
columns literally named `roc_dt_puhd`, `ap_dt_puhd`, `f1_dt_puhd`, etc., copied from the PUHD script.
Only the output directory tells you which model produced a file:

| Directory | Contents |
| --- | --- |
| `output/` | PU Hellinger tree (`*_dt_puhd`) and PU Hellinger stratified forest (`*_srf_puhd`) |
| `output_nnpu/`, `output_upu/`, `output_imbnnpu/` | CSBoost baselines, mislabeled as `*_dt_puhd` |
| `output_max_f1_test/` | runs including the oracle best-achievable test F1 |
| `evaluation/` | aggregated mean/std tables per flip ratio |

Filenames ending `_sar` are the `label_noise='knn'` runs; the rest are `uniform`.

Metric prefixes, all sharing the same `_dt_puhd` / `_srf_puhd` suffix convention:

| Prefix | Meaning |
| --- | --- |
| `roc_`, `ap_` | ROC-AUC and average precision, threshold-free |
| `f1_` | F1 of the positive class at threshold 0.50 |
| `f1_pos_`, `f1_neg_`, `f1_macro_` | per-class and macro F1, also at threshold 0.50 |
| `f1_ci_` | F1 of the positive class at a threshold set to the class prior |
| `f1_ci_pos_`, `f1_ci_neg_`, `f1_ci_macro_` | per-class and macro F1, at that same class-prior threshold |
| `f1_max_` | F1 at a threshold tuned on the validation half |
| `f1_max_test_` | oracle F1 at a threshold tuned on test — upper bound, not a fair score |

Two pairs are the same number by construction, because sklearn's `f1_score` defaults to
`pos_label=1` and each pair is computed from one shared label array: `f1_` equals `f1_pos_` (threshold
0.50) and `f1_ci_` equals `f1_ci_pos_` (class-prior threshold). The duplicates are kept so each
pos/neg/macro trio reads as a set.

On these imbalanced datasets the `_neg_` and `_macro_` variants compress the differences between
methods: a model that predicts almost nothing positive still scores ~0.9 on the negative class, so
macro F1 has a high floor regardless of how badly the positive class fails. Read them alongside `ap_`,
not instead of it.

`gather_all_results.py` fills any metric a CSV does not carry with NaN and reports it as unpopulated
rather than skipping the file, so older result sets still load. The class-prior trio
(`f1_ci_pos_`/`f1_ci_neg_`/`f1_ci_macro_`) was added after the first full sweep, so it needs a
regeneration to appear.

Two defects in the **pre-sweep** results, both fixed by a regeneration:

- `output/results_satelite_sar.csv` spells the dataset `satelite` (one `l`) in both the filename and
  the `dataset` column; every other directory uses `satellite`. The sweep writes the correct
  `results_satellite_sar.csv`, so **delete the misspelled file** — leaving it is worse than the
  original typo, because `DATASET_ALIASES` maps it onto the same dataset and
  `gather_all_results.py` would then load *both* into the satellite/knn cell (20 rows where 10 are
  expected, silently mixing an old run with a new one). The partition-count audit will not catch it,
  since `nunique('data_partition')` still reports 10.
- Four `output/` files are truncated partial runs — `results_piechart.csv` (1 partition),
  `results_poker_sar.csv` (3), `results_piechart_sar.csv` and `results_poker.csv` (5) — against 10
  partitions everywhere else, so the PUHD numbers for piechart and poker are averaged over far fewer
  seeds than the baselines they are compared against. Cause: an uncaught `IndexError` on a degenerate
  fold killed the process mid-sweep (see "Resume and partial results" above). The current scripts
  catch it, so these datasets now yield a near-complete file plus a reported skip count.

Because `output*` is gitignored (`.gitignore:4`), none of these CSVs are tracked — copy a directory
aside before regenerating it, or the old numbers are gone.

`data_partition` is **not** comparable across directories: every experiment script calls
`np.random.shuffle(d)` unseeded, so partition *i* is a different underlying split in each run.
Within a single file the models are genuinely paired; across directories they are not, so
cross-model comparison should aggregate per dataset before ranking.

## Aggregating results

`gather_all_results.py` loads all four directories into one tidy long table, resolving the
`*_dt_puhd` column ambiguity by directory, and prints mean scores and mean ranks per
metric / label-noise / flip-ratio. It reports the defects above as warnings rather than hiding them.

```bash
python gather_all_results.py --sources output --metrics ap
python gather_all_results.py --no-per-dataset --csv-dir evaluation/gathered
```

## Data

`data/*.csv` are headerless; the last column is the binary label, everything before it is features
(already scaled). Load with `pd.read_csv(path, header=None)`.
