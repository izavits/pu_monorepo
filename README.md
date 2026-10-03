# PU monorepo

Positive-Unlabeled (PU) learning prototype monorepo, in three projects:

| Folder | Contents |
| --- | --- |
| `iFPNU/` | PU risk estimators, PNU risk estimators, one-step and two-step methods |
| `PU_Hellinger_Trees/` | PU learning with tree-based classifiers (Hellinger trees, Cython) and different losses. |
| `ALM-PU/` | ALM-PU on the same tabular benchmarks. |
| `data/` | The 14 tabular benchmark datasets (`scaled_*.csv`) used by all three projects. |

Each project has its own Python version and dependencies, so each one has its own virtual environment. Run scripts from inside the project folder: paths such as `../data` (the shared datasets) are relative to it.

## Setup

```bash
# iFPNU (Python 3.9). The lock file pins the exact environment; --no-deps keeps those pins as they are.
cd iFPNU
python3.9 -m venv venv
venv/bin/pip install --no-deps -r requirements-lock.txt

# PU_Hellinger_Trees (Python 3.8), then build the Cython extensions
cd ../PU_Hellinger_Trees
python3.8 -m venv venv
venv/bin/pip install --no-deps -r requirements-lock.txt
venv/bin/python setup.py build_ext --inplace

# ALM-PU (Python 3.14)
cd ../ALM-PU
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage examples

```bash
cd iFPNU && venv/bin/python main_LBE.py         # settings in config.ini
cd ALM-PU && .venv/bin/python pu_benchmark.py
cd PU_Hellinger_trees && venv/bin/python uPU_clean.py
```

`iFPNU/gather_results.py` prints results from all methods. It reads ALM-PU and PU_Hellinger_Trees results straight from `../ALM-PU/output_ALMPU/` and `../PU_Hellinger_Trees/output*/`

## Notes
- **xgboost in PU_Hellinger_Trees.** The working environment uses xgboost 2.1.3 built locally from source without OpenMP. 