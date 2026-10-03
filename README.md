# PU monorepo

Positive-Unlabeled (PU) learning research code, in three projects:

| Folder | Contents |
| --- | --- |
| `iFPNU/` | PU risk estimators, PNU risk estimators, one-step and two-step methods, and the benchmark that compares all methods (`gather_results.py`). Formerly the `pu_learning` repo. |
| `PU_Hellinger_Trees/` | PU learning with tree-based classifiers (Hellinger trees, Cython) and different losses. Formerly the `pu_trees` repo. |
| `ALM-PU/` | ALM-PU on the same tabular benchmarks, in the `iFPNU/main_LBE.py` setting. Formerly the `ALM-PU-Tab` repo. |

Each project has its own Python version and dependencies, so each one has its own virtual environment. Run scripts from inside the project folder: paths such as `./data` in `iFPNU/config.ini`, and `../iFPNU/output_LBE` in ALM-PU, are relative to it.

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

## Usage

```bash
cd iFPNU && venv/bin/python main_LBE.py         # settings in config.ini
cd ALM-PU && .venv/bin/python pu_benchmark.py   # compares against ../iFPNU/output_LBE
```

Experiment outputs (`output*/` folders) are not committed, except `ALM-PU/output_ALMPU/`.

## Notes

- **xgboost in PU_Hellinger_Trees.** The working environment uses xgboost 2.1.3 built locally from source without OpenMP. The PyPI wheel that `requirements-lock.txt` installs instead needs `libomp.dylib` under `/usr/local/opt/libomp` (x86_64), and fails to import without it. Either install an x86_64 libomp there, or reuse the locally built xgboost package.
- **`PU_Hellinger_Trees/expsetup_clean.py` runs its sweep at import time.** It has no `if __name__ == '__main__':` guard, so `import expsetup_clean` starts the experiments and writes to `output/`.
