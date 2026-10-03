"""Single source of truth for the benchmark datasets and the label-noise mechanisms.

Paths are relative to the PU_Hellinger_Trees folder, so run the experiment scripts from there.
Method-level choices (models, metrics, partitions, flip ratios) deliberately stay
in each experiment script; only the dataset table and the mechanical write/report
helpers live here, because those are facts about the datasets rather than method.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = '../data'  # monorepo root, shared with iFPNU and ALM-PU

# (file in data/, value written to the `dataset` column and used as the filename stem)
DATASETS = [
    ('scaled_cardio.csv', 'cardio'),
    ('scaled_climate.csv', 'climate'),
    ('scaled_cover.csv', 'cover'),
    ('scaled_letter.csv', 'letter'),
    ('scaled_mammography.csv', 'mammography'),
    ('scaled_piechart2.csv', 'piechart'),
    ('scaled_pizzacutter1.csv', 'pizzacutter'),
    ('scaled_poker-8-9_vs_5.csv', 'poker'),
    ('scaled_satellite.csv', 'satellite'),
    ('scaled_segment0.csv', 'segment'),
    ('scaled_seismic.csv', 'seismic'),
    ('scaled_shuttle.csv', 'shuttle'),
    ('scaled_thyroid.csv', 'thyroid'),
    ('scaled_yeast.csv', 'yeast'),
]

# (make_noisy_negatives label_noise value, output filename suffix)
LABEL_NOISES = [('uniform', ''), ('knn', '_sar')]

FLIP_RATIOS = [0.25, 0.50, 0.75]

# A bad fold, not a bad build. The inner 50/50 split can leave zero observed
# positives in the tuning half, after which a tree fits one class and
# predict_proba(...)[:, 1] raises IndexError; roc_auc_score on a single-class
# test fold raises ValueError. Anything else must stay loud.
SKIPPABLE = (ValueError, IndexError)


def sweep_units():
    """The 28 (dataset, noise) work units, dataset-major."""
    return [(filename, stem, noise, suffix)
            for filename, stem in DATASETS
            for noise, suffix in LABEL_NOISES]


def load_dataset(filename):
    """Owned float64 array: features in [:, :-1], binary label in [:, -1].

    np.array rather than np.asarray: for the 9 single-block files asarray returns
    a view onto the DataFrame, so np.random.shuffle would mutate it in place.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, filename), header=None)
    return np.array(df, dtype=np.float64)


def output_path(out_dir, stem, noise_suffix, model_suffix=''):
    """output/results_cardio.csv, output_nnpu/results_cardio_sar_nnpu.csv, ..."""
    return os.path.join(out_dir, f'results_{stem}{noise_suffix}{model_suffix}.csv')


def write_unit(path, rows, columns):
    """Write one unit's rows atomically, so an interrupted write leaves no stub."""
    tmp = path + '.tmp'
    pd.DataFrame(rows, columns=columns).to_csv(tmp, index=False)
    os.replace(tmp, path)


def report_sweep(written, skipped_existing, failures, expected_rows):
    """Print what the sweep produced, what it reused, and what it could not do."""
    print('\n' + '=' * 78)
    print('SWEEP SUMMARY')
    print('=' * 78)
    print(f'  files written:            {len(written)}')
    print(f'  units skipped (existing): {len(skipped_existing)}')
    print(f'  iterations failed:        {len(failures)}')

    short = [(p, n) for p, n in written if n < expected_rows]
    if short:
        print(f'\n  incomplete files (fewer than {expected_rows} rows):')
        for path, n in short:
            print(f'    {path}  {n}/{expected_rows} rows')

    if failures:
        rolled = {}
        for stem, noise, r, flip, msg in failures:
            rolled.setdefault((stem, noise, flip), []).append(msg)
        print('\n  failures by dataset / noise / flip_ratio:')
        for (stem, noise, flip), msgs in sorted(rolled.items()):
            print(f'    {stem:12s} {noise:8s} flip={flip:.2f}  '
                  f'{len(msgs)} failed  {msgs[0]}')
        print('\n  note: these folds had no observed positives in the tuning half, so the\n'
              '  model degenerated to a single class. A data-size limit, not a code error.')


if __name__ == '__main__':
    print(f'{"dataset":12s} {"file":28s} {"shape":>14s} {"pos":>6s}')
    for filename, stem in DATASETS:
        d = load_dataset(filename)
        print(f'{stem:12s} {filename:28s} {str(d.shape):>14s} '
              f'{int(d[:, -1].sum()):>6d}')
    print(f'\n{len(DATASETS)} datasets x {len(LABEL_NOISES)} noise mechanisms '
          f'= {len(sweep_units())} units per script')
