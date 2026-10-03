"""Gather the PU-learning result CSVs from the output* directories into one view.

Each output directory holds two CSVs per dataset (uniform / knn a.k.a. SAR). The
baseline directories write their metrics into columns named `*_dt_puhd`, copied
from the PUHD script, so the *directory* — not the column name — identifies the
model. This script does that mapping and reports mean scores and mean ranks.

Usage:
    python gather_all_results.py                        # everything
    python gather_all_results.py --sources output       # just the PUHD models
    python gather_all_results.py --metrics ap f1_max --std
    python gather_all_results.py --csv-dir evaluation/gathered
"""

import argparse
import glob
import os

import pandas as pd

# Directory -> {column suffix in that directory: model name}
SOURCES = {
    'output': {'dt_puhd': 'PU-HDT', 'srf_puhd': 'PU-HDRF'},
    'output_nnpu': {'dt_puhd': 'nnPU'},
    'output_upu': {'dt_puhd': 'uPU'},
    'output_imbnnpu': {'dt_puhd': 'imbnnPU'},
}
MODEL_ORDER = ['PU-HDT', 'PU-HDRF', 'nnPU', 'uPU', 'imbnnPU']
METRICS = ['roc', 'ap', 'f1', 'f1_pos', 'f1_neg', 'f1_macro',
           'f1_ci', 'f1_ci_pos', 'f1_ci_neg', 'f1_ci_macro', 'f1_max']
METRIC_LABELS = {
    'roc': 'ROC-AUC',
    'ap': 'PR-AUC (average precision)',
    'f1': 'F1 @ threshold 0.50',
    'f1_pos': 'F1 positive class @ threshold 0.50',
    'f1_neg': 'F1 negative class @ threshold 0.50',
    'f1_macro': 'F1 macro @ threshold 0.50',
    'f1_ci': 'F1 @ class-prior threshold',
    'f1_ci_pos': 'F1 positive class @ class-prior threshold',
    'f1_ci_neg': 'F1 negative class @ class-prior threshold',
    'f1_ci_macro': 'F1 macro @ class-prior threshold',
    'f1_max': 'F1 @ validation-tuned threshold',
}
KEYS = ['dataset', 'label_noise', 'flip_ratio', 'data_partition']
NOISE_LABELS = {'uniform': 'uniform (SCAR)', 'knn': 'knn (SAR)'}

# results_satelite_sar.csv spells it with one 'l' in both the filename and the
# dataset column; every other directory uses 'satellite'.
DATASET_ALIASES = {'satelite': 'satellite'}
EXPECTED_PARTITIONS = 10


def load(sources):
    """Read every result CSV into one long table.

    Metric columns are taken where present and filled with NaN where not, so a
    directory written before a metric was added still loads. Absent metrics are
    reported per model rather than per file.
    """
    frames, issues = [], []
    absent = {}
    for directory, mapping in sources.items():
        if not os.path.isdir(directory):
            issues.append(f'missing directory: {directory}')
            continue
        paths = sorted(glob.glob(os.path.join(directory, '*.csv')))
        if not paths:
            issues.append(f'no CSVs in {directory}')
        for path in paths:
            df = pd.read_csv(path)
            df['dataset'] = df['dataset'].replace(DATASET_ALIASES)
            for suffix, model in mapping.items():
                renames = {f'{m}_{suffix}': m for m in METRICS
                           if f'{m}_{suffix}' in df.columns}
                if not renames:
                    issues.append(f'{path}: no `*_{suffix}` metric columns')
                    continue
                sub = df[KEYS + list(renames)].rename(columns=renames)
                for metric in METRICS:
                    if metric not in sub.columns:
                        sub[metric] = float('nan')
                        absent.setdefault(model, set()).add(metric)
                sub.insert(0, 'model', model)
                sub['source_file'] = path
                frames.append(sub[['model'] + KEYS + METRICS + ['source_file']])
    for model, metrics in absent.items():
        issues.append(f'{model}: not yet in the CSVs -- '
                      f'{", ".join(sorted(metrics))} (rerun to populate)')
    if not frames:
        raise SystemExit('No result files found.')
    long = pd.concat(frames, ignore_index=True)
    long['model'] = pd.Categorical(
        long['model'],
        categories=[m for m in MODEL_ORDER if m in set(long['model'])],
        ordered=True,
    )
    return long, issues


def audit(long, issues):
    """Report anything that makes the aggregate less trustworthy."""
    print('=' * 78)
    print('DATA AUDIT')
    print('=' * 78)

    for issue in issues:
        print(f'  [warn] {issue}')

    counts = (long.groupby(['model', 'dataset', 'label_noise', 'flip_ratio'],
                           observed=True)['data_partition']
              .nunique().reset_index(name='n_partitions'))
    short = counts[counts['n_partitions'] < EXPECTED_PARTITIONS]
    if not short.empty:
        print(f'  [warn] incomplete runs (fewer than {EXPECTED_PARTITIONS} partitions):')
        for _, r in short.iterrows():
            print(f'         {r["model"]:<8} {r["dataset"]:<12} '
                  f'{r["label_noise"]:<8} flip={r["flip_ratio"]:.2f}  '
                  f'-> {r["n_partitions"]} partitions')

    cover = (long.groupby(['label_noise', 'flip_ratio', 'model'], observed=True)['dataset']
             .nunique().unstack('model'))
    if cover.stack().nunique() > 1:
        print('  [warn] models do not all cover the same number of datasets:')
        print(cover.to_string().replace('\n', '\n         '))

    print(f'\n  rows: {len(long)}   models: {list(long["model"].cat.categories)}')
    print(f'  datasets ({long["dataset"].nunique()}): '
          f'{", ".join(sorted(long["dataset"].unique()))}')
    print('\n  Note: `data_partition` is NOT comparable across directories -- each\n'
          '  experiment script calls np.random.shuffle(d) unseeded, so partition i\n'
          '  is a different split in each run. Ranks below are therefore computed\n'
          '  on per-dataset means, not paired per-partition.\n')


def with_metric(long, metric):
    """Restrict to the models that actually have values for `metric`."""
    counts = long.groupby('model', observed=True)[metric].count()
    keep = [m for m in counts.index if counts[m] > 0]
    if not keep:
        return None
    sub = long[long['model'].isin(keep)].copy()
    sub['model'] = sub['model'].cat.remove_unused_categories()
    return sub


def score_table(long, metric, noise, flip, with_std=False):
    """dataset x model table of mean (optionally mean±std over partitions)."""
    sel = long[(long['label_noise'] == noise) & (long['flip_ratio'] == flip)]
    if sel.empty:
        return None
    mean = sel.pivot_table(index='dataset', columns='model', values=metric,
                           aggfunc='mean', observed=True)
    if not with_std:
        return mean.round(4)
    std = sel.pivot_table(index='dataset', columns='model', values=metric,
                          aggfunc='std', observed=True)
    return mean.round(3).astype(str) + ' ± ' + std.round(3).astype(str)


def mean_ranks(long, metric):
    """Mean rank per model (1 = best) for each noise/flip condition.

    Ranked on per-dataset means and restricted to datasets every model covers,
    so the comparison is not skewed by partial coverage.
    """
    rows = {}
    for (noise, flip), grp in long.groupby(['label_noise', 'flip_ratio'], observed=True):
        table = grp.pivot_table(index='dataset', columns='model', values=metric,
                                aggfunc='mean', observed=True).dropna()
        if table.empty:
            continue
        ranked = table.rank(axis=1, ascending=False).mean()
        ranked['n_datasets'] = len(table)
        rows[(NOISE_LABELS.get(noise, noise), flip)] = ranked
    if not rows:
        return None
    out = pd.DataFrame(rows).T
    out.index.names = ['label_noise', 'flip_ratio']
    return out


SPREAD_LABELS = {
    'partition': 'std over partitions, averaged over datasets (run-to-run stability)',
    'dataset': 'std across dataset means (consistency across problems)',
}


def headline(long, metric, spread='partition'):
    """Mean score per model, averaged over datasets, with a standard deviation.

    Two different spreads are available and they answer different questions:

    `partition`  per-dataset std over the 10 data partitions, then averaged over
                 datasets -- how stable one method is across reruns of the same
                 problem. This is what the older per-file scripts reported.
    `dataset`    std of the per-dataset means -- how much a method's score swings
                 between problems. Much larger, since the datasets differ wildly
                 in base rate.
    """
    cond = ['label_noise', 'flip_ratio']
    per_dataset_mean = long.groupby(cond + ['dataset', 'model'],
                                    observed=True)[metric].mean()

    mean = per_dataset_mean.groupby(cond + ['model'], observed=True).mean()
    if spread == 'partition':
        per_dataset_std = long.groupby(cond + ['dataset', 'model'],
                                       observed=True)[metric].std()
        std = per_dataset_std.groupby(cond + ['model'], observed=True).mean()
    else:
        std = per_dataset_mean.groupby(cond + ['model'], observed=True).std()

    table = pd.concat({'mean': mean.unstack('model').round(4),
                       'std': std.unstack('model').round(4)}, axis=1)
    table = table.swaplevel(axis=1)
    models = [m for m in MODEL_ORDER if m in table.columns.get_level_values(0)]
    table = table.reindex(columns=pd.MultiIndex.from_product([models, ['mean', 'std']]))
    table.index = pd.MultiIndex.from_tuples(
        [(NOISE_LABELS.get(n, n), f) for n, f in table.index],
        names=['label_noise', 'flip_ratio'])
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sources', nargs='+', choices=list(SOURCES),
                    default=list(SOURCES), help='result directories to include')
    ap.add_argument('--metrics', nargs='+', choices=METRICS, default=METRICS)
    ap.add_argument('--noise', nargs='+', choices=['uniform', 'knn'],
                    default=['uniform', 'knn'])
    ap.add_argument('--flip', nargs='+', type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument('--std', action='store_true',
                    help='show mean ± std over partitions in the per-dataset tables')
    ap.add_argument('--spread', choices=['partition', 'dataset'], default='partition',
                    help='which standard deviation the summary table reports '
                         '(default: partition)')
    ap.add_argument('--no-per-dataset', action='store_true',
                    help='print only the summary and rank tables')
    ap.add_argument('--csv-dir', help='also write the long table and summaries here')
    args = ap.parse_args()

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 50)

    long, issues = load({d: SOURCES[d] for d in args.sources})
    audit(long, issues)

    long = long[long['label_noise'].isin(args.noise) & long['flip_ratio'].isin(args.flip)]
    if long.empty:
        raise SystemExit('No rows left after filtering.')

    for metric in args.metrics:
        print('=' * 78)
        print(f'{METRIC_LABELS[metric]}  [{metric}]')
        print('=' * 78)

        sub = with_metric(long, metric)
        if sub is None:
            print('\n  no model has this metric yet -- rerun the experiment scripts.\n')
            continue
        dropped = [m for m in long['model'].cat.categories
                   if m not in sub['model'].cat.categories]
        if dropped:
            print(f'\n  excluded (metric not in their CSVs): {", ".join(dropped)}')

        print(f'\n-- mean over datasets | std = {SPREAD_LABELS[args.spread]}')
        print('-' * 78)
        print(headline(sub, metric, args.spread).to_string())

        ranks = mean_ranks(sub, metric)
        if ranks is not None:
            print('\n-- mean rank (1 = best) ' + '-' * 52)
            print(ranks.to_string(float_format=lambda v: f'{v:.2f}'))

        if not args.no_per_dataset:
            for noise in args.noise:
                for flip in args.flip:
                    table = score_table(sub, metric, noise, flip, args.std)
                    if table is None:
                        continue
                    print(f'\n-- per dataset | {NOISE_LABELS.get(noise, noise)} | '
                          f'flip_ratio={flip} ' + '-' * 20)
                    print(table.to_string())
        print()

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        long.to_csv(os.path.join(args.csv_dir, 'all_results_long.csv'), index=False)
        for metric in args.metrics:
            sub = with_metric(long, metric)
            if sub is None:
                continue
            headline(sub, metric, args.spread).to_csv(
                os.path.join(args.csv_dir, f'summary_{metric}.csv'))
            ranks = mean_ranks(sub, metric)
            if ranks is not None:
                ranks.to_csv(os.path.join(args.csv_dir, f'ranks_{metric}.csv'))
        print(f'Wrote long table and per-metric summaries to {args.csv_dir}/')


if __name__ == '__main__':
    main()
