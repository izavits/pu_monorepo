import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasets = [
    'SCALED_CARDIO',
    'SCALED_CLIMATE',
    'SCALED_COVER',
    'SCALED_LETTER',
    'SCALED_MAMMOGRAPHY',
    'SCALED_PIECHART',
    'SCALED_PIZZACUTTER',
    'SCALED_POKER',
    'SCALED_SATELLITE',
    'SCALED_SEGMENT',
    'SCALED_SEISMIC',
    'SCALED_SHUTTLE',
    'SCALED_THYROID',
    'SCALED_YEAST'
]

MECHANISMS = ['SCAR', 'SAR']
METRICS = ["roc_auc", "ap", "f1", "std_roc", "std_ap", "std_f1"]
ETAS = [0.25, 0.5, 0.75]
PRIORS = ['0.25p', '0.5p', 'p', '1.5p', '2p', '4p']


def empty_results():
    return {
        prior: {eta: {m: [] for m in METRICS} for eta in ETAS}
        for prior in PRIORS
    }


results = {mech: empty_results() for mech in MECHANISMS}


def parse_result_csv(path):
    df = pd.read_csv(path)
    mean_roc_auc = df["roc_auc"].mean()
    mean_ap = df["ap"].mean()
    mean_f1 = df["f1"].mean()
    std_roc = df["roc_auc"].std()
    std_ap = df["ap"].std()
    std_f1 = df["f1"].std()
    return mean_roc_auc, mean_ap, mean_f1, std_roc, std_ap, std_f1


def gather_results():
    #SCAR
    for prior in PRIORS:
        for eta in ETAS:
            for data in datasets:
                path = f'./output_2step_sensitivity/{data}_SCAR_0.75_dif_iFPNU_{prior}_{eta}_results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['SCAR'][prior][eta]['roc_auc'].append(roc)
                results['SCAR'][prior][eta]['ap'].append(ap)
                results['SCAR'][prior][eta]['f1'].append(f1)
                results['SCAR'][prior][eta]['std_roc'].append(std_roc)
                results['SCAR'][prior][eta]['std_ap'].append(std_ap)
                results['SCAR'][prior][eta]['std_f1'].append(std_f1)
    #SAR
    for prior in PRIORS:
        for eta in ETAS:
            for data in datasets:
                path = f'./output_2step_sensitivity/{data}_SAR_0.75_dif_iFPNU_{prior}_{eta}_results.csv'
                roc, ap, f1, std_roc, std_ap, std_f1 = parse_result_csv(path)
                results['SAR'][prior][eta]['roc_auc'].append(roc)
                results['SAR'][prior][eta]['ap'].append(ap)
                results['SAR'][prior][eta]['f1'].append(f1)
                results['SAR'][prior][eta]['std_roc'].append(std_roc)
                results['SAR'][prior][eta]['std_ap'].append(std_ap)
                results['SAR'][prior][eta]['std_f1'].append(std_f1)


def plot_sensitivity_results():
    #METRIC, STD_KEY, YLABEL = "ap", "std_ap", "AP (mean ± std)"  # or "roc_auc"/"std_roc", "f1"/"std_f1"
    METRIC, STD_KEY, YLABEL = "f1", "std_f1", "F1 (mean ± std)"
    PRIORS = ["0.25p", "0.5p", "p", "1.5p", "2p", "4p"]
    LABELS = {"0.25p": r"$0.25\pi$", "0.5p": r"$0.5\pi$", "p": r"$\pi$",
              "1.5p": r"$1.5\pi$", "2p": r"$2\pi$", "4p": r"$4\pi$"}
    ETAS = [0.25, 0.5, 0.75]
    OFFSET = 0.06  # small horizontal jitter so error bars don't ove

    def pooled_std(s):  # run-to-run std, pooled over datasets
        return float(np.sqrt(np.nanmean(np.square(s))))

    for mech in ["SCAR", "SAR"]:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
        x = np.arange(len(PRIORS))
        for k, eta in enumerate(ETAS):
            means = [np.mean(results[mech][p][eta][METRIC]) for p in PRIORS]
            stds = [pooled_std(results[mech][p][eta][STD_KEY]) for p in PRIORS]
            ax.errorbar(x + (k - 1) * OFFSET, means, yerr=stds, marker="o", capsize=3,
                        label=rf"$\eta={eta}$")
        ax.set_xticks(x);
        ax.set_xticklabels([LABELS[p] for p in PRIORS])
        ax.set_xlabel("Estimated class prior");
        ax.set_ylabel(YLABEL)
        ax.set_title(mech);
        ax.legend(loc="lower center", ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"./output_2step_sensitivity/prior_sensitivity_{METRIC}_{mech}.png")


if __name__ == '__main__':
    gather_results()
    #print(results)
    plot_sensitivity_results()
