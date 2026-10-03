#
# statistical_tests.py
# Perform statistical tests between samples
#

from scipy.stats import wilcoxon, ttest_rel, kruskal, friedmanchisquare
from statds.no_parametrics import iman_davenport, friedman
import scikit_posthocs as sp
import numpy as np
import pandas as pd
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

alpha = 0.05
# Test iFPU, PUHDT and PUHRF in terms of PR_AUC across datasets
ifpu = {
    'SCAR': {
        0.25: [0.93, 0.33, 0.87, 0.41, 0.65, 0.17, 0.28, 0.1, 0.7, 0.98, 0.2, 0.99, 0.93, 0.76],
        0.50: [0.9, 0.26, 0.84, 0.36, 0.6, 0.13, 0.28, 0.06, 0.68, 0.97, 0.17, 0.99, 0.92, 0.75],
        0.75: [0.87, 0.21, 0.81, 0.23, 0.5, 0.1, 0.21, 0.06, 0.62, 0.96, 0.15, 0.98, 0.85, 0.61]
    },
    'SAR': {
        0.25: [0.94, 0.32, 0.89, 0.46, 0.61, 0.14, 0.32, 0.08, 0.72, 0.98, 0.19, 1, 0.96, 0.73],
        0.50: [0.9, 0.26, 0.85, 0.4, 0.51, 0.08, 0.22, 0.06, 0.67, 0.97, 0.16, 0.99, 0.93, 0.72],
        0.75: [0.85, 0.22, 0.77, 0.27, 0.41, 0.08, 0.24, 0.02, 0.52, 0.94, 0.13, 0.98, 0.91, 0.61]
    }
}
puhrf = {
    'SCAR': {
        0.25: [0.97, 0.34, 0.92, 0.67, 0.66, 0.09, 0.41, 0.06, 0.8, 0.99, 0.17, 0.99, 0.96, 0.83],
        0.50: [0.94, 0.28, 0.9, 0.57, 0.58, 0.13, 0.31, 0.03, 0.7, 0.99, 0.15, 0.99, 0.95, 0.77],
        0.75: [0.86, 0.28, 0.72, 0.37, 0.48, 0.07, 0.24, 0.05, 0.66, 0.98, 0.13, 0.96, 0.92, 0.68]
    },
    'SAR': {
        0.25: [0.97, 0.34, 0.9, 0.73, 0.65, 0.14, 0.37, 0.05, 0.76, 0.99, 0.14, 0.99, 0.95, 0.82],
        0.50: [0.95, 0.27, 0.9, 0.57, 0.46, 0.05, 0.3, 0.02, 0.66, 0.99, 0.1, 0.99, 0.95, 0.71],
        0.75: [0.83, 0.23, 0.73, 0.34, 0.22, 0.04, 0.19, 0.01, 0.34, 0.96, 0.07, 0.95, 0.85, 0.58]
    }
}
puhdt = {
    'SCAR': {
        0.25: [0.77, 0.18, 0.64, 0.18, 0.43, 0.05, 0.19, 0.02, 0.46, 0.95, 0.14, 0.95, 0.83, 0.61],
        0.50: [0.66, 0.19, 0.71, 0.19, 0.39, 0.07, 0.2, 0.02, 0.43, 0.93, 0.12, 0.94, 0.81, 0.56],
        0.75: [0.57, 0.16, 0.6, 0.11, 0.3, 0.04, 0.14, 0.03, 0.45, 0.89, 0.11, 0.89, 0.82, 0.46]
    },
    'SAR': {
        0.25: [0.73, 0.19, 0.64, 0.18, 0.42, 0.05, 0.16, 0.01, 0.39, 0.95, 0.14, 0.95, 0.77, 0.65],
        0.50: [0.69, 0.16, 0.56, 0.17, 0.3, 0.02, 0.16, 0.02, 0.28, 0.91, 0.1, 0.91, 0.82, 0.5],
        0.75: [0.51, 0.17, 0.42, 0.14, 0.19, 0.03, 0.13, 0.01, 0.12, 0.88, 0.07, 0.8, 0.75, 0.38]
    }
}
ifpnu_dif = {
    'SCAR': {
        0.25: [0.97, 0.29, 0.91, 0.7, 0.69, 0.1, 0.37, 0.12, 0.74, 1, 0.17, 1, 0.96, 0.74],
        0.50: [0.94, 0.29, 0.88, 0.6, 0.62, 0.1, 0.39, 0.09, 0.71, 0.99, 0.17, 0.99, 0.96, 0.66],
        0.75: [0.89, 0.29, 0.81, 0.39, 0.54, 0.1, 0.33, 0.04, 0.65, 0.97, 0.15, 0.99, 0.91, 0.5]
    },
    'SAR': {
        0.25: [0.96, 0.3, 0.92, 0.69, 0.65, 0.11, 0.39, 0.11, 0.74, 1, 0.17, 1, 0.96, 0.73],
        0.50: [0.95, 0.28, 0.88, 0.63, 0.58, 0.13, 0.29, 0.07, 0.71, 0.99, 0.15, 0.99, 0.92, 0.66],
        0.75: [0.89, 0.22, 0.79, 0.4, 0.44, 0.05, 0.3, 0.04, 0.64, 0.97, 0.11, 0.99, 0.91, 0.56]
    }
}

sarem = {
    'SCAR': {
        0.25: [0.93, 0.36, 0.94, 0.24, 0.45, 0.07, 0.29, 0.02, 0.69, 0.98, 0.19, 0.96, 0.89, 0.77],
        0.50: [0.89, 0.32, 0.9, 0.24, 0.53, 0.05, 0.29, 0.05, 0.6, 0.98, 0.18, 0.95, 0.87, 0.79],
        0.75: [0.86, 0.26, 0.84, 0.24, 0.53, 0.07, 0.26, 0.06, 0.49, 0.98, 0.17, 0.95, 0.84, 0.73]
    },
    'SAR': {
        0.25: [0.92, 0.31, 0.95, 0.24, 0.4, 0.11, 0.31, 0.02, 0.67, 0.98, 0.18, 0.96, 0.88, 0.78],
        0.50: [0.89, 0.27, 0.92, 0.23, 0.44, 0.1, 0.29, 0.03, 0.58, 0.98, 0.18, 0.96, 0.86, 0.74],
        0.75: [0.84, 0.26, 0.9, 0.18, 0.48, 0.07, 0.22, 0.07, 0.47, 0.98, 0.15, 0.95, 0.78, 0.73]
    }
}

lbe = {
    'SCAR': {
        0.25: [0.90, 0.26, 0.91, 0.18, 0.45, 0.09, 0.24, 0.08, 0.81, 0.98, 0.18, 0.95, 0.80, 0.65],
        0.50: [0.87, 0.25, 0.90, 0.19, 0.51, 0.06, 0.21, 0.08, 0.76, 0.98, 0.17, 0.95, 0.84, 0.68],
        0.75: [0.78, 0.21, 0.84, 0.19, 0.57, 0.08, 0.21, 0.09, 0.38, 0.97, 0.17, 0.94, 0.76, 0.63]
    },
    'SAR': {
        0.25: [0.88, 0.25, 0.93, 0.19, 0.47, 0.08, 0.22, 0.06, 0.81, 0.99, 0.18, 0.94, 0.75, 0.64],
        0.50: [0.86, 0.24, 0.90, 0.19, 0.55, 0.07, 0.22, 0.06, 0.81, 0.99, 0.18, 0.94, 0.81, 0.65],
        0.75: [0.77, 0.24, 0.75, 0.17, 0.53, 0.05, 0.21, 0.05, 0.42, 0.98, 0.16, 0.94, 0.76, 0.62]
    }
}

print('')
cases = ['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9',
         'data10', 'data11', 'data12', 'data13', 'data14']

for labeling in ['SCAR', 'SAR']:
    for ratio in [0.25, 0.50, 0.75]:
        print(f'TESTING {labeling} {ratio}')
        df = pd.DataFrame()
        df['cases'] = cases
        df['ifpu'] = ifpu[labeling][ratio]
        df['puhrf'] = puhrf[labeling][ratio]
        df['sarem'] = sarem[labeling][ratio]
        df['lbe'] = lbe[labeling][ratio]
        df['ifpnu_dif'] = ifpnu_dif[labeling][ratio]
        # Perform statistical tests
        rankings, statistic, p_value, critical_value, hypothesis = iman_davenport(df, alpha,
                                                                                  minimize=False)
        print('Iman - Davenport test')
        print(hypothesis)
        print(f"Statistic {statistic}, p-value {p_value}")
        print(rankings)

        print('')
        print('')
        print('Wilcoxon paired test')
        test_statistic, p_value = wilcoxon(ifpu[labeling][ratio], sarem[labeling][ratio])
        print(f'Test_statistic_Wilcoxon = {test_statistic},   p_value = {p_value}')
        if p_value < alpha:
            print(
                f'p_value < {alpha}, reject null hypothesis. Data distributions are probably not equal.')
        else:
            print(
                f'p_value > {alpha}, the null hypothesis cannot be rejected. Data distributions are probably equal.')
        print('----------------------------------------------------------\n')


print('\n\n')
print('FRIEDMAN')
print('--------')
for labeling in ['SCAR', 'SAR']:
    for ratio in [0.25, 0.50, 0.75]:
        print(f'TESTING {labeling} {ratio}')
        df = pd.DataFrame()
        df['cases'] = cases
        df['ifpu'] = ifpu[labeling][ratio]
        df['puhrf'] = puhrf[labeling][ratio]
        df['sarem'] = sarem[labeling][ratio]
        df['lbe'] = lbe[labeling][ratio]
        df['ifpnu_dif'] = ifpnu_dif[labeling][ratio]
        statistic, p_value = friedmanchisquare(df['ifpu'], df['puhrf'], df['sarem'], df['lbe'], df['ifpnu_dif'])
        print(f'Statistic {statistic}, p-value {p_value}')

        # Combine three groups into one array
        data = np.array([df['ifpu'], df['puhrf'], df['sarem'], df['lbe'], df['ifpnu_dif']])
        # Conduct the Nemenyi post-hoc test
        print(sp.posthoc_nemenyi_friedman(data.T))
        print('')
