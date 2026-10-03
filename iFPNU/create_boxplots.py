import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
data_ifpu_scar = {
    "Cover": [0.84, 0.85, 0.86, 0.86, 0.92, 0.52, 0.82, 0.82, 0.79, 0.83],
    "Poker": [0.02, 0.17, 0.01, 0.06, 0.02, 0.01, 0.01, 0.02, 0.14, 0.09],
    "Satellite": [0.55, 0.5, 0.63, 0.61, 0.78, 0.68, 0.77, 0.58, 0.45, 0.61]
}

data_puhrf_scar = {
    "Cover": [0.92, 0.79, 0.51, 0.69, 0.96, 0.87, 0.77, 0.73, 0.77, 0.83],
    "Poker": [0.01, 0.01, 0.07, 0.05, 0.05, 0.05, 0.14, 0.01, 0.09, 0.1],
    "Satellite": [0.59, 0.6, 0.6, 0.44, 0.72, 0.79, 0.66, 0.5, 0.73, 0.52]
}

data_ifpu_sar = {
    "Cover": [0.9, 0.91, 0.16, 0.82, 0.85, 0.92, 0.73, 0.77, 0.62, 0.98],
    "Poker": [0.013, 0.01, 0.012, 0.013, 0.02, 0.01, 0.043, 0.029, 0.013, 0.046],
    "Satellite": [0.69, 0.5, 0.36, 0.51, 0.54, 0.44, 0.79, 0.46, 0.47, 0.44]
}

data_puhrf_sar = {
    "Cover": [0.66, 0.76, 0.71, 0.84, 0.85, 0.75, 0.89, 0.73, 0.61, 0.51],
    "Poker": [0.01, 0.01, 0.01, 0.01, 0.01, 0.015, 0.015, 0.02, 0.009, 0.009],
    "Satellite": [0.63, 0.25, 0.38, 0.25, 0.23, 0.37, 0.52, 0.2, 0.33, 0.26]
}

# Convert to DataFrame
df1_scar = pd.DataFrame(data_ifpu_scar)
df2_scar = pd.DataFrame(data_puhrf_scar)
# Convert to DataFrame for the second plot
df1_sar = pd.DataFrame(data_ifpu_sar)
df2_sar = pd.DataFrame(data_puhrf_sar)

# Number of datasets
num_datasets = len(df1_scar.columns)

# Positions for the box plots
positions1 = np.arange(num_datasets) * 2  # Method 1
positions2 = positions1 + 0.5  # Method 2 (shifted)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
# Titles for the two subplots
titles = ["(a) iFPU vs PUHRF (SCAR 0.25%)", "(b) iFPU vs PUHRF (SAR 0.25%)"]
datasets = [(df1_scar, df2_scar), (df1_sar, df2_sar)]  # Pair the datasets

# Loop to generate two similar box plots for different settings
for i in range(2):
    ax = axes[i]
    # Plot Method 1 (Blue shades)
    ax.boxplot(datasets[i][0].values, positions=positions1, widths=0.4, patch_artist=True,
               boxprops=dict(facecolor='deepskyblue', color='blue'),
               medianprops=dict(color='darkblue'),
               meanline=True, showmeans=True,
               meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='blue', markersize=8))

    # Plot Method 2 (Green shades)
    ax.boxplot(datasets[i][1].values, positions=positions2, widths=0.4, patch_artist=True,
               boxprops=dict(facecolor='lightcoral', color='red'),
               medianprops=dict(color='darkred'),
               meanline=True, showmeans=True,
               meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='red', markersize=8))

    # Labels and customization
    ax.set_xticks((positions1 + positions2) / 2)
    ax.set_xticklabels(df1_sar.columns)  # Centered labels
    ax.set_title(titles[i])
    ax.set_ylabel("PR-AUC")
    ax.grid(True)

# Custom legend
plt.legend(handles=[
    plt.Line2D([0], [0], color='deepskyblue', lw=4, label="iFPU"),
    plt.Line2D([0], [0], color='lightcoral', lw=4, label="PUHRF")
], loc="lower right")

plt.tight_layout()
plt.show()
