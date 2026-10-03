import tqdm
import os
import numpy as np
import time
import pandas as pd

from data_loaders.utils import load_data_one_year
from data_loaders.metrics import get_ranking_scores

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from pu.fpu_xgb import PUFLLossNonNegative
import xgboost as xgb


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


random_state = 42
use_metadata = True

# path to save results
path_to_save_res = os.path.abspath("./results_misstatement/ifpu_xgb_no_cik.csv")
# path to load data
base_data_dir = os.path.abspath("./data/misstatement_splits")

if use_metadata:
    cat_features = [0, 1]
    drop_nan = True
else:
    cat_features = []
    drop_nan = False

# iterate over splits
results = []
for folder in tqdm.tqdm(os.listdir(base_data_dir)):
    full_path = os.path.join(base_data_dir, folder)
    X_train, X_test, y_train, y_test = load_data_one_year(
        full_path, use_metadata=use_metadata, drop_nan=drop_nan
    )

    loss_prior = y_train.mean()
    time_s = time.time()

    # declare classifier
    clf = Pipeline(
        [
            (
                "tr",
                make_column_transformer(
                    [OneHotEncoder(handle_unknown="ignore"), cat_features],
                    remainder="passthrough",
                ),
            ),
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler(with_mean=False)),
            (
                "clf",
                xgb.XGBClassifier(n_estimators=1000, max_depth=0, tree_method="hist", learning_rate=0.1,
                                  objective=PUFLLossNonNegative(gamma=3, pi_p=loss_prior))
            ),
        ]
    )
    clf.fit(X_train, y_train)
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    time_e = time.time() - time_s

    # Get results
    metrics = get_ranking_scores(y_test, y_test_proba)
    cur_res = {
        "split": int(folder.split("_")[1]),
        "num_train": y_train.shape[0],
        "num_pos_train": y_train.sum(),
        "num_test": y_test.shape[0],
        "num_pos_test": y_test.sum(),
        "time": time_e,
    }
    cur_res.update(metrics)
    results.append(cur_res)

# Save and Print results
df_res = pd.DataFrame(results, columns=cur_res.keys())
df_res.sort_values("split", inplace=True)
df_res.to_csv(
    path_to_save_res,
    index=False,
)

print(f"\niFPU XGB Average scores: \n{df_res[metrics.keys()].mean().to_string()}")
