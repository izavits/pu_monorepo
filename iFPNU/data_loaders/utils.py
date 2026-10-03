import pandas as pd
import numpy as np
import os

FINANCIAL_FEATURES: list[str] = [
    "act",
    "ap",
    "at",
    "ceq",
    "che",
    "cogs",
    "csho",
    "dlc",
    "dltis",
    "dltt",
    "dp",
    "ib",
    "invt",
    "ivao",
    "ivst",
    "lct",
    "lt",
    "ni",
    "ppegt",
    "pstk",
    "re",
    "rect",
    "sale",
    "sstk",
    "txp",
    "txt",
    "xint",
    "prcc_f",
    "dch_wc",
    "ch_rsst",
    "dch_rec",
    "dch_inv",
    "soft_assets",
    "ch_cs",
    "ch_cm",
    "ch_roa",
    "issue",
    "bm",
    "dpi",
    "reoa",
    "EBIT",
]


METADATA_TO_USE: list[str] = [
    #"CIK",
    "SIC",
    "State of Inc",
]


def load_data_one_year(
    path_to_file: str,
    use_metadata: bool = False,
    drop_nan: bool = False,
    target_col: str = "Misstatement",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loader of train, test instances from original split by Ilias.

    Args:
        path_to_file (str): Path to split folder (i.e. split_2010)
        use_metadata (bool, optional): Whether to keep company metadata. Defaults to False.
        drop_nan (bool, optional): Whether to drop any sample with nan data. Defaults to False.
        target_col (str, optional): Target column to use. Defaults to "Misstatement".
    Returns:
        [np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """

    df_train = pd.read_csv(os.path.join(path_to_file, "train_instances.csv"))
    df_train[["CIK", "SIC"]] = df_train[["CIK", "SIC"]].astype(int)

    df_test = pd.read_csv(os.path.join(path_to_file, "test_instances.csv"))
    df_test[["CIK", "SIC"]] = df_test[["CIK", "SIC"]].astype(int)

    to_keep = FINANCIAL_FEATURES
    if use_metadata:
        to_keep = METADATA_TO_USE + FINANCIAL_FEATURES

    train = df_train[to_keep + [target_col]]
    test = df_test[to_keep + [target_col]]
    if drop_nan:
        train = train.dropna()
        test = test.dropna()
    X_train = train[to_keep].values
    y_train = train[target_col].values.astype(int)

    X_test = test[to_keep].values
    y_test = test[target_col].values.astype(int)

    return X_train, X_test, y_train, y_test