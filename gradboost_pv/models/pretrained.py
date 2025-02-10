"""Model built from pre-trained CNN passthrough"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation

from gradboost_pv.models.utils import (
    DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
    TRIG_DATETIME_FEATURE_NAMES,
    build_rolling_linear_regression_betas,
)
from gradboost_pv.preprocessing.pretrained import build_local_save_path

AUTO_REGRESSION_TARGET_LAG = np.timedelta64(1, "h")  # to avoid look ahead bias
AUTO_REGRESSION_COVARIATE_LAG = AUTO_REGRESSION_TARGET_LAG + np.timedelta64(1, "h")


def load_local_preprocessed_slice(
    forecast_horizon_step: int, directory: Path = DEFAULT_DIRECTORY_TO_PROCESSED_NWP
) -> pd.DataFrame:
    """Load local processed NWP data from path

    Args:
        forecast_horizon_step (int): Forecast step slice of NWP data
        directory (Path, optional): Path to data. Defaults to DEFAULT_DIRECTORY_TO_PROCESSED_NWP.

    Returns:
        pd.DataFrame: Processed NWP data.
    """
    return pd.read_pickle(build_local_save_path(forecast_horizon_step, directory))


def build_datasets_from_local(
    X: pd.DataFrame,
    national_gsp: xr.Dataset,
    forecast_horizon: np.timedelta64,
    summarize_buckets: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates features for region-masked model.

    Args:
        X (pd.DataFrame): Processed NWP data performed at an earlier stage
        national_gsp (xr.Dataset): National GSP PV data
        forecast_horizon (np.timedelta64): forecast horizon for features
        summarize_buckets (bool, optional): Used to simplify downsampled data from pretrained model.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X and y
    """

    if summarize_buckets:
        # group by variable, calculate mean and std among the buckets
        # the preprocessing averages the PRETRAINED_OUTPUT_DIMS (~1000) to a lower dimension.
        X.columns = X.columns.str.split("_").str[0]
        group_var = X.groupby(level=0, axis=1)
        mean = group_var.mean(numeric_only=True)
        mean.columns = [f"{col}_mean" for col in mean.columns]
        std = group_var.std(numeric_only=True)
        std.columns = [f"{col}_std" for col in mean.columns]
        X = pd.concat([mean, std], axis=1)

    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    )

    # shift y by the step forecast
    y = gsp.shift(freq=-forecast_horizon).dropna()

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(y.shift(freq=forecast_horizon).index.values)
    _X = pd.DataFrame(_X, index=y.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([X, _X], axis=1)

    # add lagged values of GSP PV
    ar_2 = gsp.shift(freq=np.timedelta64(2, "h"))
    ar_1 = gsp.shift(freq=np.timedelta64(1, "h"))
    ar_day = gsp.shift(freq=np.timedelta64(1, "D"))
    ar_2.columns = ["PV_LAG_2HR"]
    ar_1.columns = ["PV_LAG_1HR"]
    ar_day.columns = ["PV_LAG_DAY"]

    # estimate linear trend of the PV
    pv_covariates = gsp.shift(
        freq=AUTO_REGRESSION_COVARIATE_LAG
    )  # add lag to PV data to avoid lookahead
    pv_target = y.shift(freq=(AUTO_REGRESSION_TARGET_LAG))
    betas = build_rolling_linear_regression_betas(pv_covariates, pv_target)

    X = pd.concat([X, ar_1, ar_day, betas], axis=1).dropna()
    y = y.loc[X.index]

    return X, y
