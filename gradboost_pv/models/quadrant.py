"""Quadrant model - 4 point geospatial downsampling"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation

from gradboost_pv.models.utils import (
    DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
    TRIG_DATETIME_FEATURE_NAMES,
    build_lagged_features,
)
from gradboost_pv.preprocessing.quadrant_downsample import build_local_save_path

AUTO_REGRESSION_TARGET_LAG = np.timedelta64(1, "h")  # to avoid look ahead bias
AUTO_REGRESSION_COVARIATE_LAG = AUTO_REGRESSION_TARGET_LAG + np.timedelta64(1, "h")


def load_local_preprocessed_slice(
    forecast_horizon_step: int, directory: Path = DEFAULT_DIRECTORY_TO_PROCESSED_NWP
) -> pd.DataFrame:
    """Loads pickled NWP DataFrame."""
    return pd.read_pickle(build_local_save_path(forecast_horizon_step, directory))


def build_datasets_from_local(
    processed_nwp_slice: pd.DataFrame,
    national_gsp: xr.Dataset,
    forecast_horizon: np.timedelta64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates features for region-masked model.

    Args:
        processed_nwp_slice (pd.DataFrame): Processed NWP data performed at an earlier stage
        national_gsp (xr.Dataset): National GSP PV data
        forecast_horizon (np.timedelta64): forecast horizon for features

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X and y
    """

    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    )

    # shift y by the step forecast
    y = gsp.shift(freq=-forecast_horizon)

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(
        gsp.index.shift(freq=forecast_horizon).sort_values(ascending=False).values
    )
    _X = pd.DataFrame(_X, index=y.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([processed_nwp_slice, _X], axis=1)

    # add lagged values of GSP PV
    pv_autoregressive_lags = build_lagged_features(gsp, forecast_horizon)

    X = pd.concat([X, pv_autoregressive_lags], axis=1).dropna()
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    return X, y
