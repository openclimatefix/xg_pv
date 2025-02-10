"""Basic Model with single point downsampling"""

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
from gradboost_pv.preprocessing.basic import build_local_save_path


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

    X = processed_nwp_slice.sort_index(ascending=False)

    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    ).sort_index(ascending=False)

    assert pd.infer_freq(gsp.index) == "-30T"

    # shift y by the step forecast
    y = gsp.shift(freq=-forecast_horizon)

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(
        gsp.index.shift(freq=forecast_horizon).sort_values(ascending=False).values
    )
    _X = pd.DataFrame(_X, index=gsp.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    X = pd.concat([X, _X], axis=1).sort_index(ascending=False).dropna()

    pv_autoregressive_lags = build_lagged_features(gsp, forecast_horizon)

    X = pd.concat([X, pv_autoregressive_lags], axis=1).dropna()
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    return X, y
