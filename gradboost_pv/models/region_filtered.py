"""Model using geospatial masking"""

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
    build_solar_pv_features,
)
from gradboost_pv.preprocessing.region_filtered import (
    DEFAULT_VARIABLES_FOR_PROCESSING,
    build_local_save_path,
)


def load_local_preprocessed_slice(
    forecast_horizon_step: int,
    variable: str,
    year: int,
    directory: Path = DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
) -> pd.DataFrame:
    """Loads locally preprocessed NWP data from file"""

    inner_fpath, outer_fpath = build_local_save_path(
        forecast_horizon_step, variable, year, directory
    )

    return pd.concat([pd.read_pickle(inner_fpath), pd.read_pickle(outer_fpath)], axis=1)


def load_all_variable_slices(
    forecast_horizon_step: int,
    variables: list[str] = DEFAULT_VARIABLES_FOR_PROCESSING,
    years: list[int] = [2020, 2021],
    directory: Path = DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
) -> pd.DataFrame:
    """Load all preprocessed NWP from file for specific forecast horizon"""
    X = list()
    for variable in variables:
        var_data = list()
        for year in years:
            var_data.append(
                load_local_preprocessed_slice(forecast_horizon_step, variable, year, directory)
            )

        X.append(pd.concat(var_data, axis=0).sort_index(ascending=False))
    X = pd.concat(X, axis=1)
    return X


def build_datasets_from_local(
    processed_nwp_slice: pd.DataFrame,
    national_gsp: xr.Dataset,
    forecast_horizon: np.timedelta64,
    add_noise: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates features for region-masked model.

    Args:
        processed_nwp_slice (pd.DataFrame): Processed NWP data performed at an earlier stage
        national_gsp (xr.Dataset): National GSP PV data
        forecast_horizon (np.timedelta64): forecast horizon for features
        add_noise (bool, optional): Used to identify non-contributing features,
        for model analysis only.
        Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X and y
    """

    processed_nwp_slice = processed_nwp_slice.sort_index(ascending=False)

    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    ).sort_index(ascending=False)

    _within = [col for col in processed_nwp_slice.columns if "within" in col]
    _outer = [col for col in processed_nwp_slice.columns if "outer" in col]
    X_diff = pd.DataFrame(
        data=(processed_nwp_slice[_within].values - processed_nwp_slice[_outer].values),
        columns=[col.replace("_within", "_diff") for col in _within],
        index=processed_nwp_slice.index,
    )

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(
        gsp.index.shift(freq=forecast_horizon).sort_values(ascending=False).values
    )
    _X = pd.DataFrame(_X, index=gsp.index, columns=TRIG_DATETIME_FEATURE_NAMES)
    # Only pick indicies that are in the processed_nwp_slice
    _X = _X.loc[processed_nwp_slice.index]
    X = pd.concat([processed_nwp_slice, X_diff, _X], axis=1).sort_index(ascending=False).dropna()

    solar_variables = build_solar_pv_features(
        gsp.index.shift(freq=forecast_horizon).sort_values(ascending=False)
    )
    solar_variables.index = gsp.index

    # shift y by the step forecast
    y = gsp.shift(freq=-forecast_horizon)

    # add lagged values of GSP PV
    pv_autoregressive_lags = build_lagged_features(gsp, forecast_horizon)
    common_index = solar_variables.index.intersection(pv_autoregressive_lags.index).intersection(
        X.index
    )
    X = X.loc[common_index]
    pv_autoregressive_lags = pv_autoregressive_lags.loc[common_index]
    solar_variables = solar_variables.loc[common_index]
    # Print the index that does exist in X but does not in solar_variables
    # Remove duplicate index from X
    X = X[~X.index.duplicated(keep="first")]
    X = pd.concat(
        [
            X,
            pv_autoregressive_lags,
            solar_variables,
        ],
        axis=1,
    ).dropna()

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    if add_noise:
        # use random noise as a benchmark for uninformative features
        # used only in model analysis/benchmarking ONLY

        noise = pd.DataFrame(columns=["RANDOM_NOISE"], data=np.random.randn(len(X)), index=X.index)

        X = pd.concat([X, noise], axis=1)

    return X, y
