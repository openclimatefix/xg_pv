"""Basic NWP Preprocessing"""

from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from gradboost_pv.models.utils import DEFAULT_DIRECTORY_TO_PROCESSED_NWP


def build_local_save_path(
    forecast_horizon_step: int, directory: Path = DEFAULT_DIRECTORY_TO_PROCESSED_NWP
) -> Path:
    """Builds save path based on directory and forecast step index."""

    return directory / f"basic_nwp_preprocessed_step_{forecast_horizon_step}.pickle"


def _process_nwp(nwp_slice: xr.Dataset) -> xr.Dataset:
    """Basic Geospatial averaging. Providing large NWP datasets can cause RAM issues.

    Args:
        nwp_slice (xr.Dataset): Subset of NWP dataset

    Returns:
        xr.Dataset: _description_
    """
    nwp_slice = nwp_slice.mean(dim=["x", "y"])

    return nwp_slice


def bulk_preprocess_nwp(
    nwp_forecast_step_slice: xr.Dataset,
    interpolate: bool = False,
    interpolation_points: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Process NWP data along single forecast horizon axis

    Given slice along forecast horizon of NWP data, downsample
    the data [variables, init_time, x, y] into [variables, init_time, _process_nwp(x, y)]

    Args:
        nwp_forecast_step_slice (xr.Dataset): NWP dataset slice
        interpolate (bool, optional): Whether to interpolate along init_time. Defaults to False.
        interpolation_points (Optional[pd.DatetimeIndex], optional): Defaults to None.

    Returns:
        pd.DataFrame: Processed NWP data.
    """
    if interpolate:
        assert interpolation_points is not None

    variables = nwp_forecast_step_slice.coords["variable"].values
    time_points = (
        interpolation_points if interpolate else nwp_forecast_step_slice.coords["init_time"].values
    )

    nwp_forecast_step_slice = _process_nwp(nwp_forecast_step_slice)

    if interpolate:
        nwp_forecast_step_slice = nwp_forecast_step_slice.interp(
            init_time=interpolation_points, method="linear"
        )

    # calculate transformation
    _X = nwp_forecast_step_slice.sel(variable=variables).to_array().as_numpy()

    # reshape into a square matrix (n_time_points x variables)
    _X = _X.values.reshape(len(variables), len(time_points)).T

    # cast to pandas dataframe
    _X = pd.DataFrame(index=time_points, columns=variables, data=_X)

    return _X
