"""Quadrant downsampling preprocessing"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from gradboost_pv.models.utils import DEFAULT_DIRECTORY_TO_PROCESSED_NWP

DOWNSAMPLE_TO_NUM_POINTS = 2


def build_local_save_path(
    forecast_horizon_step: int, directory: Path = DEFAULT_DIRECTORY_TO_PROCESSED_NWP
) -> Path:
    """Builds filepath based on forecast horizon step.

    Args:
        forecast_horizon_step (int): Forecast step index
        directory (Path, optional): Directory to data.
        Defaults to DEFAULT_DIRECTORY_TO_PROCESSED_NWP.

    Returns:
        Path: Path to processed NWP forecast slice.
    """
    return directory / f"quadrant_nwp_processed_step_{forecast_horizon_step}.pickle"


def _process_nwp(nwp_slice: xr.Dataset) -> xr.Dataset:
    nwp_slice = nwp_slice.coarsen(
        dim={
            "x": int(548 / DOWNSAMPLE_TO_NUM_POINTS),
            "y": int(704 / DOWNSAMPLE_TO_NUM_POINTS),
        },
        boundary="exact",
    ).mean()

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

    # reshape into a square matrix (n_time_points x (4 x variables))
    _X = np.concatenate(
        _X.values.reshape(
            len(variables),
            len(time_points),
            DOWNSAMPLE_TO_NUM_POINTS * DOWNSAMPLE_TO_NUM_POINTS,
        ),
        axis=1,
    )

    # cast to pandas dataframe
    _columns = [
        f"{var}_{idx}"
        for var in variables
        for idx in range(DOWNSAMPLE_TO_NUM_POINTS * DOWNSAMPLE_TO_NUM_POINTS)
    ]
    _X = pd.DataFrame(index=time_points, columns=_columns, data=_X)

    return _X
