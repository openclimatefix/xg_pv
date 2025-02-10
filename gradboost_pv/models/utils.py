"""Model utilities"""

import logging
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
from pvlib.location import Location

logger = logging.getLogger(__name__)

NWP_VARIABLE_NUM = 17
NWP_STEP_HORIZON = 37
# GCP paths for nwp and gsp data
NWP_FPATH = "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr/"
GSP_FPATH = "gs://solar-pv-nowcasting-data/PV/GSP/v5/pv_gsp.zarr"
DEFAULT_DIRECTORY_TO_PROCESSED_NWP = Path(__file__).parents[2] / "data"


ORDERED_NWP_FEATURE_VARIABLES = [
    "cdcb",
    "lcc",
    "mcc",
    "hcc",
    "sde",
    "hcct",
    "dswrf",
    "dlwrf",
    "h",
    "t",
    "r",
    "dpt",
    "vis",
    "si10",
    "wdir10",
    "prmsl",
    "prate",
]


TRIG_DATETIME_FEATURE_NAMES = [
    "SIN_MONTH",
    "COS_MONTH",
    "SIN_DAY",
    "COS_DAY",
    "SIN_HOUR",
    "COS_HOUR",
]

DEFAULT_ROLLING_LR_WINDOW_SIZE = 10
LATITUDE_UK_SOUTH_CENTER = 52.80111
LONGITUDE_UK_SOUTH_CENTER = -1.0967
DEFAULT_UK_SOUTH_LOCATION = Location(LATITUDE_UK_SOUTH_CENTER, LONGITUDE_UK_SOUTH_CENTER)
PATH_TO_LOCAL_NWP_COORDINATES = Path(__file__).parents[2] / "data" / "nwp_grid_coordinates.npz"


def save_nwp_coordinates(x: np.ndarray, y: np.ndarray):
    """Convenience method for saving nwp grid coordiantes locally"""
    np.savez(PATH_TO_LOCAL_NWP_COORDINATES, x=x, y=y)


def load_nwp_coordinates(
    path: Path = PATH_TO_LOCAL_NWP_COORDINATES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience method for loading nwp grid coordinates locally.

    Args:
        path (Path, optional): Defaults to PATH_TO_LOCAL_NWP_COORDINATES.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y coordinates respectively.
    """
    logger.debug("Loading x and y coords")
    coords = np.load(path)
    return coords["x"], coords["y"]


def clipped_univariate_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    upper_clip: float = 10,
    lower_clip: float = -10,
    epsilon: float = 0.01,
) -> float:
    """Simple univariate regression, with values clipped.

    Args:
        X (np.ndarray): single covariate
        y (np.ndarray): regression target
        upper_clip (float, optional): clip values above this threshold. Defaults to 10.
        lower_clip (float, optional): clip values below this threshold. Defaults to -10.
        epsilon (float, optional): regularisation term. Defaults to 0.01.

    Returns:
        float: _description_
    """
    return max(min((1 / ((X.T @ X) + epsilon)) * (X.T @ y), upper_clip), lower_clip)


def build_rolling_linear_regression_betas(
    X: Union[pd.Series, pd.DataFrame],
    y: Union[pd.Series, pd.DataFrame],
    window_size: int = DEFAULT_ROLLING_LR_WINDOW_SIZE,
    regression_function: Callable[
        [np.ndarray, np.ndarray], float
    ] = clipped_univariate_linear_regression,
) -> pd.Series:
    """Performs a rolling time-series of univariate regressions.

    returns a time-series of _betas_, where _beta_ is the following:
                        y = _beta_ * x
    in our rolling regressions.

    Args:
        X (Union[pd.Series, pd.DataFrame]): time-series of 1-D variable
        y (Union[pd.Series, pd.DataFrame]): time-series of 1-D target
        window_size (int, optional): size of rolling window.
            Defaults to DEFAULT_ROLLING_LR_WINDOW_SIZE.
        regression_function (Callable[[np.ndarray, np.ndarray], float]): Function to
            regress x and y.

    Returns:
        pd.Series: time-series of the regression coefficients
    """

    assert len(X) == len(y)

    betas = np.nan * np.empty(len(y))
    for n in range(window_size, len(y)):
        _x, _y = (
            X.iloc[(n - window_size) : n].values,
            y.iloc[(n - window_size) : n].values,
        )
        _beta = regression_function(_x, _y)
        betas[n] = _beta

    return pd.Series(data=betas, index=y.index, name="LR_Beta")


def build_solar_pv_features(
    times_of_forecast: pd.DatetimeIndex, location: Location = DEFAULT_UK_SOUTH_LOCATION
) -> pd.DataFrame:
    """Build PV/Solar features given times and a location.

    Args:
        times_of_forecast (pd.DatetimeIndex): times at which to compute solar data.
        location (Location, optional): Location for computation.
        Defaults to DEFAULT_UK_SOUTH_LOCATION.

    Returns:
        pd.DataFrame: A DataFrame with various solar position / clear sky features.
        Indexed by forecast times
    """
    clear_sky = location.get_clearsky(times_of_forecast)[["ghi", "dni"]]
    solar_position = location.get_solarposition(times_of_forecast)[
        ["zenith", "elevation", "azimuth", "equation_of_time"]
    ]
    solar_variables = pd.concat([clear_sky, solar_position], axis=1)
    return solar_variables


def build_lagged_features(gsp: pd.DataFrame, forecast_horizon: np.timedelta64) -> pd.DataFrame:
    """Builds AR lagged features using the most recent day's information.

    Builds 1HR, 2HR and 1DAY lagged features for the most recent data of the same
    point in the day.
    For example, if it is 1pm now and we forecast 5 hours ahead, the 1DAY lag is yesterday at 6pm.
    If we forecast 36 hours ahead, the 1DAY lag is 1am today.

    Args:
        gsp (pd.DataFrame): national gsp data, indexed by time at 30 minute intervals
        forecast_horizon (np.timedelta64): forecast amount

    Returns:
        pd.DataFrame: DataFrame with 1DAY, 1HR and 2HR lagged values
    """
    assert pd.infer_freq(gsp.index) == "-30T"

    NUM_GSP_OBS_BETWEEN_FORECAST = int(forecast_horizon / np.timedelta64(30, "m"))

    ar_day_lag = gsp.shift(
        freq=np.timedelta64(int(24 - ((NUM_GSP_OBS_BETWEEN_FORECAST / 2) % 24)), "h")
    )

    ar_2_hour_lag = gsp.shift(
        freq=np.timedelta64(int(24 - ((NUM_GSP_OBS_BETWEEN_FORECAST / 2 - 2) % 24)), "h")
    )

    ar_1_hour_lag = gsp.shift(
        freq=np.timedelta64(int(24 - ((NUM_GSP_OBS_BETWEEN_FORECAST / 2 - 1) % 24)), "h")
    )

    pv_autoregressive_lags = (
        pd.concat([ar_day_lag, ar_1_hour_lag, ar_2_hour_lag], axis=1)
        .sort_index(ascending=False)
        .dropna()
    )
    pv_autoregressive_lags.columns = ["PV_LAG_DAY", "PV_LAG_1HR", "PV_LAG_2HR"]

    return pv_autoregressive_lags.reindex(gsp.index).dropna()
