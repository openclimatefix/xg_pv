"""Baseline NWP model with no NWP data"""

from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.utils.utils import trigonometric_datetime_transformation

from gradboost_pv.models.utils import TRIG_DATETIME_FEATURE_NAMES


def build_datasets_from_local(
    national_gsp: xr.Dataset,
    forecast_horizon: np.timedelta64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds covariates and targets for simple model.

    Simple baseline dataset is one with only the
    datetime features of the time of inference.

    Args:
        national_gsp (xr.Dataset): timeseries of gsp power and capacity values
        forecast_horizon (np.timedelta64): shift amount for target

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: returns features and target for model training.
    """

    gsp = pd.DataFrame(
        national_gsp["generation_mw"] / national_gsp["installedcapacity_mwp"],
        index=national_gsp.coords["datetime_gmt"].values,
        columns=["target"],
    ).sort_index(ascending=False)

    # shift y by the step forecast
    y = gsp.shift(freq=forecast_horizon)

    # add datetime methods for the point at which we are forecasting e.g. now + step
    _X = trigonometric_datetime_transformation(gsp.index.values + forecast_horizon)
    _X = pd.DataFrame(_X, index=gsp.index, columns=TRIG_DATETIME_FEATURE_NAMES)

    y = y.reindex(_X.index).dropna()
    _X = _X.loc[y.index]

    return _X, y
