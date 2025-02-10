"""Function to save results to datbase"""

import logging

import pandas as pd
from nowcasting_datamodel.models.convert import convert_df_to_national_forecast
from nowcasting_datamodel.save.save import save
from sqlalchemy.orm import Session

import gradboost_pv
from gradboost_pv.inference.utils import filter_forecasts_on_sun_elevation

logger = logging.getLogger(__name__)


def save_to_database(results_df: pd.DataFrame, session: Session):
    """
    Method to save results to a database
    """

    # TODO fix, wrong units somewhere
    results_df["forecast_mw"] = results_df["forecast_mw"].astype(float)
    results_df["target_datetime_utc"] = pd.to_datetime(results_df["datetime_of_target_utc"])

    # select columns
    results_df.set_index("datetime_of_target_utc", drop=True, inplace=True)
    cols = ["forecast_mw", "forecast_mw_plevel_10", "forecast_mw_plevel_90"]
    results_df = results_df[cols]

    # make all columns are floats
    for c in cols:
        results_df[c] = results_df[c].astype(float)

    # interpolate to every 30 minutes
    results_df = results_df.resample("30T").interpolate(method="linear")

    results_df["target_datetime_utc"] = results_df.index
    logger.debug(results_df[cols])

    forecast_sql = convert_df_to_national_forecast(
        forecast_values_df=results_df,
        session=session,
        model_name="National_xg",
        version=gradboost_pv.__version__,
    )

    # zero out night times
    forecasts = filter_forecasts_on_sun_elevation(forecasts=[forecast_sql])

    save(forecasts=forecasts, session=session, update_national=True, update_gsp=False)
