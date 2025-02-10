"""Script to process using Basic downsampling"""

from argparse import ArgumentParser
from pathlib import Path

import xarray as xr

from gradboost_pv.models.utils import (
    DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
    GSP_FPATH,
    NWP_FPATH,
    NWP_STEP_HORIZON,
)
from gradboost_pv.preprocessing.basic import build_local_save_path, bulk_preprocess_nwp
from gradboost_pv.utils.logger import getLogger

logger = getLogger("basic-process-nwp-data")


def parse_args():
    """Parse command line arguments.

    Returns:
        args: Returns arguments
    """
    parser = ArgumentParser(
        description="Script to bulk process NWP xarray data for later use in simple ML model."
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
        help="Directory to save collated data.",
    )
    args = parser.parse_args()
    return args


def main():
    """
    Bulk process NWP data to single pointwise entry.
    """

    args = parse_args()

    gsp = xr.open_zarr(GSP_FPATH)
    nwp = xr.open_zarr(NWP_FPATH)

    evalutation_timeseries = (
        gsp.coords["datetime_gmt"]
        .where(
            (gsp["datetime_gmt"] >= nwp.coords["init_time"].values[0])
            & (gsp["datetime_gmt"] <= nwp.coords["init_time"].values[-1]),
            drop=True,
        )
        .values
    )  # create a subset at same interval as GSP (30 mins) but over the timespan of NWP

    for forecast_horizon in range(NWP_STEP_HORIZON):
        # could be multiprocessed, but I am running overnight anyway
        X = bulk_preprocess_nwp(
            nwp.isel(step=forecast_horizon),
            interpolate=True,
            interpolation_points=evalutation_timeseries,
        )
        fpath = build_local_save_path(forecast_horizon, args.save_dir)
        X.to_pickle(fpath)

        logger.info(f"Completed processing of data for step: {forecast_horizon}")


if __name__ == "__main__":
    main()
