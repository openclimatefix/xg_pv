"""Script to perform quadrant downsampling of NWP data."""

from argparse import ArgumentParser
from pathlib import Path

import xarray as xr

from gradboost_pv.models.utils import (
    DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
    GSP_FPATH,
    NWP_FPATH,
    NWP_STEP_HORIZON,
)
from gradboost_pv.preprocessing.quadrant_downsample import (
    build_local_save_path,
    bulk_preprocess_nwp,
)
from gradboost_pv.utils.logger import getLogger

logger = getLogger("quadrant-process-nwp-data")


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
    Bulk process NWP data to  a 2x2 grid, taking quadrants.
    """

    args = parse_args()

    gsp = xr.open_zarr(GSP_FPATH)
    nwp = xr.open_zarr(NWP_FPATH)

    evaluation_timeseries = (
        gsp.coords["datetime_gmt"]
        .where(
            (gsp["datetime_gmt"] >= nwp.coords["init_time"].values[0])
            & (gsp["datetime_gmt"] <= nwp.coords["init_time"].values[-1]),
            drop=True,
        )
        .values
    )

    for forecast_horizon in range(NWP_STEP_HORIZON):
        # could be multiprocessed, but I am running overnight anyway
        X = bulk_preprocess_nwp(
            nwp.isel(step=forecast_horizon),
            interpolate=True,
            interpolation_points=evaluation_timeseries,
        )

        X.to_pickle(build_local_save_path(forecast_horizon, args.save_dir))

        logger.info(f"Completed processing of data for step: {forecast_horizon}")


if __name__ == "__main__":
    main()
