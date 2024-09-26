"""Script for processing raw NWP data"""
import datetime as dt
import itertools
import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import xarray as xr

from gradboost_pv.models.region_filtered import build_local_save_path
from gradboost_pv.models.utils import (
    DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
    GSP_FPATH,
    NWP_FPATH,
    NWP_STEP_HORIZON,
)
from gradboost_pv.preprocessing.region_filtered import (
    DEFAULT_VARIABLES_FOR_PROCESSING,
    NWPUKRegionMaskedDatasetBuilder,
)
from gradboost_pv.utils.logger import getLogger

logger = getLogger("uk-region-filter-nwp-data")

formatString = (
    "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"  # specify a format string
)
logLevel = logging.DEBUG  # specify standard log level
logging.basicConfig(format=formatString, level=logLevel, datefmt="%Y-%m-%d %I:%M:%S")

logging.getLogger("gcsfs").setLevel(logging.INFO)
logging.getLogger("geopandas").setLevel(logging.INFO)

FORECAST_HORIZONS = range(NWP_STEP_HORIZON)


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


def main(base_save_directory: Path):
    """
    Script to preprocess NWP data, overnight
    """

    logger.info("Loading GSP  data")
    gsp = xr.open_zarr(GSP_FPATH)
    logger.info("Loading NWP data")
    nwp = xr.open_zarr(NWP_FPATH)
    # logger.info("chunking NWP data")
    # nwp = nwp.chunk({"step": 1, "variable": 1, "init_time": 50})

    # if we consider all nwp together the polygon mask is too big to fit in memory
    # instead, iterate over each year in the dataset and save locally

    years = pd.DatetimeIndex(nwp.init_time.values).year.unique().values
    date_years = [dt.datetime(year=year, month=1, day=1) for year in years]

    # add the next year so that we include over the last year in the data
    date_years.append(dt.datetime(year=years[-1] + 1, month=1, day=1))

    for i in range(len(date_years) - 1):
        year = date_years[i].year
        logger.info("Loading NWP data for year %s", year)
        start_datetime, end_datetime = date_years[i], date_years[i + 1]
        _nwp = nwp.sel(init_time=slice(start_datetime, end_datetime))

        # time points to interpolate our nwp data onto.
        logger.info("Loading GSP datetimes for year %s", year)
        evaluation_timeseries = (
            gsp.coords["datetime_gmt"]
            .where(
                (gsp["datetime_gmt"] >= _nwp.coords["init_time"].values[0])
                & (gsp["datetime_gmt"] <= _nwp.coords["init_time"].values[-1]),
                drop=True,
            )
            .values
        )

        logger.debug(f"{evaluation_timeseries=}")
        logger.debug(f"{evaluation_timeseries.shape=}")

        dataset_builder = NWPUKRegionMaskedDatasetBuilder(
            _nwp,
            evaluation_timeseries,
        )

        iter_params = list(itertools.product(DEFAULT_VARIABLES_FOR_PROCESSING, FORECAST_HORIZONS))
        for var, step in iter_params:
            logger.info(f"Processing var: {var}, step: {step}, year: {year}")
            uk_region, outer_region = dataset_builder.build_region_masked_covariates(var, step)

            inner_fpath, outer_fpath = build_local_save_path(
                step, var, year, directory=base_save_directory
            )

            logger.debug("Saving files")
            uk_region.to_pickle(inner_fpath)
            outer_region.to_pickle(outer_fpath)

            logger.info(
                f"({i}/{len(years)}) Completed UK + Outer Region Feature Extraction for "
                f"var: {var}, step: {step}"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args.save_dir)
