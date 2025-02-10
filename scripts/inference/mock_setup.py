"""Script to simulate data read, model inference and prediction write"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import xarray as xr
from xgboost import XGBRegressor

from gradboost_pv.inference.data_feeds import MockDataFeed
from gradboost_pv.inference.models import Hour, NationalBoostInferenceModel, NationalPVModelConfig
from gradboost_pv.inference.run import MockDatabaseConnection, NationalBoostModelInference
from gradboost_pv.models.s3 import build_object_name, create_s3_client, load_model
from gradboost_pv.models.utils import GSP_FPATH, NWP_FPATH, load_nwp_coordinates

MOCK_DATE_RANGE = slice(np.datetime64("2020-08-02T00:00:00"), np.datetime64("2020-08-04T00:00:00"))
DEFAULT_PATH_TO_MOCK_DATABASE = (
    Path(__file__).parents[2] / "data" / "mock_inference_database.pickle"
)


def parse_args():
    """Parse command line arguments.

    Returns:
        args: Returns arguments
    """
    parser = ArgumentParser(
        description="Script to run mock inference of NationalPV model using data from GCP."
    )
    parser.add_argument(
        "--path_to_database",
        type=Path,
        required=False,
        default=DEFAULT_PATH_TO_MOCK_DATABASE,
    )
    parser.add_argument(
        "--s3_access_key", type=str, required=False, default=None, help="s3 API Access Key"
    )
    parser.add_argument(
        "--s3_secret_key", type=str, required=False, default=None, help="s3 API Secret Key"
    )
    args = parser.parse_args()
    return args


def load_nwp() -> xr.Dataset:
    """TODO - Load in NWP Data for use, remove GCP instances?

    Returns:
        xr.Dataset: NWP Dataset
    """
    return xr.open_zarr(NWP_FPATH)


def load_gsp() -> xr.Dataset:
    """TODO - Load in National PV Data for use, remove GCP instances?

    Returns:
        xr.Dataset: National PV Dataset
    """
    return xr.open_zarr(GSP_FPATH).sel(gsp_id=0)


def create_date_range_slice(
    nwp: xr.Dataset,
    gsp: xr.Dataset,
    datetime_range: slice = MOCK_DATE_RANGE,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Helper function to slice nwp and gsp data on their respective time dimensions.

    Args:
        nwp (xr.Dataset): NWP data
        gsp (xr.Dataset): GSP data
        datetime_range (slice, optional): Defaults to MOCK_DATE_RANGE.

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: Sliced data.
    """
    return nwp.sel(init_time=datetime_range), gsp.sel(datetime_gmt=datetime_range)


def main(path_to_database: Path, model_loader_by_hour: Callable[[Hour], XGBRegressor]):
    """Run mock inference of a NationalBoost Model.

    Args:
        path_to_database (Path): Path for mock local database.
        model_loader_by_hour (Callable[[Hour], XGBRegressor]): Function for loading model by step
    """
    # load data to feed into mock data feed
    nwp, gsp = load_nwp(), load_gsp()
    nwp, gsp = create_date_range_slice(nwp, gsp)

    data_feed = MockDataFeed(nwp, gsp)
    data_feed.initialise()

    # load in our national pv model
    x, y = load_nwp_coordinates()
    config = NationalPVModelConfig(
        "mock_inference",
        overwrite_read_datetime_at_inference=False,
        time_variable_name="init_time",
        nwp_variable_name="variable",
        x_coord_name="x",
        y_coord_name="y",
        gsp_time_variable_name="datetime_gmt",
        gsp_pv_generation_name="generation_mw",
        gsp_installed_capacity_name="installedcapacity_mwp",
    )
    model = NationalBoostInferenceModel(config, model_loader_by_hour, x, y)
    model.initialise()

    # create a mock database to write to
    database_conn = MockDatabaseConnection(path_to_database, overwrite_database=True)

    inference_pipeline = NationalBoostModelInference(model, data_feed, database_conn)
    inference_pipeline.run()

    # open and print the database to console
    database_conn = MockDatabaseConnection(path_to_database, overwrite_database=False)
    database_conn.connect()
    print(database_conn.database.data)


if __name__ == "__main__":
    args = parse_args()

    if args.s3_access_key is None or args.s3_secret_key is None:
        client = create_s3_client()
    else:
        client = create_s3_client(args.s3_access_key, args.s3_secret_key)

    def load_model_by_hour(hour: Hour):
        """Wrapper function for s3 client."""
        return load_model(client, build_object_name(hour))

    main(args.path_to_database, load_model_by_hour)
