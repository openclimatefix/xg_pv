"""Configure and load test NWP/PV data"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_forecasts
from nowcasting_datamodel.models import Base_Forecast, ForecastSQL
from testcontainers.postgres import PostgresContainer

import gradboost_pv

PATH_TO_TEST_DATA_DIRECTORY = Path(gradboost_pv.__file__).parents[1] / "data" / "test"
# data that looks like GCP training data
PATH_TO_NWP_DATA_FOR_PROCESSING = PATH_TO_TEST_DATA_DIRECTORY / "nwp_single_observation.zarr"


@pytest.fixture(scope="session")
def db_connection():
    """Database engine fixture."""
    with PostgresContainer("postgres:16") as postgres:
        # TODO need to setup postgres database with docker
        url = postgres.get_connection_url()

        connection = DatabaseConnection(url=url)
        connection.create_all()
        connection.make_partitions()

        yield connection

        Base_Forecast.metadata.drop_all(connection.engine)


@pytest.fixture(scope="function", autouse=True)
def db_session(db_connection):
    """Creates a new database session for a test."""

    with db_connection.get_session() as s:
        s.begin()
        yield s
        s.rollback()
        s.close()
        s.flush()


@pytest.fixture(scope="session")
def nwp_single_observation() -> xr.Dataset:
    """Loads a sample NWP [x, y] observation from file.

    This data looks like the GCP data used to fit model, not live prod data
    Returns:
        xr.Dataset: NWP Observation.
    """
    # open a single observation of nwp variable at a specific forecast horizon
    nwp = xr.open_zarr(PATH_TO_NWP_DATA_FOR_PROCESSING)

    return nwp


@pytest.fixture(scope="session")
def mock_gsp_data() -> pd.DataFrame:
    """Build Mock PV Dataset

    Returns:
        pd.DataFrame: Mock PV Dataset with 'target' column
    """
    datetimes = pd.DatetimeIndex(
        pd.date_range(
            np.datetime64("2020-02-02T00:00:00"),
            np.datetime64("2020-02-03T00:00:00"),
            freq="30T",
        )
    )

    data = np.arange(0, len(datetimes))
    mock_gsp_data = pd.DataFrame(index=datetimes, data=data, columns=["target"]).sort_index(
        ascending=False
    )
    return mock_gsp_data


@pytest.fixture
def forecasts(db_session) -> List[ForecastSQL]:
    """Make fake forecasts"""
    # create
    f = make_fake_forecasts(gsp_ids=list(range(1, 2)), session=db_session)

    # # add
    db_session.add_all(f)

    return f
