"""Generate s3 client, and model configs for testing"""

from pathlib import Path

import pytest
import xarray as xr
from xgboost import XGBRegressor
import numpy as np

from nowcasting_datamodel.fake import make_fake_me_latest

import gradboost_pv
from gradboost_pv.inference.models import NationalPVModelConfig

PATH_TO_TEST_DATA_DIRECTORY = Path(gradboost_pv.__file__).parents[1] / "data" / "test"
PATH_TO_SAMPLE_NWP = PATH_TO_TEST_DATA_DIRECTORY / "sample_prod_nwp.zarr"
PATH_TO_SAMPLE_GSP = PATH_TO_TEST_DATA_DIRECTORY / "sample_prod_gsp.zarr"
MODEL_FORECAST_HORIZON_HOUR = 5
PATH_TO_TEST_MODEL_CONFIG = Path(__file__).parents[1] / "configs" / "test_model_config.yaml"


@pytest.fixture
def mock_model():
    """Loads a model at example forecast horizon"""

    model = XGBRegressor(quantile_alpha=np.array([0.1, 0.5, 0.9]), objective="reg:quantileerror")

    # 33 features
    xtrain = [[1] * 33]
    ytrain = [1]

    model.fit(xtrain, ytrain)

    return model


@pytest.fixture
def model_config() -> NationalPVModelConfig:
    """Loads test model configuration from yaml file."""
    return NationalPVModelConfig.load_from_yaml(PATH_TO_TEST_MODEL_CONFIG)


@pytest.fixture
def model_config_path() -> Path:
    """configuration yaml file."""
    return PATH_TO_TEST_MODEL_CONFIG


@pytest.fixture
def sample_prod_nwp_data() -> xr.Dataset:
    """Loads a sample NWP [x, y] observation from file.

    This data looks like the (processed) NWP data from prod aws.
    Returns:
        xr.Dataset: NWP Observation.
    """
    print(PATH_TO_SAMPLE_NWP)
    nwp = xr.open_zarr(PATH_TO_SAMPLE_NWP)
    return nwp


@pytest.fixture
def sample_prod_gsp_data() -> xr.Dataset:
    """Loads a sample GSP observation from file.

    This data looks like the prod GSP data read from database.
    Returns:
        xr.Dataset: GSP Observations.
    """
    print(PATH_TO_SAMPLE_GSP)
    gsp = xr.open_zarr(PATH_TO_SAMPLE_GSP)
    return gsp


@pytest.fixture()
def me_latest(db_session):
    metric_values = make_fake_me_latest(session=db_session)
    db_session.add_all(metric_values)
    db_session.commit()
